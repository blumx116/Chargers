from math import ceil
from typing import List, Union, Tuple, NamedTuple
import warnings

import gym
from gym.spaces import Space
import numpy as np
from numpy.random import RandomState
from sklearn.preprocessing import normalize

from data_structures import PriorityQueue
from env.simulation_events import ArrivalEvent, QueryEvent

Action = int # [range: 0 -> n_stations-1]
Reward = np.ndarray # np.ndarray[float32] : [n_stations,]
SimulationState = NamedTuple(
    "SimulationState",
    [('station_idx', np.ndarray),  # np.ndarray[f32] : [n_stations, 1]
    ('station_locations', np.ndarray),  # np.ndarray[float32] : [n_stations, 2] => (x, y)
     ("station_occs", np.ndarray),  # np.ndarray[int32] : [n_stations, 1]
     ("station_maxes", np.ndarray), # np.ndarray[int32] : [n_stations, 1]
     ("car_locs", np.ndarray),  # np.ndarray[float32]: [n_stations 2 => (x, y)
     ("car_dest_idx", np.ndarray),  # np.ndarray[int32] : [n_stations, 1]
     ("car_dest_loc", np.ndarray),  # np.ndarray[float32] : [n_stations, 2] => (x, y)
     ("t", np.ndarray),  # np.ndarray[int32] : (1, 1) => self.t in range[0, max_t]
     ('query_loc', np.ndarray),  # np.ndarray[float32] : [1, 2] => (x,y)
     ('remaining_queries', np.ndarray)  # np.ndarray[int32] : [1, 1] => num_remaining_queries
     ])
State = SimulationState

def get_distances(
        query: np.ndarray,
        stations: np.ndarray) -> np.ndarray:
    return np.linalg.norm(stations - query, axis=1)


class ContinuousSimulationEngine:
    def __init__(self,
            arrivals: List[List[ArrivalEvent]],
            queries: List[List[QueryEvent]],
            departures: List[List[int]],
            initial_station_state: np.ndarray,
            initial_car_state: np.ndarray,
            station_info: np.ndarray,
            car_speed: float,
            logging: bool = False,
            **kwargs):
        """

        :param arrivals: List[List[ArrivalEvents]]: [max_t, ]
            list of arrivals at each timestep
        :param queries: List[List[QueryEvent]]: [max_ t, ]
            list of queries at each timestep
        :param departures: List[List[int]]: [max_t, ]
            list of indices of stations being departed from at each
            timestep. Multiple cars may depart from same station at
            one timestep
        :param initial_station_state:  np.ndarray[int8] : [n_stations, 1]
            initial number of occupied slots at stations
        :param initial_car_state: np.ndarray[float32] : [max_cars, 7]
            idx => (used, car_x, car_y, station_idx, station_x, station_y,
                duration)
            initial locations and destinations of cars by rows
            used should be 1 if row has data and 0 if empty
        :param station_info: np.ndarray[float32] : [n_stations, 3]
            idx => (x, y, max_occupancy)
            information about each station, indexed by rows
        :param car_speed: float
            how far each car moves towards destination at each timestep
        :param logging: bool = False
            whether or not to store a summary to generate
        :param kwargs: for compatibility
        """
        assert len(arrivals) == len(queries)
        assert np.all(station_info[:, 2] > 0)  # all stations have spots

        self.car_speed: float = car_speed

        self.arrivals: List[List[ArrivalEvent]] = arrivals
        self.queries: List[List[QueryEvent]] = queries
        self.departures: List[List[int]] = departures
        self.station_state: np.ndarray = initial_station_state[:, np.newaxis]
        self.car_state: np.ndarray = initial_car_state
        self.station_info: np.ndarray = station_info
        self.logging = logging

        self.open_car_indices: PriorityQueue[int] = PriorityQueue()
        for idx in reversed(range(self.car_state.shape[0])):
            if not np.any(self.car_state[idx, :]):  # all 0
                self.open_car_indices.push(idx, idx)

        # these variables should stay static, only for tracking
        # and clarity of code
        self.total_arrivals: int = sum(map(len, arrivals))
        self.total_queries: int = sum(map(len, queries))
        self.max_cars: int = self.car_state.shape[0]
        self.n_stations: int = self.station_state.shape[0]
        self.total_spots: int = np.sum(initial_car_state[:, 2])
        self.max_t: int = len(arrivals)

        if self.max_cars < self.total_queries + self.total_arrivals:
            # warnings.warn("Warning: Fill up of car slots possible")
            pass

        self.t: int = 0
        self._cur_reward: np.ndarray = self._zero_reward_()
        self._summary_ = {
            'distances travelled': [],
            'failed dispatches': 0,
            'timesteps travelled': [],
            'nearest distances': [],
            'organic fails': 0,
            'original queries': list(map(len, self.queries)),
            'actual queries': [0 for _ in range(self.max_t)]
        }

    def done(self) -> bool:
        """
        We are done when all of the time has elapsed
        :return: True if we are at or past the end of the simulation
        False otherwise
        """
        return self.t >= self.max_t

    def info(self) -> int:
        """
        Currently just returns the number of queries remaining to be handled
        at this timestep.
        If 0, can safely pass None as an action to step to proceed to next
        timestep
        :return: remaining number of unhandled queries
        """
        return len(self._cur_queries_())

    def reward(self) -> Reward:
        return self._cur_reward.copy()

    def state(self) -> State:
        msk: np.ndarray = self.car_state[:, 0].astype(bool)
        # np.ndarray[bool] : [max_cars, ]
        if len(self._cur_queries_()) > 0:
            query: QueryEvent = self._cur_queries_()[0]
            query_loc: np.ndarray = np.asarray([[query.x, query.y]]).astype(np.float32)
        else:
            query_loc: np.ndarray = np.zeros(shape=(1,2), dtype=np.float32)
        return State(
            station_idx=np.arange(0, self.n_stations).reshape((-1, 1)).astype(np.int32),
            station_locations=self.station_info[:, :2].copy(),
            station_occs=self.station_state.copy(),
            station_maxes=self.station_info[:, 2, np.newaxis].copy(),
            car_dest_idx=self.car_state[msk, 3, np.newaxis].copy(),
            car_locs=self.car_state[msk, 1:3].copy(),
            car_dest_loc=self.car_state[msk, 4:6].copy(),
            t=np.asarray([[self.t]]).astype(np.int32),
            query_loc=query_loc,
            remaining_queries=np.asarray([[len(self._cur_queries_())]]).astype(np.int32))

    def step(self,
             action: Action = None) -> Tuple[State, Reward, bool, int]:
        self._cur_reward = self._zero_reward_()
        if len(self._cur_queries_()) > 0:
            self._refer_car_(action)
            # note, this will reduce the number of queries by one
        if len(self._cur_queries_()) == 0:
            # we can answer all current queries and timestep forward
            self._time_forward_()
            self._process_departures_()
            self.t += 1
        return self.state(), self.reward(), self.done(), self.info()

    def summary(self):
        return self._summary_

    def _cur_queries_(self) -> List[QueryEvent]:
        if self.done():
            return []
        return self.queries[self.t]

    def _cur_arrivals_(self) -> List[ArrivalEvent]:
        if self.done():
            return []
        return self.arrivals[self.t]

    def _cur_departures_(self) -> List[int]:
        if self.done():
            return []
        return self.departures[self.t]

    def _move_cars_(self) -> None:
        msk: np.ndarray = self.car_state[:, 0].astype(bool)
        # np.ndarray[bool] : [max_cars,]
        # cur_n_cars = sum(msk)
        if not np.any(msk):
            # no cars to move
            return
        cur_locs: np.ndarray = self.car_state[msk, 1:3]
        dest_locs: np.ndarray = self.car_state[msk, 4:6]
        # both np.ndarray[float32] : [cur_n_cars, 2] => (x, y)
        directions: np.ndarray = dest_locs - cur_locs
        directions: np.ndarray = normalize(directions, axis=1, norm='l2')
        # np.ndarray[float32]: [cur_n_cars, 2] => (dx, dy) unit vecs
        self.car_state[msk, 1:3] += self.car_speed * directions
        dest_station_idxs: np.ndarray = self.car_state[msk, 3].astype(np.int8)
        # np.ndarray[int8] : [cur_n_cars]
        for idx in dest_station_idxs:
            self._cur_reward[idx] -= self.car_speed

    def _process_natural_arrivals_(self) -> None:
        for arrival in self._cur_arrivals_():
            arrived: bool = self._process_arrival_(arrival.idx, arrival.duration)
            if self.logging and not arrived:
                self._summary_['organic fails'] += 1

    def _process_arrival_(self,
            station_idx: int,
            duration: int) -> bool:
        """
        :param station_idx: idx of the station the car is arriving at
        :param duration: how long the car stays
        :return: whether or not the car found a spot
        """
        max_station_occupancy: int = self.station_info[station_idx, 2]
        cur_station_occupancy: int = self.station_state[station_idx]
        if cur_station_occupancy < max_station_occupancy:
            self.station_state[station_idx] += 1
            departure_time: int = self.t + duration
            if departure_time < self.max_t:
                self.departures[departure_time].append(station_idx)
            return True
        else:
            if self.t < self.max_t - 1:
                self.queries[self.t + 1].append(
                    QueryEvent(
                        x=self.station_info[station_idx, 0],
                        y=self.station_info[station_idx, 1],
                        duration=duration,
                        og_distance=0))
            return False

    def _process_departures_(self) -> None:
        departures: List[int] = self._cur_departures_()
        for departure in departures:
            assert self.station_state[departure] > 0
            self.station_state[departure] -= 1

    def _process_referred_arrivals_(self) -> None:
        msk: np.ndarray = self.car_state[:, 0].astype(bool)
        # np.ndarray[bool] : [max_cars,]
        # cur_n_cars = sum(msk)
        if sum(msk) == 0:
            return
        cur_locs: np.ndarray = self.car_state[msk, 1:3]
        dest_locs: np.ndarray = self.car_state[msk, 4:6]
        # both np.ndarray[float32] : [cur_n_cars, 2] => (x, y)
        distances: np.ndarray = np.linalg.norm(dest_locs - cur_locs, axis=1)
        # np.ndarray[float32] : [cur_n_cars, 1] => dist
        new_arrivals: np.ndarray = distances <= self.car_speed
        # np.ndarray[bool] : [cur_n_cars, ]
        # n_cars_arriving = sum(new_arrivals)
        # arrivals are cars that can move to their destination w/in 1 tstep
        station_indices: np.ndarray = self.car_state[msk, 3][new_arrivals]
        station_indices = station_indices.astype(np.int32)
        # np.ndarray[np.int32] : [n_cars_arriving, ]
        durations: np.ndarray = self.car_state[msk, -1][new_arrivals]
        durations = durations.astype(np.int32)
        # np.ndarray[np.int32] : [durations, ]
        for idx, duration in zip(station_indices, durations):
            # process all cars that are arriving
            if self._process_arrival_(idx, duration):
                self._cur_reward[idx] += 3
            else:
                self._summary_['failed dispatches'] += 1
                # give 3 reward whenever we have a car successfully arrive
        indices: np.ndarray = np.flatnonzero(msk)
        # list of nonzero indices
        indices = indices[np.flatnonzero(new_arrivals)]
        # gives indices that both had a car and arrived
        self.car_state[indices, :] = 0
        # set all of the data spots for cars that just arrived to 0

        # this is ugly, but we need to put all of the indices back in the pool
        processed_indices: np.ndarray = np.nonzero(msk)[0]
        # np.ndarray[int8] : [cur_n_cars]
        for idx, arrived in zip(processed_indices, new_arrivals):
            if arrived:
                self.open_car_indices.push(idx, idx)

    def _refer_car_(self, referral: Action) -> None:
        query: QueryEvent = self._cur_queries_().pop()
        idx: int = self.open_car_indices.pop()

        station_x, station_y = self.station_info[referral, :2]

        self.car_state[idx, :] = np.asarray(
            [1, query.x, query.y,
             referral, station_x, station_y,
             query.duration])

        if self.logging:
            query_loc = np.asarray([query.x, query.y], dtype=np.float32)
            station_loc = np.asarray([station_x, station_y], dtype=np.float32)
            distance: float = np.linalg.norm(query_loc - station_loc)
            self._summary_['distances travelled'].append(distance)
            self._summary_['timesteps travelled'].append(ceil(distance / self.car_speed))
            self._summary_['actual queries'][self.t] += 1

            all_distances: np.ndarray = get_distances(query_loc, stations=self.state().station_locations)
            min_distance: float = np.min(all_distances)
            self._summary_['nearest distances'].append(min_distance)

    def _zero_reward_(self) -> Reward:
        return np.zeros(self.n_stations, dtype=np.float32)

    def _time_forward_(self) -> None:
        self._process_natural_arrivals_()
        self._process_referred_arrivals_()
        self._move_cars_()
