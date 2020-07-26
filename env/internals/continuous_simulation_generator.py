from math import ceil
from typing import Union, List, Tuple
from copy import deepcopy

import numpy as np
from numpy.random import RandomState

from env.internals.continuous_simulation_engine import ContinuousSimulationEngine
from env.simulation_events import ArrivalEvent, QueryEvent, Arrivals, Queries
from misc.utils import optional_random, array_shuffle, kwargify


class ContinuousSimulationGenerator:
    def __init__(self,
            arrivals: Arrivals,
            departures: List[List[int]],
            initial_station_state: np.ndarray,
            station_info: np.ndarray,
            sample_distance: float,
            sample_amount: float,
            max_cars: int,
            car_speed: float,
            **kwargs):
        """

        :param arrivals: List[List[ArrivalEvents]]: [max_t, ]
            list of arrivals at each timestep. Note: some of these will
            becomes queries
        :param departures: List[List[int]]
            list of indices of each station with departing cars at each timestep
            Note: multiple cars may depart from the same station at the same tstep
            Note: this only lists departures that are not associated with an arrival
        :param initial_station_state:  np.ndarray[int8] : [n_stations, 1]
            initial number of occupied slots at stations
        :param station_info: np.ndarray[float] : [n_stations, 3]
            idx => (x, y, max_occupancy)
            information about each station, indexed by rows
        :param sample_distance: float
            samples are generated around arrival event
                with range Normal(0, sample_distance)
        :param sample_amount: Union[float, int]
                int => generates this many queries
                float => turns this percentage of arrivals in to queries
        :param max_cars: int
                the number of slots to allocate for holding car data
        :param car_speed: float
            how far each car moves towards destination at each timestep
        :param kwargs: for compatibility
        """
        kwargs = kwargify(locals())
        self.arrivals: Arrivals = arrivals
        self.departures: List[List[int]] = departures
        self.initial_station_state: np.ndarray = initial_station_state
        self.station_info: np.ndarray = station_info
        self.sample_distance: int = sample_distance
        self.sample_amount: Union[int, float] = sample_amount
        self.car_speed: float = car_speed
        self.max_cars: int = max_cars
        self.random: RandomState = optional_random()
        self.kwargs = kwargs

    def generate(self,
            logging: bool = False) -> ContinuousSimulationEngine:
        arrivals, query_protos = self._base_query_arrival_split_()
        queries: Queries = self._to_queries_(query_protos)
        queries = self._adjust_query_time_(queries, query_protos)
        initial_car_state: np.ndarray = self._make_initial_cars_()
        kwargs = self.kwargs.copy()
        kwargs.update({
            'arrivals': arrivals,
            'queries': queries,
            'departures' : deepcopy(self.departures),
            'initial_station_state' : self.initial_station_state.copy(),
            'initial_car_state' : initial_car_state.copy(),
            'station_info': self.station_info.copy(),
            'logging': logging
        })
        return ContinuousSimulationEngine(**kwargs)

    def _adjust_query_time_(self,
            queries: Queries,
            query_protos: Arrivals) -> Queries:
        """

        :param queries:
        :param query_protos:
        :return:
            queries: List[List[QueryEvent]]
                the queries, at the correct times
            prev_queries: List[List[QueryEvent]]
                the queries that occur before time t0 -> last element is t0-1
            prev_query_protos: List[List[ArrivalEvent]]
                the protos corresponding to prev_queries -> used by
                '_make_initial_cars'
        """
        result: Queries = [[] for _ in range(len(queries))]
        assert len(queries) == len(query_protos)
        for t in range(len(queries)):
            for query, proto in zip(queries[t], query_protos[t]):
                car_loc: np.ndarray = np.asarray([query.x, query.y], dtype=np.float32)
                station_loc: np.ndarray = self.station_info[proto.idx, :2]
                # both np.ndarray[float32] => (x, y)
                dxdy: np.ndarray = station_loc - car_loc
                distance: float = np.linalg.norm(dxdy)
                tsteps_to_arrive: int = int(ceil(distance))
                time_of_query: int = t - tsteps_to_arrive
                # NOTE: this means that anything that was queried before the start
                # of the day moves to the end of this day
                result[time_of_query].append(query)
        return result

    def _arrival_to_query_(self,
            arrival: ArrivalEvent) -> QueryEvent:
        loc: np.ndarray = self.station_info[arrival.idx, :2]
        # np.ndarray[float32] => (x, y)
        dxdy: np.ndarray = self.random.normal(
            loc=0,
            scale=self.sample_distance,
            size=(2,))
        # np.ndarray[float32] => (dx, dy)
        new_x, new_y = loc + dxdy
        distance: float = np.linalg.norm(dxdy)
        return QueryEvent(new_x, new_y, arrival.duration, distance)

    def _base_query_arrival_split_(self) -> Tuple[Arrivals, Arrivals]:
        """
        Randomly splits the arrival events in to events that will be
        treated as arrivals and ones that will be converted to queries
        :return:
            arrivals: List[List[ArrivalEvent]]
                list of arrivals for each timestep
            queries: List[List[ArrivalEvent]]
                list of arrivals to be converted to queries foreach timestep
        """
        num_queries: List[int] = self._get_num_queries_by_tstep_()
        arrivals: Arrivals = []
        queries: Arrivals= []

        for t in range(len(self.arrivals)):
            shuffled = array_shuffle(self.arrivals[t], self.random)
            queries.append(shuffled[:num_queries[t]])
            arrivals.append(shuffled[num_queries[t]:])
        return arrivals, queries

    def _get_num_queries_by_tstep_(self) -> List[int]:
        arrival_counts: List[int] = list(map(len, self.arrivals))
        arrival_counts: np.ndarray = np.asarray(arrival_counts, dtype=np.int32)
        total_count: int = sum(arrival_counts)
        if isinstance(self.sample_amount, int):
            # integer means we give exactly that many queries
            assert self.sample_amount <= np.sum(arrival_counts)
            result: List[int] = [0] * len(arrival_counts)
            for _ in range(self.sample_amount):
                idx: int = self.random.choice(
                    np.arange(len(arrival_counts)),
                    p=arrival_counts / total_count)
                result[idx] += 1
                arrival_counts[idx] -= 1  # can't have more than
                total_count -= 1
            return result
        elif isinstance(self.sample_amount, float):
            # float means that percentage of arrivals turn in to queries
            assert self.sample_amount <= 1
            result: List[int] = []
            for i in range(len(arrival_counts)):
                result.append(self.random.binomial(arrival_counts[i], p=self.sample_amount))
            return result
        else:
            raise TypeError(f"self.sample_amount should be either float or int. Got: {type(self.sample_amount)}")

    def _make_initial_cars_(self) -> np.ndarray:
        """
        Makes an empty array to hold car data
        :return: np.ndarray[float32] : [max_cars, 7] all 0
        """
        return np.zeros((self.max_cars, 7), dtype=np.float32)

    def _to_queries_(self,
            query_protos: Arrivals) -> Queries:
        """
        Transforms the arrivals that should be replaced with queries in to
        queries by randomly sampling around the station location
        :param query_protos: List[List[ArrivalEvent]]
            arrivals to be replaced with queries
        :return: queries: List[List[QueryEvent]]
            the corresponding queries
        """
        return list(map(
            lambda tstep: list(map(
                self._arrival_to_query_,
                tstep)),
            query_protos))

    def seed(self,
            random: Union[int, RandomState] = None) -> None:
        self.random = optional_random(random)
