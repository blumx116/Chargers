from typing import Union, Tuple, Optional, List

import gym
from gym.spaces import Space, Dict, Discrete, Box
import numpy as np
from numpy.random import RandomState
import pandas as pd

from env.internals.continuous_simulation_engine import (
    ContinuousSimulationEngine, Reward, Action, State)
from env.internals.continuous_simulation_generator import ContinuousSimulationGenerator
from env.internals.simulation_helper import (
    _load_changes, _get_arrival_events, _get_departure_events)
from env.simulation_events import Arrivals, ArrivalEvent
from misc.config import Config


class ContinuousSimulation(gym.Env):
    def __init__(self,
                 arrivals: Arrivals,
                 departures: List[List[int]],
                 initial_station_state: np.ndarray,
                 station_info: np.ndarray,
                 config: Config):
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
        :param config:other configurations for the simulation
            config.sample_distance: float => samples are generated around arrival event
                with range Normal(0, sample_distance)
            config.sample_amount: Union[float, int]
                int => generates this many queries
                float => turns this percentage of arrivals in to queries
            config.max_cars: int
                the number of slots to allocate for holding car data
            config.car_speed: float => how far each car moves towards destination
                at each timestep
        """
        self.generator: ContinuousSimulationGenerator = ContinuousSimulationGenerator(
            arrivals,
            departures,
            initial_station_state,
            station_info,
            config)

        self.engine: ContinuousSimulationEngine = ...
        self.observation_space: Space = ...
        self.action_space: Space = ...
        self.reward_range: Tuple[float, float] = ...
        self.config: Config = config

        self.reset()

    def reset(self):
        self.engine = self.generator.generate()
        n_stations: int = self.engine.n_stations
        self.max_cars: int = self.engine.max_cars
        self.max_occ: int = np.max(self.engine.station_info[:, 2])
        self.observation_space = Dict({
            'station_locations':
                Box(0, np.inf, (n_stations, 2), dtype=np.float32),
            'station_occs':
                Box(0, self.max_occ, (n_stations, 1), dtype=np.int32),
            'station_maxes':
                Box(0, self.max_occ, (n_stations, 1), dtype=np.int32),
            'car_locs':
                Box(0, np.inf, (self.max_cars, 2), dtype=np.float32),
            'car_dest_idx':
                Box(0, n_stations, (self.max_cars, 1), dtype=np.int32),
            'car_dest_loc':
                Box(0, np.inf, (self.max_cars, 2), dtype=np.float32),
            't':
                Box(0, np.inf, (1, 1), dtype=np.int32),
            'query_loc':
                Box(0, np.inf, (1, 2), dtype=np.float32),
            'remaining_queries':
                Box(0, np.inf, (1, 1), dtype=np.int32)})
        self.action_space = Discrete(self.engine.n_stations)
        self.reward_range = (-1 * self.engine.max_cars, 3 * self.engine.max_cars)
        return self.state()

    def step(self, action: Action) -> Tuple[State, Reward, bool, int]:
        return self.engine.step(action)

    def render(self, mode='human') -> Optional[np.ndarray]:
        ...

    def close(self) -> None:
        pass

    def seed(self, seed: Union[int, RandomState] = None) -> None:
        self.generator.seed(seed)

    def state(self) -> State:
        return self.engine.state()

    def __str__(self) -> str:
        ...


def load_continuous_simulation(
        config: Config,
        handle_missing: str = 'replace',
        handle_breaking: str = 'full',
        force_reload: bool = False) -> ContinuousSimulation:
    for attr in ['car_speed', 'max_cars', 'sample_amount', 'sample_distance']:
        assert hasattr(config, attr)
    for attr in ['date', 'region']:
        assert hasattr(config, attr)
    start_date: pd.datetime = pd.to_datetime(config.date)

    daterange: pd.DatetimeIndex = pd.date_range(
        start=start_date,
        end=start_date + pd.Timedelta(days=1),
        freq='h')

    changes, charger_timeseries, charger_locations = \
        _load_changes(daterange, config.region,
                      handle_missing, handle_breaking,
                      limiting_chargers=None, force_reload=force_reload,
                      group_stations=False)

    mapping: List[int] = charger_station_mapping(charger_timeseries, charger_locations)
    # len = n_chargers
    arrivals: Arrivals = _get_arrival_events(changes)
    arrivals = arrivals_charger2station(arrivals, mapping)
    departures: List[List[int]] = _get_departure_events(changes)
    departures = departures_charger2station(departures, mapping)
    # both: len = n_tsteps
    initial_state: np.ndarray = occupancy_chargers2station(
        charger_timeseries.values[:, 0], mapping)
    max_occ: np.ndarray = max_occupancy(mapping)
    max_occ = max_occ[:, np.newaxis]
    # np.ndarray[int32] : [n_stations, 1]
    assert len(max_occ) == len(initial_state)
    locations: np.ndarray = charger_locations[['lng', 'lat']].values
    # np.ndarray[float64] : [n_stations, 2] => (x, y)
    station_info: np.ndarray = np.concatenate((locations, max_occ), axis=1)
    station_info: np.ndarray = station_info.astype(np.float32)
    # np.ndarray[float32] : [n_stations, 3] => (x, y, max_occ)

    return ContinuousSimulation(
        arrivals,
        departures,
        initial_state,
        station_info,
        config)


def charger_station_mapping(
        charger_timeseries: pd.DataFrame,
        charger_locations: pd.DataFrame) -> List[int]:
    return [charger_locations.index.get_loc(charger.split(":")[0]) for
            charger in charger_timeseries.index]


def arrivals_charger2station(
        arrivals: Arrivals,
        mapping: List[int]) -> Arrivals:
    return [
        [ArrivalEvent(mapping[event.idx], event.duration) for event in tstep_events]
        for tstep_events in arrivals]


def departures_charger2station(
        departures: List[List[int]],
        mapping: List[int]) -> List[List[int]]:
    return [
        [mapping[idx] for idx in tstep_departures]
        for tstep_departures in departures]


def occupancy_chargers2station(
        initial_occupancy: np.ndarray,
        mapping: List[int]) -> np.ndarray:
    """

    :param initial_occupancy: np.ndarray[float64] : [n_chargers, ]
        3 indicates full, 2 indicates empty, no other values accepted
    :param mapping: List[int]
        the value of the ith index is the index of the station that the
        ith charger belongs to
    :return: np.ndarray[int32] : [n_stations,]
        number of chargers in use at the station
    """
    n_stations: int = max(mapping) + 1
    result: np.ndarray = np.zeros((n_stations,), dtype=np.int32)
    # np.ndarray[int32] : [n_stations,]
    in_use: np.ndarray = (initial_occupancy == 3.)
    # np.ndarray[bool] : [n_chargers, ]
    for i in range(len(in_use)):
        if in_use[i]:
            result[mapping[i]] += 1
    return result


def max_occupancy(mapping: List[int]) -> np.ndarray:
    """

    :param mapping: List[int]
        the value of the ith index is the index of the station that the
        ith charger belongs to
    :return: np.ndarray[int32] : [n_stations,]
        number of chargers in use at the station
    """
    return np.unique(mapping, return_counts=True)[1].astype(np.int32)
