from typing import Union, Tuple, Optional, List

import gym
from gym.spaces import Space, Dict, Discrete, Box
import numpy as np
from numpy.random import RandomState
from wandb.util import PreInitObject as Config

from data_structures import PriorityQueue
from env.continuous_simulation_engine import (
    ContinuousSimulationEngine, Reward, Action, State)
from env.continuous_simulation_generator import ContinuousSimulationGenerator
from env.simulation_events import Arrivals, ArrivalEvent, Queries, QueryEvent
from misc.utils import optional_random, array_shuffle


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

        self.reset()

    def reset(self):
        self.engine = self.generator.generate()
        n_stations: int = self.engine.n_stations
        max_cars: int = self.engine.max_cars
        max_occ: int = np.max(self.engine.station_info[:,2])
        self.observation_space = Dict({
            'station_locations':
                Box(0, np.inf, (n_stations, 2), dtype=np.float32),
            'station_occs':
                Box(0, max_occ, (n_stations, 1), dtype=np.int32),
            'station_maxes':
                Box(0, max_occ, (n_stations, 1), dtype=np.int32),
            'car_locs':
                Box(0, np.inf, (max_cars, 2), dtype=np.float32),
            'car_dest_idx':
                Box(0, n_stations, (max_cars, 1), dtype=np.int32),
            'car_dest_loc':
                Box(0, np.inf, (max_cars, 2), dtype=np.float32),
            't':
                Box(0, np.inf, (1,), dtype=np.int32),
            'query_loc':
                Box(0, np.inf, (1, 2), dtype=np.float32)})
        self.action_space = Discrete(self.engine.n_stations)
        self.reward_range = (-1 * self.engine.max_cars, 3 * self.engine.max_cars)

    def __getattr__(self, name: str):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    def step(self, action: Action) -> Tuple[State, Reward, bool, int]:
        return self.engine.step(action)

    def render(self, mode='human') -> Optional[np.ndarray]:
        ...

    def close(self) -> None:
        pass

    def seed(self, seed: Union[int, RandomState] = None) -> None:
        self.generator.seed(seed)

    def __str__(self) -> str:
        ...
