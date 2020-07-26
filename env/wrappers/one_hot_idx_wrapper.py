from copy import deepcopy

import gym
from gym.spaces import Space, Dict, Box
import numpy as np

from env import ContinuousSimulation, State
from misc.utils import np_onehot


class OneHotIndexWrapper(gym.core.ObservationWrapper, ContinuousSimulation):
    def __init__(self,
            env: ContinuousSimulation):
        super().__init__(env)
        assert isinstance(self.env.observation_space, Dict)
        self.observation_space: Dict = deepcopy(self.env.observation_space)
        self.n_stations = self.observation_space.spaces['station_locations'].shape[0]
        self.max_cars = self.observation_space.spaces['car_locs'].shape[0]
        self.observation_space.spaces['car_dest_idx'] = Box(
            0, 1, (self.max_cars, self.n_stations), dtype=np.int32)
        self.observation_space.spaces['station_idx'] = Box(
            0, 1, (self.n_stations, self.n_stations), dtype=np.int32)

    def observation(self,
            observation: State) -> State:
        car_indices: np.ndarray = np_onehot(observation.car_dest_idx, max=self.n_stations-1)
        # np.ndarray[int32] : [cur_n_cars, 1, self.n_stations]
        car_indices= car_indices[:, 0, :]
        # [cur_n_cars, self.n_stations]
        station_indices: np.ndarray = np_onehot(observation.station_idx, max=self.n_stations-1)
        station_indices = station_indices[:, 0, :]
        # [n_stations, n_stations] should be eye(n_stations)
        return State(
            station_idx=station_indices,
            station_locations=observation.station_locations,
            station_maxes=observation.station_maxes,
            station_occs=observation.station_occs,
            car_dest_loc=observation.car_dest_loc,
            car_dest_idx=car_indices,
            car_locs=observation.car_locs,
            query_loc=observation.query_loc,
            t=observation.t,
            remaining_queries=observation.remaining_queries)
