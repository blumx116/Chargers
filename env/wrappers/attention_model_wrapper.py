from typing import NamedTuple, Tuple

import gym
from gym.spaces import Space, Dict, Box
import numpy as np

from env import ContinuousSimulation, State

AttentionState = NamedTuple("AttentionState",
                    [('stations', np.ndarray), ('cars', np.ndarray)])


class AttentionModelWrapper(gym.core.ObservationWrapper, ContinuousSimulation):
    def __init__(self,
            env: ContinuousSimulation,
            pad_zeros: bool = True):
        super().__init__(env)
        self.pad_zeros = pad_zeros
        station_dims: int = 0
        car_dims: int = 0
        og_space: Dict = self.env.observation_space
        def n_features(key):
            return og_space[key].shape[1]
        for dim in og_space.spaces.keys():
            if 'station' in dim:
                station_dims += n_features(dim)
            elif 'car' in dim:
                car_dims += n_features(dim)
        station_dims += n_features('query_loc')
        station_dims += n_features('t')
        car_dims += n_features('t')
        self.n_stations: int = og_space['station_locations'].shape[0]
        self.max_cars: int = og_space['car_locs'].shape[0]

        self.observation_space: Space = Dict({
            'stations': Box(0, np.inf, (self.n_stations, station_dims)),
            'cars' : Box(0, np.inf, (self.max_cars, car_dims))
        })
        # NOTE: car size is variable, but will never be larger than max_cars

    def observation(self, observation: State) -> AttentionState:
        # NOTE: all calculations assume that State is as it was in SimulationState
        # at least for comments. Code is fully compatibl
        stations: np.ndarray = np.concatenate(
            (observation.station_locations,
             observation.station_occs,
             observation.station_maxes), axis=1).astype(np.float32)
        # np.ndarray[float32] : [n_stations, 4] => (x, y, occ, max)
        cars: np.ndarray = np.concatenate(
            (observation.car_locs,
             observation.car_dest_idx,
             observation.car_dest_loc), axis=1).astype(np.float32)
        # np.ndarray[float32] : [cur_n_cars, 5]
        n_stations: int = stations.shape[0]
        stations = np.concatenate(
            (stations,
             np.repeat(observation.t, n_stations, axis=0),
             np.repeat(observation.query_loc, n_stations, axis=0),
             np.repeat(observation.remaining_queries, n_stations, axis=0)),
            axis=1)
        # append time to each stations' input
        # np.ndarray[float32] : [n_stations, 7]
        cur_n_cars: int = cars.shape[0]
        cars: np.ndarray = np.concatenate(
            (cars,
             np.repeat(observation.t, cur_n_cars, axis=0)),
            axis=1)
        if self.pad_zeros:
            cars = self._pad_zeros_(cars, (self.max_cars, cars.shape[1]))
        return AttentionState(stations, cars)


    @staticmethod
    def _pad_zeros_(
            arr: np.ndarray,
            shape: Tuple[int]) -> np.ndarray:
        # TODO: this should be the same as the other one in
        # static_flat_wrapper, make sure to refactor that
        assert len(shape) == len(arr.shape)
        for orig_shape, new_shape in zip(arr.shape, shape):
            assert new_shape >= orig_shape
        result: np.ndarray = np.zeros(shape, dtype=arr.dtype)
        slices: Tuple[slice] = tuple(map(lambda dim: slice(None, dim), arr.shape))
        result[slices] = arr
        return result