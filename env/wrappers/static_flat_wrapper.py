from typing import Tuple, Dict, List

import gym
from gym.spaces import Space, Box
from gym.spaces import Dict as DictSpace
import numpy as np

from env import ContinuousSimulation, State


class StaticFlatWrapper(gym.core.ObservationWrapper):
    def __init__(self,
            env: ContinuousSimulation):
        super().__init__(env)

        high: float = -np.inf
        low: float = np.inf
        og: DictSpace = self.env.observation_space
        shape: int = 0
        for key in og.spaces.keys():
            space: Box = og[key]
            high = max(high, np.max(space.high))
            low = min(low, np.min(space.low))
            shape += np.prod(space.shape)
        self.observation_space = Box(low, high, (shape, ), dtype=np.float32)

    @staticmethod
    def _pad_zeros_(
            arr: np.ndarray,
            shape: Tuple[int]) -> np.ndarray:
        assert len(shape) == len(arr.shape)
        for orig_shape, new_shape in zip(arr.shape, shape):
            assert new_shape >= orig_shape
        result: np.ndarray = np.zeros(shape, dtype=arr.dtype)
        slices: Tuple[slice] = tuple(map(lambda dim: slice(None, dim), arr.shape))
        result[slices] = arr
        return result

    def observation(self, observation: State) -> np.ndarray:
        """

        :param observation: as defined in ContinuousSimulation
        :return: np.ndarray with info in this order
            station_locations, station_occs, station_maxes, car_locs
            car_dest_idx, car_dest_loc, t, query_loc
            dtype: float32
            size: (n_stations * 4) + (max_cars * 5) + 3
        """
        max_cars: int = self.env.max_cars
        observation: Dict = observation._asdict()
        arrs: List[np.ndarray] = []
        for key in observation.keys():
            correct_shape: Tuple[int] = self.env.observation_space[key].shape
            # the shape specified in observation_space isn't accurate, because
            # the dimensions are variable. Here, we're fixing it to max size
            padded: np.ndarray = self._pad_zeros_(observation[key], correct_shape)
            # np.ndarray[Any] : correct_shape
            flattened: np.ndarray = padded.flatten().astype(np.float32)
            arrs.append(flattened)
        return np.concatenate(arrs, axis=0)
