from copy import copy

import gym
from gym.spaces import Dict, Box, Space
import numpy as np

from env import ContinuousSimulation, State


class PositionalEncoder:
    def __init__(self, dim: int):
        assert dim % 2 == 0

        self.dim: int = dim
        frequencies: np.ndarray = np.arange(0, self.dim / 2)
        frequencies = 2 * frequencies / dim
        frequencies = np.power(50, frequencies)
        self.frequencies: np.ndarray = 1 / frequencies

    def encode(self, val: float) -> np.ndarray:
        """
        see: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
        :param val: the value (preferably between 0 and 50) to encode
        :return: np.ndarray[float32] : [1, dim/ 2]
        """
        sines: np.ndarray = np.sin(val * self.frequencies)[np.newaxis, :]
        coses: np.ndarray = np.cos(val * self.frequencies)[np.newaxis, :]
        # both np.ndarray[float32] : [1, dim / 2]
        return np.concatenate((sines, coses), axis=1).astype(np.float32)


class TimeEncodingWrapper(gym.core.ObservationWrapper, ContinuousSimulation):
    def __init__(self,
            env: ContinuousSimulation,
            dimension: int):
        super().__init__(env)
        self.encoder: PositionalEncoder = PositionalEncoder(dimension)

        og: Dict = self.env.observation_space
        spaces = {}
        for name in og.spaces.keys():
            space: Box = copy(og[name])
            if 't' == name:
                space.shape = (space.shape[0], space.shape[1] * dimension)
            spaces[name] = space
        self.observation_space: Dict = Dict(spaces)

    def observation(self, observation: State):
        state = State(
            observation.station_locations,
            observation.station_occs,
            observation.station_maxes,
            observation.car_locs,
            observation.car_dest_idx,
            observation.car_dest_loc,
            self.encoder.encode(observation.t),
            observation.query_loc)


class PositionEncodingWrapper(gym.core.ObservationWrapper, ContinuousSimulation):
    def __init__(self,
            env: ContinuousSimulation,
            dimension: int):
        super().__init__(env)
        self.encoder: PositionalEncoder = PositionalEncoder(dimension)
        og: Dict = self.env.observation_space
        spaces = { }
        for name in og.spaces.keys():
            space: Box = copy(og[name])
            if 'loc' in name:
                space.shape = (space.shape[0], space.shape[1] * dimension)
            spaces[name] = space
        self.observation_space: Dict = Dict(spaces)

    def _encode_loc(self, loc: np.ndarray) -> np.ndarray:
        """
        :param loc: np.ndarray[float32] : [1, 2]
        :return: np.ndarray[float32] : [1, self.dimension * 2]
        """
        return np.concatenate(
            (self.encoder.encode(loc[0]), self.encoder.encode(loc[1])),
            axis=1)

    def observation(self, observation: State):
        return State(
            self._encode_loc(observation.station_locations),
            observation.station_occs,
            observation.station_maxes,
            self._encode_loc(observation.car_locs),
            observation.car_dest_idx,
            self._encode_loc(observation.car_dest_loc),
            observation.t,
            self._encode_loc(observation.query_loc))
