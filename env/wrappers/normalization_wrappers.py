from copy import copy

import gym
from gym.spaces import Space, Dict, Box
import numpy as np

from env import BoundsMapper, ContinuousSimulation, State

class NormalizedPositionWrapper(gym.core.ObservationWrapper, ContinuousSimulation):
    def __init__(self,
            env: ContinuousSimulation,
            normalize_offset: bool = True,
            normalize_scale: bool = True,
            scale: float = 1):
        super().__init__(env)
        self._normalize_offset: bool = normalize_offset
        if normalize_scale:
            assert normalize_offset
        self._normalize_scale: bool = normalize_scale
        self.scale: float = scale

        region: str = self.env.config.region
        mapper = BoundsMapper(region, coordsys='gjc02')
        self._map_offset: np.ndarray = np.asarray(
            (mapper.bounds['left'], mapper.bounds['bottom'])).astype(np.float32)
        self._map_scale: np.ndarray = np.asarray(
            (mapper.bounds['right'], mapper.bounds['top'])).astype(np.float32)
        self._map_scale = self._map_scale - self._map_offset
        # both np.ndarray[float32] : (x, y)

    def _normalize_loc_(self, loc: np.ndarray) -> np.ndarray:
        # (x, y) => (x, y)
        assert loc.shape[1] == 2
        if np.all(loc == 0):
            return loc # don't normalize 0 vector
        if self._normalize_offset:
            loc -= self._map_offset
            if self._normalize_scale:
                loc /= self._map_scale
        loc *= self.scale
        return loc

    def observation(self, observation: State) -> State:
        return State(
            station_locations=self._normalize_loc_(observation.station_locations),
            station_occs=observation.station_occs,
            station_maxes=observation.station_maxes,
            car_locs=self._normalize_loc_(observation.car_locs),
            car_dest_idx=observation.car_dest_idx,
            car_dest_loc=self._normalize_loc_(observation.car_dest_loc),
            t=observation.t,
            query_loc=self._normalize_loc_(observation.query_loc),
            remaining_queries=observation.remaining_queries)


class NormalizedTimeWrapper(gym.core.ObservationWrapper, ContinuousSimulation):
    def __init__(self,
            env: ContinuousSimulation):
        super().__init__(env)
        self._max_t:int  = self.env.max_t

        og: Dict = self.env.observation_space
        spaces = { }
        for name in og.spaces.keys():
            space: Box = copy(og[name])
            if 't' == name:
                space.high = 1.
                space.dtype = np.float32
            spaces[name] = space
        self.observation_space: Dict = Dict(spaces)

    def observation(self, observation: State) -> State:
        new_t: float = (observation.t / self._max_t).astype(np.float32)
        return State(
            station_locations=observation.station_locations,
            station_occs=observation.station_occs,
            station_maxes=observation.station_maxes,
            car_locs=observation.car_locs,
            car_dest_idx=observation.car_dest_idx,
            car_dest_loc=observation.car_dest_loc,
            t=new_t,
            query_loc=observation.query_loc,
            remaining_queries=observation.remaining_queries)