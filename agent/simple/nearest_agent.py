from typing import Union

from gym.spaces import Discrete
import numpy as np
from numpy.random import RandomState

from agent import Agent
from env import State
from env.internals.continuous_simulation_engine import get_distances
from misc.utils import optional_device


class NearestAgent(Agent):
    def __init__(self,
            device: torch.device = None,
            **kwargs):
        """
        :param (optional) device: torch.device
            device to put returned values on
        :param kwargs: for compatibility
        """
        self.device: torch.device = optional_device(device)

    def score(self,
            observation: State,
            context: State,
            network='q') -> torch.Tensor:
        """
        gives closer stations higher scores
        :param observation: raw from unwrapped environment
        :param context: same as observations
        :param network: ignored, present for compatibility
        :return: torch[f32, device] : [1, n_stations]
        """
        scores: np.ndarray = -get_distances(observation.query_loc,
                stations=observation.station_locations)
        scores: torch.Tensor = torch.from_numpy(scores)
        # [n_stations, ]
        return scores.type(torch.float32).to(self.device).unsqueeze((0))

    def remember(self,
            state,
            context,
            action,
            reward,
            next_state,
            next_context,
            done: bool) -> None:
        pass

    def act(self,
            observation: State,
            context: State,
            mode='test',
            network='q') -> torch.Tensor:
        """
        gives closer stations higher scores
        :param observation: raw from unwrapped environment
        :param context: same as observations
        :param mode: ignored, present for compatibility
        :param network: ignored, present for compatibility
        :return: torch[int64, device] : max_index
        """
        scores: torch.Tensor = self.score(observation, observation)
        # Tensor[f32, dev] : [1, n_stations]
        return scores.max(1)[1].data[0]

    def optimize(self) -> None:
        pass

    def step(self, global_timestep: int) -> None:
        pass

    def log(self, global_timestep: int) -> None:
        pass
