from agent import Agent

from gym.spaces import Discrete
import numpy as np
import tensorflow as tf

from env import State
from misc.utils import optional_device

class MostOpenAgent(Agent):
    def __init__(self,
            device: tf.device = None,
            use_percent: bool = True,
            **kwargs):
        """
        :param device: tf.device
            uses gpu if not provided
        :param kwargs: for compatibility
        """
        self.device: tf.device = optional_device(device)
        self.use_percent: bool = use_percent


    def score(self,
            observation: State,
            context: State,
            network: str = 'q') -> tf.Tensor:
        """
        gives closer stations higher scores
        :param observation: raw from unwrapped environment
        :param context: same as observations
        :param network: ignored, present for compatibility
        :return: torch[f32, device] : [1, n_stations]
        """
        maxes: np.ndarray = observation.station_maxes.astype(np.float32)
        currents: np.ndarray = observation.station_occs.astype(np.float32)
        # both np.ndarray[f32] : [n_stations, 1]
        if self.use_percent:
            scores: np.ndarray = 1 - (currents / maxes)
        else:
            scores = maxes - currents
        # np.ndarray[f32] : [n_stations, 1]
        scores: tf.Tensor = torch.from_numpy(scores).type(torch.float32)
        # tf.Tensor[f32, cpu]: [n_stations, 1]
        return scores.to(self.device).squeeze(1).unsqueeze(0)

    def act(self,
            observation: State,
            context: State,
            mode='test',
            network='q') -> tf.Tensor:
        """
        gives closer stations higher scores
        :param observation: raw from unwrapped environment
        :param context: same as observations
        :param mode: ignored, present for compatibility
        :param network: ignored, present for compatibility
        :return: torch[int64, device] : max_index
        """
        scores: tf.Tensor = self.score(observation, observation)
        # Tensor[f32, dev] : [1, n_stations]
        return scores.max(1)[1].data[0]

    def optimize(self) -> None:
        pass

    def remember(self,
            state,
            context,
            action,
            reward,
            next_state,
            next_context,
            done: bool) -> None:
        pass

    def step(self, global_timestep: int) -> None:
        pass

    def log(self, global_timestep: int) -> None:
        pass

