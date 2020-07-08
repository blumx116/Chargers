import gym
import numpy as np

from env import ContinuousSimulation
from env import Reward as OldReward

Reward = float


class SummedRewardWrapper(gym.core.RewardWrapper, ContinuousSimulation):
    def __init__(self,
            env: ContinuousSimulation):
        super().__init__(env)

    def reward(self, reward: OldReward) -> Reward:
        return float(np.sum(reward))