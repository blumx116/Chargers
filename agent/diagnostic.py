from typing import Iterable, List

import numpy as np
import wandb
from wandb.util import PreInitObject as Config

from agent import Agent
from agent.dqn import ReplayBuffer
from env import ContinuousSimulation


def diagnostic(
        agent: Agent,
        env: ContinuousSimulation,
        config: Config,
        seeds: Iterable[int] = None) -> None:
    if seeds is None:
        seeds = range(0, 10)
    seed = next(seeds, None)

    rewards: List[float] = []
    losses: List[float] = []


    while seed is not None:
        env.seed(0)
        state = env.reset()
        episode_reward: float = 0
        done = False

        replay_buffer = ReplayBuffer(100000, seed)

        while not done:
            action = agent.act(state, 0)

            next_state, reward, done, _ = env.step(int(action.cpu()))
            episode_reward += reward

            replay_buffer.push(state, action,reward, next_state, done)

        rewards.append(episode_reward)
        if hasattr(agent, 'compute_td_loss'):
            loss = agent.compute_td_loss(50, replay_buffer, None, config.gamma)
            losses.append(loss.data)

    wandb.log({'Reward': np.mean(rewards)})
    if hasattr(agent, 'compute_td_loss'):
        wandb.log({'Loss': np.mean(losses)})
