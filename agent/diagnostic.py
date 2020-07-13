from typing import Iterator, List, Callable

import numpy as np
import wandb

from agent import Agent
from agent.dqn import ReplayBuffer
from env import ContinuousSimulation
from misc.config import Config, log

def diagnostic(
        agent: Agent,
        env: ContinuousSimulation,
        config: Config,
        seeds: Iterator[int] = None) -> None:
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

    log({'Reward': np.mean(rewards)})
    if hasattr(agent, 'compute_td_loss'):
        log({'Loss': np.mean(losses)})

def train(agent: Agent,
          env: ContinuousSimulation,
          config: Config,
          env_seeds: Iterator[int] = None,
          test: Callable[[Agent], None] = None) -> None:
    episode_reward = 0

    if env_seeds is not None:
        env.seed(next(env_seeds))
    state = env.reset()
    context = env.unwrapped.state()

    n_eps_completed: int = 0
    for ts in range(1, config.max_ts + 1):
        action = agent.act(state, context, mode='train', network='q')

        next_state, reward, done, _ = env.step(int(action.cpu()))
        next_context = env.unwrapped.state()

        agent.remember(state, context, action, reward,
                       next_state, next_context, done)

        if done:
            n_eps_completed += 1
            state = env.reset()
            if env_seeds is not None:
                env.seed(next(env_seeds))

        agent.optimize()
        agent.step(ts)

        if ts % config.log_every == 0:
            log(config,
                { 'global timestep': ts,
                'num updates': int(ts / config.target_network_update_freq),
                'num episodes': n_eps_completed})
            agent.log(ts)

        if ts % config.test_every == 0 and test is not None:
            test(agent)
