from numbers import Number
from typing import Iterator, List, Callable, Dict

import numpy as np
import wandb

from agent import Agent
from agent.dqn import ReplayBuffer
from env import ContinuousSimulation
from misc.utils import flatmap
from misc.wandb_utils import log, log_histogram


def diagnostic(
        agent: Agent,
        env: ContinuousSimulation,
        wandb: bool = True,
        seeds: Iterator[int] = None) -> None:
    """

    :param agent: Agent
        Agent to evaluate (will not be changed)
    :param env: ContinuousSimulation
        environment to test in
    :param (optional) wandb: bool
        whether or not to use wandb
        defaults to True
    :param seeds:
        seeds to reset environment with, tests will be evaluated on these seeds
        defaults to range(0, 10)
    :return: all results logged to either wandb or console
    """
    if seeds is None:
        seeds = iter(range(0, 10))
    seed = next(seeds, None)

    rewards: List[float] = []
    losses: List[float] = []
    replay_buffer = ReplayBuffer(100000, seed)

    summmaries: List[Dict] = []

    while seed is not None:
        env.seed(seed)
        state = env.reset(logging=True)
        context = env.unwrapped.state()
        episode_reward: float = 0
        done = False



        while not done:
            action = agent.act(state, 0)

            next_state, reward, done, _ = env.step(int(action.cpu()))
            next_context = env.unwrapped.state()
            episode_reward += reward

            replay_buffer.push(state, context, action, reward,
                    next_state, next_context, done)

            state, context = next_state, next_context

        rewards.append(episode_reward)
        if hasattr(agent, 'compute_td_loss'):
            loss = agent.compute_td_loss(*replay_buffer.sample(50))
            losses.append(loss.cpu().data)

        seed = next(seeds, None)

        summmaries.append(env.summary())

    aggregate_summary = {}
    for key in summmaries[0]:
        if isinstance(summmaries[0][key], Number):
            # dtype is a number
            aggregate_summary[key] = list(map(
                lambda s: s[key], summmaries))
        else:
            aggregate_summary[key] = flatmap(map(
                lambda s: s[key], summmaries))

    histograms: List[str] = ['distances travelled', 'timesteps travelled', 'nearest distances']
    log_histogram(wandb, {key: aggregate_summary[key] for key in histograms})
    log(wandb, {key: aggregate_summary[key]  for key in aggregate_summary.keys()
                if (key not in histograms)})
    log(wandb, {'Reward': np.mean(rewards)})
    if hasattr(agent, 'compute_td_loss'):
        log(wandb, {'Loss': np.mean(losses)})


def train(agent: Agent,
          env: ContinuousSimulation,
          log_every: int,
          test_every: int,
          target_network_update_freq: int,
          max_ts: int,
          wandb: bool = True,
          env_seeds: Iterator[int] = None,
          test: Callable[[Agent], None] = None,
          **kwargs) -> Agent:
    """

    :param agent: Agent
        agent to be trained, modified in place
    :param env: ContinuousSimulation
        environment to train in
    :param log_every: int
        logs to either stdout or wandb every 'log_every' timesteps
    :param test_every:
        every 'test_every' timesteps, runs the test function on agent and env
    :param target_network_update_freq: int
        updates the target_network with this frequency (in timesteps)
    :param max_ts: int
        will stop training after max_ts timesteps
    :param (optional) wandb: bool = True
        whether to log to wandb or not. If not, logs to console
    :param env_seeds:
        seeds to seed environment with upon resets, in order
        doesn't set seed if not provided
        assumed that there are enough seeds to run max_ts timesteps
    :param (optional) test: Callable[Agent, None]
        uses this function to test. No testing done if not provided
        should probably be a lambda function of 'test' in this file
    :param kwargs: for compatibility
    :return: the Agent, after training
    """
    episode_reward = 0

    if env_seeds is not None:
        env.seed(next(env_seeds))
    state = env.reset()
    context = env.unwrapped.state()

    n_eps_completed: int = 0
    for ts in range(1, max_ts + 1):
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

        if ts % log_every == 0:
            log(wandb,
                { 'global timestep': ts,
                'num updates': int(ts / target_network_update_freq),
                'num episodes': n_eps_completed})
            agent.log(ts)

        if ts % test_every == 0 and test is not None:
            test(agent)

    return agent