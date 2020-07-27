from numbers import Number
from typing import Iterator, List, Callable, Dict

import numpy as np
import tensorflow as tf

from agent import Agent
from agent.dqn import ReplayBuffer
from env import ContinuousSimulation
from misc.utils import flatmap, optional_writer

def initial_log(
        agent: Agent,
        env: ContinuousSimulation,
        writer: tf.summary.SummaryWriter = None,
        **kwargs) -> None:
    writer = optional_writer(writer)

    state = env.reset()
    context = env.unwrapped.state()

    if hasattr(agent, 'score'):
        # RUN 1-Time code to log graph
        pass




def diagnostic(
        agent: Agent,
        env: ContinuousSimulation,
        seeds: Iterator[int] = None,
        writer: tf.summary.SummaryWriter = None) -> None:
    """

    :param agent: Agent
        Agent to evaluate (will not be changed)
    :param env: ContinuousSimulation
        environment to test in
    :param seeds:
        seeds to reset environment with, tests will be evaluated on these seeds
        defaults to range(0, 10)
    :param writer: tf.summary.SummaryWriter
        writer to use as context. Writes no-ops if None
    :return: all results logged to either wandb or console
    """
    if seeds is None:
        seeds = iter(range(0, 10))
    writer = optional_writer(writer)
    seed = next(seeds, None)

    rewards: List[float] = []
    losses: List[float] = []
    replay_buffer = ReplayBuffer(100000, seed)

    summaries: List[Dict] = []

    while seed is not None:
        env.seed(seed)
        state = env.reset(logging=True)
        context = env.unwrapped.state()
        episode_reward: float = 0
        done = False

        while not done:
            action = agent.act(state, 0)

            next_state, reward, done, _ = env.step(int(action))
            next_context = env.unwrapped.state()
            episode_reward += reward

            replay_buffer.push(state, context, action, reward,
                    next_state, next_context, done)

            state, context = next_state, next_context

        rewards.append(episode_reward)
        if hasattr(agent, 'compute_td_loss'):
            loss = agent.compute_td_loss(*replay_buffer.sample(50))
            losses.append(loss.numpy())

        seed = next(seeds, None)

        summaries.append(env.summary())

    agg_summary = {}
    for key in summaries[0]:
        if isinstance(summaries[0][key], Number):
            # dtype is a number
            agg_summary[key] = list(map(
                lambda s: s[key], summaries))
        else:
            agg_summary[key] = flatmap(map(
                lambda s: s[key], summaries))
    max_ts: int = max(map(lambda s: len(s['actual queries']), summaries))
    for key in ['original queries', 'actual queries']:
        agg_summary[key] = np.zeros(max_ts)
        agg_summary['min ' + key] = np.full(max_ts, np.inf)
        agg_summary['max ' + key] = np.full(max_ts, -np.inf)
        for s in summaries:
            for t in range(max_ts):
                agg_summary[key][t] += s[key][t]
                agg_summary['max ' + key][t] = max(agg_summary['max ' + key][t], s[key][t])
                agg_summary['min ' + key][t] = min(agg_summary['min ' + key][t], s[key][t])

    # TODO: this assumes that all stations are ordered in the same order, which may not be correct
    max_stations: int = max(map(lambda s: s['n_stations'], summaries))
    agg_summary['recommendation freq'] = np.zeros(max_stations)
    for summ in summaries:
        agg_summary['recommendation freq'][0:summ['n_stations']] += summ['recommendation freq']

    # these values count the number of queries per timestep summed over all episodes
    histograms: List[str] = ['distances travelled', 'timesteps travelled', 'nearest distances']
    dist_histogram = np.histogram(agg_summary['distances travelled'], bins=20)
    near_histogram = np.histogram(agg_summary['nearest distances'], bins=20)

    exp_bins: np.ndarray = 2 ** np.arange(0, 10)
    exp_bins = np.insert(exp_bins, 0, 0, axis=0)
    # [0, 1, 2, 4, 8, 16, 32, 64, 128, 512]
    failed_histogram = np.histogram(agg_summary['failed dispatches'], bins=exp_bins)
    organic_histogram = np.histogram(agg_summary['organic fails'], bins=exp_bins)
    time_histogram = np.histogram(agg_summary['timesteps travelled'], bins=np.arange(max_ts+1))

    with writer.as_default():
        tf.summary.histogram('distances travelled', agg_summary['distances travelled'])
        tf.summary.histogram('nearest distances', agg_summary['nearest distances'])
        tf.summary.histogram('timesteps travelled', agg_summary['timesteps travelled'])
        tf.summary.histogram('failed dispatches', agg_summary['failed dispatches'])
        tf.summary.histogram('organic fails', agg_summary['organic fails'])
        tf.summary.scalar('Reward', np.mean(rewards))
        if hasattr(agent, 'compute_td_loss'):
            tf.summary.scalar('loss', np.mean(losses))


def train(agent: Agent,
          env: ContinuousSimulation,
          log_every: int,
          test_every: int,
          target_network_update_freq: int,
          max_ts: int,
          writer: tf.summary.SummaryWriter = None,
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
    :param writer: tf.SummaryWriter
        writer to write logs with
    :param max_ts: int
        will stop training after max_ts timesteps
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
    writer = optional_writer(writer)
    state = env.reset()
    context = env.unwrapped.state()

    n_eps_completed: int = 0
    for ts in range(1, max_ts + 1):
        action = agent.act(state, context, mode='train', network='q')

        next_state, reward, done, _ = env.step(int(action))
        next_context = env.unwrapped.state()

        agent.remember(state, context, action, reward,
                       next_state, next_context, done)

        state, context = next_state, next_context

        if done:
            n_eps_completed += 1
            state = env.reset()
            context = env.unwrapped.state()
            if env_seeds is not None:
                env.seed(next(env_seeds))

        if ts > 100:
            agent.optimize()
        agent.step(ts)

        with writer.as_default():
            if ts % log_every == 0:
                tf.summary.experimental.set_step(ts)
                tf.summary.scalar('global timestep', ts)
                tf.summary.scalar('num updates', ts // target_network_update_freq)
                tf.summary.scalar('num episodes', n_eps_completed)
                agent.log(ts, writer)

        if ts % test_every == 0 and test is not None:
            test(agent)

    return agent