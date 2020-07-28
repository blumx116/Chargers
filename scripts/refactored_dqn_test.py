from copy import deepcopy
from datetime import datetime
import os
from time import time
import warnings

import numpy.random as rand
import numpy as np

from agent.diagnostic import train, diagnostic
from agent import make_agent
from env import make_and_wrap_env
from misc.utils import root_dir
import tensorflow as tf


warnings.simplefilter('once')

LOG_TENSORBOARD = True

settings = {
    'algorithm': 'dqn',
    'model': 'feedforward',
    'n_layers': 2,
    'n_heads': 4,
    'n_nodes' : 160,
    'normalize': True,
    'max_cars': 200,
    'car_speed': 0.1,
    'sample_distance': 0.3,
    'sample_amount': 5,
    'date': '06-27-2019',
    'region': 'haidian',
    'learning_rate': 1e-5,
    'replay_size': 100000,
    'target_network_update_freq': 10000,
    'log_every': 10,
    'test_every': 101,
    'gamma': 0.99,
    'start_train_ts': 1000,
    'batch_size': 32,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 300000,
    'max_ts': 140000,
    'seed': 0}

np.random.seed(settings['seed'])

sim = make_and_wrap_env(**settings)

settings.update({
    'observation_space': sim.observation_space,
    'action_space': sim.action_space,
    'device': tf.device("/physical_device:GPU:0"),
})

if LOG_TENSORBOARD:
    cur_time: str = datetime.fromtimestamp(time()).strftime("%d-%m-%Y-%H-%M")
    run_name = f"{settings['algorithm']} ({settings['model']})=={settings['date']}=={cur_time}"
    writer = tf.summary.create_file_writer(os.path.join(root_dir, "logs", run_name))
else:
    writer = tf.summary.create_noop_writer()

agent = make_agent(**settings)
test = lambda agent: diagnostic(
    agent=agent, env=sim,
    seeds=iter(range(1, 10)),
    writer=writer)

with writer.as_default():
    train(agent, sim, test=test, writer=writer, **settings)
