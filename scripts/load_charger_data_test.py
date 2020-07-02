import pandas as pd

import numpy.random as rand
import numpy as np

from env.load_charger_data import load_charger_data
from env.simulation_helper import load_continuous_simulation
from env import load_day_simulation
import wandb

"""
data = load_charger_data(
    pd.date_range('06-27-2019', '06-28-2019', freq='h'),
    'haidian', group_stations=True, handle_missing='replace',
    force_reload=True)
"""
wandb.login()
wandb.init('debug')

wandb.config.update({
    'max_cars' : 1000,
    'car_speed': 1,
    'sample_distance': 1,
    'sample_amount': 1,
    'date': '06-27-2019',
    'region': 'haidian'
})


sim = load_continuous_simulation(wandb.config)
sim.seed(0)
sim.reset()
i = 0
done = False
while not done:
    _, reward, done, _ = sim.step(rand.randint(0, 5))
    print(i, np.sum(reward))
    i += 1

print()