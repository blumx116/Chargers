import numpy.random as rand
import numpy as np

from misc.utils import listify
from env import load_continuous_simulation
from env.wrappers import SummedRewardWrapper, StaticFlatWrapper, NormalizedPositionWrapper
import wandb

wandb.login()
wandb.init(project='debug')

wandb.config.update({
    'max_cars' : 1000,
    'car_speed': 0.15,
    'sample_distance': 0.3,
    'sample_amount': 1,
    'date': '06-27-2019',
    'region': 'haidian'
})

sim = load_continuous_simulation(wandb.config)
sim = NormalizedPositionWrapper(sim)
sim = StaticFlatWrapper(sim)

sim = SummedRewardWrapper(sim)
sim.seed(0)
sim.reset()

i = 0
done = False
while not done:
    state, reward, done, _ = sim.step(rand.randint(0, 5))
    print(i, np.sum(reward))
    i += 1

while True:
    pass

print()