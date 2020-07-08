from wandb.util import PreInitObject as Config

from env import ContinuousSimulation, load_continuous_simulation
from env.wrappers import (
    PositionEncodingWrapper,
    NormalizedPositionWrapper,
    StaticFlatWrapper,
    SummedRewardWrapper,
    TimeEncodingWrapper,
    AttentionModelWrapper,
    OneHotIndexWrapper)


def make_and_wrap_env(config: Config) -> ContinuousSimulation:
    sim = load_continuous_simulation(config)
    model: str = config.model.lower()
    if model == 'feedforward':
        sim = NormalizedPositionWrapper(sim)
        sim = StaticFlatWrapper(sim)
        sim = SummedRewardWrapper(sim)
        return sim
    if model == 'transformer':
        sim = NormalizedPositionWrapper(sim)
        sim = PositionEncodingWrapper(sim, dimension=10)
        sim = TimeEncodingWrapper(sim, dimension=10)
        sim = OneHotIndexWrapper(sim)
        sim = AttentionModelWrapper(sim)
        return sim

