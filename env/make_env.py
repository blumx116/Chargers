from env import ContinuousSimulation, load_continuous_simulation
from env.wrappers import (
    PositionEncodingWrapper,
    NormalizedPositionWrapper,
    StaticFlatWrapper,
    SummedRewardWrapper,
    TimeEncodingWrapper,
    AttentionModelWrapper,
    OneHotIndexWrapper)


def make_and_wrap_env(
        algorithm: str,
        model: str,
        **kwargs) -> ContinuousSimulation:
    """
    :param algorithm: str
        one of [nearest, dqn]
    :param model: str
        one of [feedforward, transformer]
        ignored if algorithm is nearest
    :param kwargs: passed to load_continuous_simulation
    :return: ContinuousSimulation, wrapped as necessary
    """
    sim = load_continuous_simulation(**kwargs)
    algorithm = algorithm.lower()
    if algorithm in ['nearest', 'open']:
        return sim
    assert algorithm == 'dqn', f"Expected algorithm to be one of [nearest, open, dqn], got {algorithm}"
    model = model.lower()
    if model == 'feedforward':
        sim = NormalizedPositionWrapper(sim, kwargs['region'])
        sim = StaticFlatWrapper(sim)
        sim = SummedRewardWrapper(sim)
        return sim
    if model in ['transformer', 'trxl', 'trxli']:
        sim = NormalizedPositionWrapper(sim, kwargs['region'])
        sim = PositionEncodingWrapper(sim, dimension=10)
        sim = TimeEncodingWrapper(sim, dimension=10)
        sim = OneHotIndexWrapper(sim)
        sim = AttentionModelWrapper(sim)
        sim = SummedRewardWrapper(sim)
        return sim
    raise Exception(f"Expected model to be one of [transformer, trxl, trxli], got {model}")

