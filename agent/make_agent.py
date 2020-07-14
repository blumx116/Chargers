from copy import deepcopy

from agent import Agent
from agent.dqn import make_model, DQNAgent
from agent.simple import NearestAgent
from misc.utils import kwargify

def make_agent(
        algorithm: str,
        **kwargs) -> Agent:
    """
    :param algorithm: str
        one of [nearest, dqn]
    :param kwargs: seed NearestAgent, MakeModel and DQNAgent
    :return: the agent needed
    """
    kwargs = kwargify(locals())
    algorithm: str = algorithm.lower()
    if algorithm == 'nearest':
        return NearestAgent(**kwargs)
    elif algorithm == 'dqn':
        model = make_model(**kwargs)
        target = deepcopy(model)
        return DQNAgent(model, target, **kwargs)