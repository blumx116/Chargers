from tensorflow.keras.models import clone_model

from agent import Agent
from agent.dqn import make_model, DQNAgent
from agent.simple import NearestAgent, MostOpenAgent
from misc.utils import kwargify

def make_agent(
        algorithm: str,
        **kwargs) -> Agent:
    """
    :param algorithm: str
        one of [nearest, dqn, open]
    :param kwargs: seed NearestAgent, MakeModel and DQNAgent
    :return: the agent needed
    """
    kwargs = kwargify(locals())
    algorithm: str = algorithm.lower()
    if algorithm == 'nearest':
        return NearestAgent(**kwargs)
    elif algorithm == 'open':
        return MostOpenAgent(**kwargs)
    elif algorithm == 'dqn':
        model = make_model(**kwargs)
        target = make_model(**kwargs)
        # target.set_weights(model.get_weights())
        # can't copy weights because models haven't been built yet
        return DQNAgent(model, target, **kwargs)