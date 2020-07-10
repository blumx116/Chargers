from agent import Agent

class AgentWrapper(Agent):
    def __init__(self, agent: Agent):
        self.agent = agent

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    def score(self, observation, context, network='q'):
        return self.agent.score(observation, context, network)

    def act(self, observation, context, mode='test', network='q'):
        return self.agent.act(observation, context, mode, network)

    def optimize(self):
        return self.agent.optimize()

    def remember(self,
            state,
            context,
            action,
            reward,
            next_state,
            next_action,
            done: bool):
        return self.agent.remember(state, context, action,
            reward, next_state, next_action, done)