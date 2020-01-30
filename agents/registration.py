import importlib
from gym import error, logger 

def load(name):
        mod_name, attr_name = name.split(":")
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, attr_name)
        return fn

class AgentSpec(object):
        def __init__(self, id, entry_point=None, kwargs=None):
                self.id = id
                self.entry_point = entry_point
                self._kwargs = {} if kwargs is None else kwargs

        def make(self, **kwargs):
                """Instantiates an instance of the environment with appropriate kwargs"""
                if self.entry_point is None:
                        raise error.Error('Attempting to make deprecated agent {}. (HINT: is there a newer registered version of this agent?)'.format(self.id))
                _kwargs = self._kwargs.copy()
                _kwargs.update(kwargs)
                if callable(self.entry_point):
                        agent = self.entry_point(**_kwargs)
                else:
                        cls = load(self.entry_point)
                        agent = cls(**_kwargs)

                return agent

class AgentRegistry(object):
        def __init__(self):
                self.agent_specs = {}

        def make(self, path, **kwargs):
                if len(kwargs) > 0:
                        logger.info('Making new env: %s (%s)', path, kwargs)
                else:
                        logger.info('Making new env: %s', path)
                spec = self.spec(path)
                agent = spec.make(**kwargs)

                return agent

        def all(self):
                return self.agent_specs.values()

        def spec(self, path):
                if ':' in path:
                        mod_name, _sep, id = path.partition(':')
                        try:
                                importlib.import_module(mod_name)
                        except ImportError:
                                raise error.Error('A module ({}) was specified for the agent but was not found, make sure the package is installed with `pip install` before calling `exa_gym_agent.make()`'.format(mod_name))

                else:
                        id = path

                try:
                        return self.agent_specs[id]
                except KeyError:
                        raise error.Error('No registered agent with id: {}'.format(id))

        def register(self, id, **kwargs):
                if id in self.agent_specs:
                        raise error.Error('Cannot re-register id: {}'.format(id))
                self.agent_specs[id] = AgentSpec(id, **kwargs)


# Global agent registry
registry = AgentRegistry()

def register(id, **kwargs):
        return registry.register(id, **kwargs)

def make(id, **kwargs):
        return registry.make(id, **kwargs)

def spec(id):
	return registry.spec(id)
