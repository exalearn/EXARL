from enum import Enum

# TODO: An Interface class might be better, but quick hack to test
class MEMORY_TYPE(str, Enum):
    UNIFORM_REPLAY = 'uniform'
    PRIORITY_REPLAY = 'priority'
    HINDSIGHT_REPLAY = 'hindsight'
