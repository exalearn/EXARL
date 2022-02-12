from abc import ABC, abstractmethod

class Replay(ABC):

    def __init__(self, max_size):
        super(Replay, self).__init__()
        self._memory_size = max_size
        self._mem_length = 0

    @property
    def is_full(self) -> bool:
        return self._mem_length == self._memory_size

    @property
    def buffer_size(self) -> int:
        return self._memory_size

    def __len__(self):
        return self._mem_length

    @abstractmethod
    def sample_buffer(self, batch_size):
        raise NotImplementedError

    @abstractmethod
    def store(self):
        raise NotImplementedError
