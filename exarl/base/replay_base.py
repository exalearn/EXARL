from abc import ABC, abstractmethod
import numpy as np

class Replay_Base(ABC):
    """
    Base class of a replay buffer

    Attributes
    ----------
    _capacity : int
        Max allotted number of elements for buffer
    _count : int
        Total number of elements added
    _data : np.array
        The buffer of actual data
    
    """
    def __init__(self, capacity, name=None):
        """
        Parameters
        ----------
        capacity : int
            Max allotted number of elements for buffer
        """
        self._name = name
        self._capacity = capacity
        self._count = 0
        self._data = None

    def _preallocate(self, items, capacity=None):
        """
        Takes a list of representative data and allocates space for buffer
        capacity : int
            Max allotted number of elements for buffer.  If None, _capacity will be used.
        """
        if capacity is None:
            capacity = self._capacity
        temp = []
        for item in items:
            temp.append(np.asarray(item))

        self._data = [np.zeros(dtype=x.dtype, shape=(capacity,) + x.shape)
                    for x in temp]

    def reset(self):
        """
        Resets the replay
        """
        self._data = None

    def get_data_from_indices(self, indices):
        """
        Returns a list of the of data at given indices
        """
        assert self._data is not None, str(self) + " -- not initialized!"
        return [slot[indices] for slot in self._data]

    @property
    def is_full(self) -> bool:
        """
        Returns True if buffer is full
        """
        return self._capacity <= self._count

    @property
    def capacity(self) -> int:
        """
        Returns capacity of buffer
        """
        return self._capacity

    @property
    def size(self) -> int:
        """
        Number of elements in buffer
        """
        return min(self._capacity, self._count)

    def __len__(self):
        """
        Returns number of elements in buffer
        """
        return self.size

    def __repr__(self):
        """
        String representation for buffer
        """
        return 'Replay {}: allocated={}, capacity={}, size={}, added={}'.format(
            self._name, 
            self._data is not None,
            self._capacity,
            min(self._capacity, self._count),
            self._count)

    @abstractmethod
    def sample(self, batch_size):
        """
        Should return a sample of buffer
        """
        raise NotImplementedError

    @abstractmethod
    def store(self):
        """
        Stores new elements and should adds to the count
        """
        raise NotImplementedError

    def get_fake_data(self):
        """
        This is used to get representative data for RMA learner.
        Override to use RMA!
        """
        return None
    
    def bulk_store(self, data):
        assert len(data) == len(self._data) 
        for slot, array in zip(self._data, data):
            for i, item in enumerate(array):
                slot[(self._count + i) % self._capacity] = item
        self._count += len(data[0])