import sys
import numpy as np
import hashlib

import gym
from gym.spaces.utils import flatten


# JS: This implementation was taken from https://www.jmlr.org/papers/volume11/ben-haim10a/ben-haim10a.pdf
class histogram:
    def __init__(self, maxBuckets, minKey=0, maxKey=1E9, search=True):
        self._search = search
        # Set this as opposites for when we add new elements
        self._min = maxKey
        self._max = minKey
        self._maxBuckets = maxBuckets
        self._bins = []

    def findBinMatch(self, key):
        # JS: Binary search
        if self._search:
            left = 0
            right = len(self._bins) - 1
            while left <= right:
                middle = int((left + right) / 2)
                middleValue = self._bins[middle][0]
                if middleValue < key:
                    left = middle + 1
                elif middleValue > key:
                    right = middle - 1
                else:
                    return middle
        else:
            keys = [x[0] for x in self._bins]
            try: 
                keys.index(key)
            except:
                pass
        return -1

    # JS: Finds the lower of the two bins where bin[i].key <= key < bin[i+1].key
    def findBin(self, key):
        # JS: Modified binary search
        if self._search:
            left = 0
            right = len(self._bins) - 1
            middle = right
            
            while left <= right:
                middle = int((left + right) / 2)
                # JS: If this breaks, it means we only have two bins left
                if middle == left:
                    break

                middleValue = self._bins[middle][0]
                # JS: Only update to the middle... We don't want to pass it
                if middleValue < key:
                    left = middle
                elif middleValue > key:
                    right = middle
                else:
                    break
            return middle
        else:
            indices = [i for i, (lower, upper) in enumerate(zip(self._bins, self._bins[1:])) if lower <= key and key < upper]
            if len(indices):
                return indices[0]
        return -1

    def add(self, key, value=1):
        if key < self._min:
            self._min = key
        if key > self._max:
            self._max = key

        # JS: Check if there exists an exact bin
        binIndex = self.findBinMatch(key)
        if binIndex >= 0:
            self._bins[binIndex][1] += value
        else:
            # JS: Add new bin
            self._bins.append([key, value])
            self._bins.sort(key=lambda x:x[0])

            # JS: Do resize if bins are more than allowed
            if len(self._bins) > self._maxBuckets:
                # JS: Find the closest two bins
                diffs = [x1[0] - x0[0] for x0, x1 in zip(self._bins, self._bins[1:])]
                minDiffIndex = diffs.index(min(diffs))

                # JS: Average the bins
                qi, ki = self._bins[minDiffIndex]
                qi_1, ki_1 = self._bins[minDiffIndex + 1]
                self._bins[minDiffIndex][0] =   (qi * ki + qi_1 * ki_1) /  (ki + ki_1)
                self._bins[minDiffIndex][1] = ki + ki_1
                self._bins.pop(minDiffIndex + 1)
                
            # JS: Update min/max if we are being strict
            _min = self._bins[0][0]
            _max = self._bins[-1][0]

    def sum(self, key):
        if key < self._min:
            return 0

        if key >= self._max: 
            return sum([x[1] for x in self._bins])
            

        # JS: The paper only calcs bins within the bin keys
        # To go below we could guess half of bin zero based
        # on paper assumption.
        if key < self._bins[0][0]:
            return self._bins[0][1] / 2

        if key > self._bins[-1][0]:
            return sum([x[1] for x in self._bins]) + (self._bins[-1][1] / 2)
        
        binIndex = self.findBin(key)
        pi, mi = self._bins[binIndex]
        pi_1, mi_1 = self._bins[binIndex+1]
        
        # JS: This calculates the area of an imaginary trapizaoid
        mb = mi + ( ( (mi_1 - mi) / (pi_1 - pi) ) * (key - pi) )
        s = ( (mi + mb) / 2 ) * ( (key - pi) / (pi_1 - pi) )

        # JS: Add up all the bins below us
        s += sum([x[1] for x in self._bins[:binIndex]])

        # JS: Paper assumes that half the bin is above and below this number
        s += mi / 2
        return s

    def getValue(self, key, bucket_range=None):
        if bucket_range is None:
            bucket_range = 0.5
            # Heuristic proposed by Nathan...
            if len(self._bins) > 1:
                bucket_range = (self._max - self._min) / (len(self._bins) - 1)
      
        if key > self._max:
            ret = self.sum(self._max) - self.sum(self._max - bucket_range)    
        elif key < self._min:
            ret = self.sum(self._min + bucket_range) - self.sum(self._min)
        else:
            ret = self.sum(key + bucket_range) - self.sum(key - bucket_range)
        
        # JS: This is because we can over estimate the bucket by assuming
        # half is above... In this case the bin is empty!
        if ret < 0:
            return 0
        return ret

    def bins(self):
        return self._bins

    def getEvenBins(self):
        keys = np.linspace(self._min, self._max, self._maxBuckets)
        values = [self.getValue(x) for x in keys]
        return keys, values

class gym_space_eval:
    '''
    We want to set the space_size to the total number of exps we will run.
    This fits the case where we could int theory have 1 observation per across
    the whole space.  The bins less than that will let us save on space and do
    some interpolation for us!

    We can go higher on the space_size if we are interested in information across
    runs.  In this case maybe there are parts of the observation space not actually
    observed?!?!?! (SHOCK AND AWE)

    This conversation is all assuming a perfect hash which sha256 is not so...
    Also we did not use the internal hash because it shifts across runs 
    for "security"...
    '''
    def __init__(self, space_size, bins, space):
        assert space_size >= bins, "Space size < bin will make bin <= space size"
        self.space = space
        self.space_size = space_size
        self.histogram = histogram(bins, minKey=0, maxKey=space_size)

    def observe(self, observation):
        # JS: Do a hash of the observation
        obj = np.array2string(flatten(self.space, observation))
        obj = obj.encode('utf-8')
        hashobj = hashlib.sha256(obj)
        val_hex = hashobj.hexdigest()
        val_int = int(val_hex, 16)
        
        # JS: Convert to our max space and add to histogram
        key = val_int % self.space_size
        self.histogram.add(key)

    def bins(self, bucket_range=None):
        keys = list(range(self.space_size))
        values = [self.histogram.getValue(x, bucket_range=bucket_range) for x in range(self.space_size)]
        return keys, values
