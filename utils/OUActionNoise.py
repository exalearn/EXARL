import os
import numpy as np
import random
from datetime import datetime


class OUActionNoise2:
    def __init__(self, mean=0, start_std=0.15, stop_std=0.05, damping=0.005):
        self.mean = mean
        self.start_std = start_std
        self.stop_std = stop_std
        self.damping = damping
        self.reset()

    def __call__(self):
        random.seed(datetime.now())
        random_data = os.urandom(4)
        np.random.seed(int.from_bytes(random_data, byteorder="big"))
        dx = self.damping * (self.mean - self.x_prev)
        x = (self.x_prev + dx)
        self.x_prev = x
        return np.random.normal(0, x, 1) + np.random.normal(0, self.stop_std, 1)

    def reset(self):
        self.x_prev = self.start_std


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        random.seed(datetime.now())
        random_data = os.urandom(4)
        np.random.seed(int.from_bytes(random_data, byteorder="big"))
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
