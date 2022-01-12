# This material was prepared as an account of work sponsored by an agency of the
# United States Government.  Neither the United States Government nor the United
# States Department of Energy, nor Battelle, nor any of their employees, nor any
# jurisdiction or organization that has cooperated in the development of these
# materials, makes any warranty, express or implied, or assumes any legal
# liability or responsibility for the accuracy, completeness, or usefulness or
# any information, apparatus, product, software, or process disclosed, or
# represents that its use would not infringe privately owned rights. Reference
# herein to any specific commercial product, process, or service by trade name,
# trademark, manufacturer, or otherwise does not necessarily constitute or imply
# its endorsement, recommendation, or favoring by the United States Government
# or any agency thereof, or Battelle Memorial Institute. The views and opinions
# of authors expressed herein do not necessarily state or reflect those of the
# United States Government or any agency thereof.
#                 PACIFIC NORTHWEST NATIONAL LABORATORY
#                            operated by
#                             BATTELLE
#                             for the
#                   UNITED STATES DEPARTMENT OF ENERGY
#                    under Contract DE-AC05-76RL01830
import os
import numpy as np
import random
from datetime import datetime


class OUActionNoise2:
    """Calculates Ornstein-Uhlenbeck process noise.
            Attributes
        ----------
        mean : float
        start_std : float
            standard deviation default value
        stop_std : float
            standard deviation limit
        damping : float
            rate at which the noise trajectory is damped towards the mean
    """    
    def __init__(self, mean=0, start_std=0.15, stop_std=0.05, damping=0.005):
        """Initialize parameters to noise calculation.

        Parameters
        ----------
        mean : float, optional
            by default 0
        start_std : float, optional
            standard deviation default value, by default 0.15
        stop_std : float, optional
            standard deviation limit, by default 0.05
        damping : float, optional
            by default 0.005
        """        
        self.mean = mean
        self.start_std = start_std
        self.stop_std = stop_std
        self.damping = damping
        self.reset()

    def __call__(self):
        """Generate noise

        Returns
        ----------
        float
            noise
        """        
        random.seed(datetime.now())
        random_data = os.urandom(4)
        np.random.seed(int.from_bytes(random_data, byteorder="big"))
        dx = self.damping * (self.mean - self.x_prev)
        x = (self.x_prev + dx)
        self.x_prev = x
        return np.random.normal(0, x, 1) + np.random.normal(0, self.stop_std, 1)

    def reset(self):
        """Reset noise generator to start_std
        """        
        self.x_prev = self.start_std


class OUActionNoise:
    """Calculates Ornstein-Uhlenbeck process noise.
    """    
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        """[summary]

        Parameters
        ----------
        mean : float
    
        std_deviation : float
            [
        theta : float, optional
             by default 0.15
        dt : float, optional
            by default 1e-2
        x_initial : float, optional
            [by default None
        """        
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        """Generate noise

        Returns
        ----------
        float
            noise
        """      
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
        """Reset noise generator to x_initial or 0's
        """     
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
