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
import gym
from lammps import lammps
import exarl.mpi_settings as mpi_settings


class ExaLammpsEx1(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        # Use CartPole to provide "results"
        self._max_episode_steps = 0
        self.env = gym.make('CartPole-v0')
        self.env._max_episode_steps = self._max_episode_steps
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        # GET VALUES FROM CARTPOLE
        next_state, reward, done, info = 0, -9999, False, 'WrongRank'
        if mpi_settings.env_comm.rank == 0:
            next_state, reward, done, info = self.env.step(action)
        mpi_settings.env_comm.barrier()

        # RUN LAMMPS AS MPI TEST
        lmp = lammps(comm=mpi_settings.env_comm)
        infile = '/people/schr476/ccsdi/usecase3/cdi-thermalconductivityml/in.addtorque'
        lines = open(infile, 'r').readlines()
        for line in lines:
            lmp.command(line)
        lmp.close()

        # RETURN VALUES

        return next_state, reward, done, info

    def reset(self):
        self.env._max_episode_steps = self._max_episode_steps
        return self.env.reset()

    def render(self, mode='human', close=False):
        return self.env.render()

    def set_env(self):
        print('Use this function to set hyper-parameters, if any')
