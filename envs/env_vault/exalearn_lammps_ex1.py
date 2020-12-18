import gym
from lammps import lammps
from exarl.comm_base import ExaComm


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
        if ExaComm.env_comm.rank == 0:
            next_state, reward, done, info = self.env.step(action)
        ExaComm.env_comm.barrier()

        # RUN LAMMPS AS MPI TEST
        lmp = lammps(comm=ExaComm.env_comm)
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
