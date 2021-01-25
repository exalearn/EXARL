import exarl.mpi_settings as mpi_settings
import gym
import time
from mpi4py import MPI
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import utils.log as log
import utils.candleDriver as cd
logger = log.setup_logger(__name__, cd.run_params['log_level'])

from ase.io import read, write
from ase import Atom, Atoms

from math import log10
import subprocess
import os
import math
import tempfile


class WaterCluster(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, cfg='envs/env_vault/env_cfg/env_setup.json'):
        super().__init__()
        """
        Description:
        Toy environment used to test new agents
        """

        # Default
        self.init_structure = None
        self.current_structure = None
        self.nclusters = 0.0
        self.inital_state = 0.0
        self.current_state = 0.0
        self.reward_scale = 2.0

        self.episode = 0
        self.steps = 0
        self.app = os.path.join(self.app_dir, self.app_name)
        self.env_input_name = 'W10_geoms_lowest.xyz'  # 'input.xyz'
        self.env_input = os.path.join(self.app_dir, self.env_input_name)

        env_out = subprocess.Popen([self.app, self.env_input],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = env_out.communicate()
        logger.debug(stdout)
        logger.debug(stderr)
        stdout = stdout.decode('utf-8').splitlines()
        # Initial energy
        self.inital_state = np.array([round(float(stdout[-1].split()[-1]), 6)])

        # Read initial XYZ file
        (self.init_structure, self.nclusters) = self._load_structure(self.env_input)

        # State of the current setup
        self.current_structure = self.init_structure

        # Env state output: potential energy
        self.observation_space = spaces.Box(low=np.array([-500]),
                                            high=np.array([0]), dtype=np.float64)

        # Actions per cluster: cluster id, rotation angle, translation
        self.action_space = spaces.Box(low=np.array([0, 75, 0.3]),
                                       high=np.array([self.nclusters, 105, 0.7]), dtype=np.float64)

    def _load_structure(self, env_input):
        # Read initial XYZ file
        logger.debug('Env Input: {}'.format(env_input))
        structure = read(env_input, parallel=False)
        logger.debug('Structure: {}'.format(structure))
        nclusters = (''.join(structure.get_chemical_symbols()).count("OHH")) - 1
        logger.debug('Number of atoms: %s' % len(structure))
        logger.debug('Number of water clusters: %s ' % (nclusters + 1))
        return (structure, nclusters)

    def step(self, action):
        logger.debug('Env::step()')
        self.steps += 1
        logger.debug('Env::step(); steps[{0:3d}]'.format(self.steps))

        # Initialize outut
        done = False
        energy = 0.0  # Default energy
        reward = np.random.normal(-100.0, 0.01)  # Default penalty
        # target_scale = 200.0  # Scale for calculations

        # print('env action: ',action)
        # Make sure the action is within the defined space
        # isValid = self.action_space.contains(action)
        # if isValid == False:
        #    logger.debug('Env::step(); Invalid action...')
        #    # max_value = np.max(abs(action))
        #    logger.debug(action)
        #    # logger.debug("Reward: %s " % str(-max_value) )
        #    # done=True
        #    # return np.array([0]), np.array(-max_value), done, {}
        #    return np.array([0]), np.array(reward), done, {}

        # Extract actions
        cluster_id = math.floor(action[0])
        rotation_z = float(action[1])
        translation = float(action[2])  # (x,y,z)

        # Create water cluster from action
        natoms = 3
        atom_idx = cluster_id * natoms
        atoms = Atoms()
        for i in range(0, natoms):
            atoms.append(self.current_structure[atom_idx + i])

        # Apply rotate_z action
        # Get coordinates for each atom
        H_coords = []
        for i, atomic_number in enumerate(atoms.get_atomic_numbers()):
            if atomic_number == 8:
                O_coords = atoms.get_positions()[i]
            elif atomic_number == 1:
                H_coords.append(atoms.get_positions()[i])
            else:
                print('Atom type not in water...')
        # Calculate bisector vector along two O--H bonds.
        u = np.array(H_coords[0]) - np.array(O_coords)
        v = np.array(H_coords[1]) - np.array(O_coords)
        bisector_vector = (np.linalg.norm(v) * u) + (np.linalg.norm(u) * v)
        # Apply rotation through the z-axis of the water molecule.
        # TODO: Double check output
        atoms.rotate(rotation_z, v=bisector_vector, center=O_coords)

        # Apply translation
        atoms.translate(translation)

        # Update structure
        for i in range(0, natoms):
            self.current_structure[atom_idx + i].position = atoms[i].position

        # Save structure in xyz format
        # TODO: need to create random name
        #       Now created using the index of rank, episode, and steps
        # write('rotationz_test.xyz',self.current_structure,'xyz',parallel=False)
        # tmp_input='rotationz_test.xyz'
        prefix = 'rotationz_rank{}_episode{}_steps{}.xyz'.format(
            mpi_settings.agent_comm.rank, self.episode, self.steps)
        write(prefix, self.current_structure, 'xyz', parallel=False)
        tmp_input = prefix

        # Run the process
        env_out = subprocess.Popen([self.app, tmp_input], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = env_out.communicate()
        logger.debug(stdout)
        logger.debug(stderr)
        stdout = stdout.decode('utf-8').splitlines()

        # Check for clear problems
        if any("Error in the det" in s for s in stdout):
            logger.debug("\tEnv::step(); !!! Error in the det !!!")
            done = True
            return np.array([0]), np.array([reward]), done, {}

        # Reward is currently based on the potential energy
        try:
            logger.debug("\tEnv::step(); stdout[{}]".format(stdout))
            energy = float(stdout[-1].split()[-1])
            logger.debug("\tEnv::step(); energy[{}]".format(energy))
            energy = round(energy, 6)
            reward = self.current_state[0] - energy
            reward = np.array([round(reward, 6)])
        except:
            print('stdout:', stdout)
            print('stderr:', stderr)
            return np.array([0]), np.array([reward]), done, {}

        # Check if the structure is the same
        if round(self.current_state[0], 6) == energy:
            logger.debug('Env::step(); Same state ... terminating')
            done = True
            return np.array([energy]), np.array(reward), done, {}

        # If valid action and simulation
        # reward= (energy/target_scale - 1.0)**2
        # Current reward is based on the energy difference between
        #     the current state and the new state
        # delta = energy-self.current_state[0]
        # delta = energy-self.inital_state[0]
        # reward = np.exp(-delta/5.0)

        # logger.info('Current state: %s' % self.current_state)
        # logger.info('Next State: %s' % np.array([energy]))
        # logger.info('Reward: %s' % reward)
        # Update current state
        self.current_state = np.array([energy])
        return self.current_state, reward, done, {}

    def reset(self):
        self.episode += 1
        self.steps = 0
        logger.debug('Env::reset(); episode[{0:4d}]'.format(self.episode, self.steps))
        (self.init_structure, self.nclusters) = self._load_structure(self.env_input)
        self.current_structure = self.init_structure
        self.current_state = self.inital_state
        # Start a new random starting point
        # random_action = self.action_space.sample()
        # self.step(random_action)
        self.init_structure = self.current_structure
        self.inital_state = self.current_state
        logger.info("Resetting the environemnts.")
        logger.info("New initial state: %s" % str(self.inital_state))
        return self.current_state

    def render(self, mode='human'):
        return 0

    def close(self):
        return 0
