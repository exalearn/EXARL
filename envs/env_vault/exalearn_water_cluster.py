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

import os
import numpy as np
import torch

from ase.io import read

from schnetpack import AtomsData
from schnetpack import AtomsLoader

from schnetpack.environment import SimpleEnvironmentProvider
from schnetpack import AtomsLoader

# Jenna's code
def load_data(atoms, idx=0):
    at = atoms
    new_dataset={}
    atom_positions = at.positions.astype(np.float32)
    atom_positions -= at.get_center_of_mass() 
    environment_provider = SimpleEnvironmentProvider()
    nbh_idx, offsets = environment_provider.get_environment(at)
    new_dataset['_atomic_numbers'] = torch.LongTensor(at.numbers.astype(np.int))
    new_dataset['_positions'] = torch.FloatTensor(atom_positions)
    new_dataset['_cell'] = torch.FloatTensor(at.cell)#.astype(np.float32))
    new_dataset['_neighbors'] = torch.LongTensor(nbh_idx.astype(np.int))
    new_dataset['_cell_offset'] = torch.FloatTensor(offsets.astype(np.float32))
    new_dataset['_idx'] = torch.LongTensor(np.array([idx], dtype=np.int))
    return AtomsLoader([new_dataset], batch_size=1)


def get_activation(name, activation={}):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def get_state_embedding(model,structure):
    data_loader =  load_data(structure, idx=0)
    activation = {}
    logger.debug('torch.no_grad()')
    with torch.no_grad():
        for batch in data_loader:
            model.module.output_modules[0].standardize.register_forward_hook(get_activation('standardize',activation))
            output = model(batch)
        state_embedding = activation['standardize'].cpu().detach().numpy()

    state_embedding = state_embedding.flatten()
    state_order = np.argsort(state_embedding[::3])
    state_embedding = np.sort(state_embedding)
    state_embedding = np.insert(state_embedding, 0, float(np.sum(state_embedding)), axis=0)
    return state_embedding, state_order

class WaterCluster(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):#, cfg='envs/env_vault/env_cfg/env_setup.json'):
        super().__init__()
        """
        Description:
        Toy environment used to test new agents
        """

        # Default
        natoms = 3
        self.init_structure = None
        self.current_structure = None
        self.nclusters = 0.0
        self.inital_state = 0.0
        self.current_state = 0.0
#        self.reward_scale = 2.0
        self.current_energy = 0
        
        self.episode = 0
        self.steps = 0
        self.lowest_energy=100
        #############################################################
        # Setup water molecule application (should be configurable)
        #############################################################
        self.app_dir = cd.run_params['app_dir']
        logger.debug('Using app_dir: {}'.format(self.app_dir))
        self.app_name = 'main.x'
        self.app = os.path.join(self.app_dir, self.app_name)
        # TODO:Needs to be put in cfg
        self.env_input_name = cd.run_params['env_input_name']
        self.env_input_dir = cd.run_params['env_input_dir']
        self.env_input = os.path.join(self.env_input_dir, self.env_input_name)
        self.output_dir = cd.run_params['output_dir']

        # Schnet encodering model
        # TODO: Need to be a cfg and push model to a repo
        self.schnet_model_pfn = cd.run_params['schnet_model_pfn']
        model = torch.load(self.schnet_model_pfn, map_location='cpu')
        self.schnet_model =  torch.nn.DataParallel(model.module)

        # Read initial XYZ file
        (self.init_structure, self.nclusters) = self._load_structure(self.env_input)
        self.inital_state, self.state_order = get_state_embedding(self.schnet_model,self.init_structure) 

        # State of the current setup
        self.current_structure = self.init_structure

        # Env state output: based on the SchetPack molecule energy
        self.embedded_state_size = (self.nclusters+1)*natoms+1
        self.observation_space = spaces.Box(low=-500*np.ones(self.embedded_state_size),
                                            high=np.zeros(self.embedded_state_size), dtype=np.float64)

        # Actions per cluster: cluster id, rotation angle, translation
        self.action_space = spaces.Box(low=np.array([0, 80, 0.5]),
                                       high=np.array([self.nclusters, 120, 0.7]), dtype=np.float64)

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
        logger.debug('Current energy:{}'.format(self.current_energy))

        # Initialize outut
        done = False
        energy = 0  # Default energy
        reward = 0#np.random.normal(-100.0, 0.01)  # Default penalty

        action = action[0]
        natoms = 3
        # Extract actions
        cluster_id = math.floor(action[0])
        # TODO remap cluster_id back to sorting in schnet order[action[0]]
        cluster_id = self.state_order[cluster_id]
        rotation_z = float(action[1])
        translation = float(action[2])  # (x,y,z)

        # Create water cluster from action
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
                logger.info('Atom type not in water...')
        # Calculate bisector vector along two O--H bonds.
        u = np.array(H_coords[0]) - np.array(O_coords)
        v = np.array(H_coords[1]) - np.array(O_coords)
        bisector_vector = (np.linalg.norm(v) * u) + (np.linalg.norm(u) * v)

        # Apply rotation through the z-axis of the water molecule.
        atoms.rotate(rotation_z, v=bisector_vector, center=O_coords)

        # Apply translation
        atoms.translate(translation)

        # Update structure
        for i in range(0, natoms):
            self.current_structure[atom_idx + i].position = atoms[i].position

        # Save structure in xyz format
        new_xyz = os.path.join(self.output_dir,'rotationz_rank{}_episode{}_steps{}.xyz'.format(mpi_settings.agent_comm.rank, self.episode, self.steps))
        try:
            write(new_xyz, self.current_structure, 'xyz', parallel=False)
        except Exception as e:
            logger.debug('Error writing file: {}'.format(e))
            os.remove(new_xyz)
            # 
            done = True
            self.current_state = np.zeros(self.embedded_state_size)
            #logger.debug('Next state:{}'.format(next_state))
            #logger.debug('Hum ... reward - 1:{}'.format(reward))
            return next_state, np.array([reward]), done, {}

        # Run the process
        min_xyz = os.path.join(self.output_dir,'minimum_rank{}.xyz'.format(mpi_settings.agent_comm.rank))
        env_out = subprocess.Popen([self.app, new_xyz, min_xyz], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = env_out.communicate()
        stdout = stdout.decode('latin-1').splitlines()

        # Check for clear problems
        if any("Error in the det" in s for s in stdout):
            logger.debug("\tEnv::step(); !!! Error in the det !!!")
            os.remove(new_xyz)
            done = True
            next_state = np.zeros(self.embedded_state_size)
            return next_state, np.array([reward]), done, {}
        
        # Reward is currently based on the potential energy
        lowest_energy_xyz = ''
        (self.current_structure, self.nclusters) = self._load_structure(min_xyz)
        try:
            energy = float(stdout[-1].split()[-1])
            logger.debug('self.lowest_energy:{}'.format(self.lowest_energy))
            logger.debug('energy:{}'.format(energy))
            if  self.lowest_energy>energy:
                self.lowest_energy=energy
                lowest_energy_xyz = os.path.join(self.output_dir,'rotationz_rank{}_episode{}_steps{}_energy{}.xyz'.format(
                    mpi_settings.agent_comm.rank, self.episode, self.steps,round(self.lowest_energy,4)))
                logger.info("\t Found lower energy:{}".format(energy))

            energy = round(energy, 6)
            reward = self.current_energy - energy     
            #reward = - energy
            if energy==3:
                logger.info('Open - Odd value (3)')
                logger.info(stdout)
                logger.info(stderr)
                logger.info('End - Odd value (3)')
            #reward1 = np.array([round(reward, 6)])
            logger.debug('Pre-step current energy:{}'.format(self.current_energy))
            logger.debug('Energy:{}'.format(energy))
            logger.debug('Reward:{}'.format(reward))

            # Update state information
            self.current_state, self.state_order = get_state_embedding(self.schnet_model,self.current_structure)
            self.current_energy = energy
            # Check with Schnet predictions
            schnet_energy = self.current_state[0]
            energy_mape = np.abs(energy-schnet_energy)/(energy+schnet_energy)
            if energy_mape>0.05:
                logger.debug('Large difference model predict and Schnet MAPE :{}'.format(energy_mape))
        except Exception as e:
            logger.debug('Error with energy value: {}'.format(e))
            #logger.debug('stdout:', stdout)
            #logger.debug('stderr:', stderr)
            # Return values
            done = True
            self.current_state = np.zeros(self.embedded_state_size)
            #logger.debug('Next state:{}'.format(next_state))
            #logger.debug('Hum ... reward - 2:{}'.format(reward))
            #return next_state, np.array([reward]), done, {}

        #self.current_state, self.state_order = get_state_embedding(self.schnet_model,self.current_structure)
        #logger.debug('Schnetpack next state:{}'.format(self.current_state))
        #logger.debug('Next state:{}'.format(self.current_state))
        if lowest_energy_xyz!='':
            os.rename(new_xyz,lowest_energy_xyz)
        else:
            os.remove(new_xyz)
        logger.debug('Reward:{}'.format(reward))
        logger.debug('Energy:{}'.format(energy))
        return self.current_state, reward, done, {}

    def reset(self):
        logger.info("Resetting the environemnts.")
        logger.info("Current lowest energy: {}".format(self.lowest_energy))

        self.episode += 1
        self.steps = 0
        logger.debug('Env::reset(); episode[{0:4d}]'.format(self.episode, self.steps))
        (self.init_structure, self.nclusters) = self._load_structure(self.env_input)
        self.current_structure = self.init_structure

        state_embedding, self.state_order = get_state_embedding(self.schnet_model,self.current_structure)
        self.current_energy = state_embedding[0]
        self.current_state =  state_embedding
        logger.debug('self.current_state shape:{}'.format(self.current_state.shape))
        return self.current_state

    def render(self, mode='human'):
        return 0

    def close(self):
        return 0
