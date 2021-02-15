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
    #atoms = atoms#read(datapath, index=f'{idx}:{idx+1}')
    at = atoms#[0]
    new_dataset={}
    atom_positions = at.positions.astype(np.float32)
    atom_positions -= at.get_center_of_mass() 
    environment_provider = SimpleEnvironmentProvider()
    nbh_idx, offsets = environment_provider.get_environment(at)
    new_dataset['_atomic_numbers'] = torch.LongTensor(at.numbers.astype(np.int))
    new_dataset['_positions'] = torch.FloatTensor(atom_positions)
    new_dataset['_cell'] = torch.FloatTensor(at.cell.astype(np.float32))
    new_dataset['_neighbors'] = torch.LongTensor(nbh_idx.astype(np.int))
    new_dataset['_cell_offset'] = torch.FloatTensor(offsets.astype(np.float32))
    new_dataset['_idx'] = torch.LongTensor(np.array([idx], dtype=np.int))
    return AtomsLoader([new_dataset], batch_size=1)


def get_activation(name, model, activation={}):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook#, activation

def get_state_embedding(model,structure):
    data_loader =  load_data(structure, idx=0)
    activation = {}
    logger.debug('torch.no_grad()')
    with torch.no_grad():
        for batch in data_loader:
            model.module.output_modules[0].standardize.register_forward_hook(get_activation('standardize',model,activation))
            output = model(batch)
        state_embedding = activation['standardize'].cpu().detach().numpy()
    return state_embedding

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

        #############################################################
        # Setup water molecule application (should be configurable)
        #############################################################
        self.app_dir = '/gpfs/alpine/ast153/proj-shared/pot_ttm/'
        self.app_name = 'main.x'
        self.app = os.path.join(self.app_dir, self.app_name)
        self.env_input_name = 'W10_geoms_lowest.xyz'  # 'input.xyz'
        self.env_input = os.path.join(self.app_dir, self.env_input_name)

        # Schnet encodering model
        self.schnet_model_pfn = '/gpfs/alpine/ast153/proj-shared/schnet_encoder/best_model'
        model = torch.load(self.schnet_model_pfn, map_location='cpu')
        self.schnet_model =  torch.nn.DataParallel(model.module)
        # TODO: Migrate the rest of the code here
        
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

        # Env state output: based on the SchetPack molecule energy
        self.observation_space = spaces.Box(low=np.zeros(self.nclusters+1)*(-500),
                                            high=np.zeros(self.nclusters+1), dtype=np.float64)

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
                logger.info('Atom type not in water...')
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
        new_xyz = 'rotationz_rank{}_episode{}_steps{}.xyz'.format(mpi_settings.agent_comm.rank, self.episode, self.steps)
        logger.debug('new_xyz: {}'.format(new_xyz))
#        new_xyz_pfn = cd.run_params['output_dir'] + '/xyz/' + new_xyz
        #new_xyz_pfn = cd.run_params['output_dir'] + '/' + new_xyz
        logger.debug('new_xyz: {}'.format(new_xyz))
        try:
            write(new_xyz, self.current_structure, 'xyz', parallel=False)
        except Exception as e:
            logger.debub('Error writing file: {}'.format(e))

        # Run the process
        env_out = subprocess.Popen([self.app, new_xyz], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = env_out.communicate()
        #logger.debug('stdout:',stdout)
        #logger.debug('stderr:',stderr)
        # TODO: Move xyz
        #new_xyz_pfn = cd.run_params['output_dir'] + '/' + new_xyz
        #logger.debug('new_xyz_pfn: {}'.format(new_xyz_pfn))
        #try:
        #    write(new_xyz_pfn, self.current_structure, 'xyz', parallel=False)
        #except Exception as e:
        #    logger.error('Error writing file: {}'.format(e))
        #stdout = stdout.decode('utf-8').splitlines()

        # Get SchetPack embedding
        logger.debug('load_data()')
        data_loader =  load_data(atoms, idx=0)
        activation = {}
        logger.debug('torch.no_grad()')
        with torch.no_grad():
            for batch in data_loader:
                self.schnet_model.module.output_modules[0].standardize.register_forward_hook(get_activation('standardize',activation))
                output = self.schnet_model(batch)
        embedding = activation['standardize']
        logger.info('ScnetPack embedding:{}'.format(embedding))
        # TODO: We need a method to input(xyz) -> Model -> output(encoding)
        #dataloader = get_data_loader(new_xyz)
        #activation = get_schnet_activation(dataloader, self.schnet_model, 'cpu')
    
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
        logger.info('Current_structure :{}'.format((self.current_structure)))

        self.init_structure = self.current_structure
        self.inital_state = self.current_state
        logger.info("Resetting the environemnts.")
        

        state_embedding = get_state_embedding(self.schnet_model,self.current_structure)
        ##
        #logger.debug('load_data()')
        #data_loader =  load_data(self.current_structure, idx=0)
        #activation = {}
        #logger.debug('torch.no_grad()')
        #with torch.no_grad():
        #    for batch in data_loader:
        #        self.schnet_model.module.output_modules[0].standardize.register_forward_hook(get_activation('standardize',self.schnet_model,activation))
        #        output = self.schnet_model(batch)
        #embedding = activation['standardize'].cpu().detach().numpy()
        #logger.info('ScnetPack embedding type: {}'.format(type(embedding)))
        #logger.info('ScnetPack embedding: {}'.format(embedding))
        self.inital_state = state_embedding
        logger.info("New initial state: %s" % str(self.inital_state))
        logger.info("New initial energy: %s" % str(np.sum(self.inital_state)))
        self.current_state = self.inital_state

        return self.current_state

    def render(self, mode='human'):
        return 0

    def close(self):
        return 0
