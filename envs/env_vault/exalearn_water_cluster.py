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

import csv

from ase.io import read, write
from ase import Atom, Atoms

from math import log10
import subprocess
import os
import math
import tempfile
from shutil import copyfile
import torch

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

def update_cluster(structure, action=[0,0,0]):
    temp_struct=structure.copy()
    natoms = len(structure)

    # Actions
    cluster_id = math.floor(action[0])
    rotation_z = float(action[1])
    translation_action = float(action[2])  # applied equally to (x,y,z)

    # Create water cluster from action
    O_idx = cluster_id *3
    atoms = Atoms()
    for i in range(3):
        atoms.append(temp_struct[O_idx + i])

    # rotation
    O_coords = structure[O_idx].position
    H_coords = [structure[O_idx+1].position, structure[O_idx+2].position]
    # Calculate bisector vector along two O--H bonds.
    u = np.array(H_coords[0]) - np.array(O_coords)
    v = np.array(H_coords[1]) - np.array(O_coords)
    bisector_vector = (np.linalg.norm(v) * u) + (np.linalg.norm(u) * v)

    # Apply rotation through the z-axis of the water molecule.
    atoms.rotate(rotation_z, v=bisector_vector, center=O_coords)
    atoms.translate(translation_action)

    # Update structure
    for i in range(3):
        structure[O_idx + i].position = atoms[i].position

    return structure

def write_structure(PATH, structure, energy=0.0):
    nice_struct = [str(n).replace('[','').replace(']','') for n in structure.positions]
    nice_struct = ['  '.join(x) for x in list(zip(structure.get_chemical_symbols(),nice_struct))]
    pos='\n'.join(nice_struct)
    with open(PATH, 'w') as f:
        f.writelines(str(len(structure))+'\n')
        f.writelines(str(energy)+'\n')
        f.writelines(pos)

def write_csv(path, rank, data):
    data = [str(x) for x in data]
    data = ','.join(data)
    with open(os.path.join(path,'rank{}.csv'.format(rank)), 'a') as f:
        f.writelines(data+'\n')


def findWater(geom, Num):
    sortedGeom = []
    sortedGeom.append("O  " + str(geom[Num])+"\n")
    for line in geom:
        dist = np.linalg.norm(geom[Num] - line)
        if dist > 0.01 and dist < 1.20: #assuming angstroms
            sortedGeom.append("H  " + str(line) + "\n")
    return sortedGeom

def read_geoms(geom):
    '''
    Reads a file containing a large number of XYZ formatted files concatenated together and splits them
    into an array of arrays of vectors (MxNx3) where M is the number of geometries, N is the number of atoms,
    and 3 is from each x, y, z coordinate.
    '''
    #atoms = []
    iFile = open(geom).read()
    allCoords = []
    atomLabels = []
    header = []
    with open(geom) as ifile:
        #iMol = 0
        while True:
            atomLabels__ = []
            line = ifile.readline()
            if line.strip().isdigit():
                natoms = int(line)
                title = ifile.readline()
                header.append(str(natoms) + '\n' + title)
                coords = np.zeros([natoms, 3], dtype="float64")
                for x in coords:
                    line = ifile.readline().split()
                    atomLabels__.append(line[0])
                    temp_list = line[1:4]
                    temp_list2 = [float(o) for o in temp_list]
                    x[:] = temp_list2
                allCoords.append(coords)
                atomLabels.append(atomLabels__)
                #iMol += 1
            if not line:
                break
    return header, atomLabels, allCoords

def fix_structure_file(infile):
    '''reorders xyz to group molecule atoms together'''
    header, atomLabels, geoms = read_geoms(infile)
    sortedGeoms = []
    for iGeom, geom in enumerate(geoms):
        sortedGeom = []
        OIndices = []
        #determine which indices have an oxygen atom
        for i in range(len(atomLabels[iGeom])):
            if atomLabels[iGeom][i] == "O" or atomLabels[iGeom][i] == "o":
                OIndices.append(i)

        #loop through findWater sorting the geometry
        for line in OIndices:
            sortedGeom.append(findWater(geom, line))
        sortedGeoms.append(sortedGeom)

    # rewrite xyz file
    f = open(infile, 'w')
    for iGeom, geom in enumerate(sortedGeoms):
        f.write(header[iGeom])
        for line in geom:
            printable = "".join(str(x) for x in line).replace("[","").replace("]","")
            printable = printable[:-1]
            f.write(printable)
            f.write("\n")

def read_energy(xyz):
    with open(xyz, "r") as f:
        lines=f.readlines()
    return float(lines[1])

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
        self.initial_energy = 0 
        self.current_energy = 0
        self.episode = 0
        self.steps = 0
        self.lowest_energy = 0

        #############################################################
        # Setup water molecule application (should be configurable)
        #############################################################
        self.app_dir = cd.run_params['app_dir']
        logger.debug('Using app_dir: {}'.format(self.app_dir))
        self.app_name = 'main.x'
        self.app = os.path.join(self.app_dir, self.app_name)
        self.env_input_name = cd.run_params['env_input_name']
        self.env_input_dir = cd.run_params['env_input_dir']
        self.env_input = os.path.join(self.env_input_dir, self.env_input_name)
        self.output_dir = cd.run_params['output_dir']

        # Schnet encodering model
        self.schnet_model_pfn = cd.run_params['schnet_model_pfn']
        model = torch.load(self.schnet_model_pfn, map_location='cpu')
        self.schnet_model =  torch.nn.DataParallel(model.module)

        # Read initial XYZ file
        (init_ase, self.nclusters) = self._load_structure(self.env_input)
        self.inital_state, self.state_order = get_state_embedding(self.schnet_model,init_ase) 
        self.initial_energy = read_energy(self.env_input)#self.inital_state[0]

        # State of the current setup
        self.init_structure = self.env_input
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
        action = action[0]
        logger.debug('Action:{}'.format(action))

        # Initialize outut
        done = False
        energy = 0  # Default energy
        reward = np.random.normal(-10.0, 0.01)  # Default penalty

        #action = action[0]
        natoms = 3
        # Extract actions
        cluster_id = math.floor(action[0])
        cluster_id = self.state_order[cluster_id]
        rotation_z = round(float(action[1]),2)
        translation = round(float(action[2]),4)  # (x,y,z)
        actions = [cluster_id, rotation_z, translation]

        # read in structure as ase atom object
        try:
            current_ase = read(self.current_structure, parallel=False)
        except:
            done = True
            self.current_state = np.zeros(self.embedded_state_size)
            #reward = 0
            self.current_energy = -1
            write_csv(self.output_dir, mpi_settings.agent_comm.rank, [self.nclusters, mpi_settings.agent_comm.rank, self.episode, self.steps, cluster_id, rotation_z, translation, self.current_energy, self.current_state[0], reward, done])
            return self.current_state, reward, done, {}

        # take actions
        current_ase = update_cluster(current_ase, actions)

        # Save structure in xyz format
        self.current_structure = os.path.join(self.output_dir,'rotationz_rank{}_episode{}_steps{}.xyz'.format(mpi_settings.agent_comm.rank, self.episode, self.steps))
        write_structure(self.current_structure, current_ase)
        fix_structure_file(self.current_structure)

        # Run the process
        min_xyz = os.path.join(self.output_dir,'minimum_rank{}.xyz'.format(mpi_settings.agent_comm.rank))
        env_out = subprocess.Popen([self.app, self.current_structure, min_xyz], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = env_out.communicate()
        stdout = stdout.decode('latin-1').splitlines()

        # Check for clear problems
        if any("Error in the det" in s for s in stdout):
            logger.debug("\tEnv::step(); !!! Error in the det !!!")
            os.remove(self.current_structure)
            done = True
            self.current_state = np.zeros(self.embedded_state_size)
            #reward = 0
            self.current_energy = -2
            write_csv(self.output_dir, mpi_settings.agent_comm.rank, [self.nclusters, mpi_settings.agent_comm.rank, self.episode, self.steps, cluster_id, rotation_z, translation, self.current_energy, self.current_state[0], reward, done])
            return self.current_state, reward, done, {}
        
        # Reward is currently based on the potential energy
        lowest_energy_xyz = ''
        current_ase = read(min_xyz, parallel=False)

        try:
            energy = float(stdout[-1].split()[-1])
            logger.debug('lowest_energy:{}'.format(self.lowest_energy))
            logger.debug('energy:{}'.format(energy))
            if  self.lowest_energy>energy:
                self.lowest_energy=energy
                lowest_energy_xyz = os.path.join(self.output_dir,'rotationz_rank{}_episode{}_steps{}_energy{}.xyz'.format(
                    mpi_settings.agent_comm.rank, self.episode, self.steps,round(self.lowest_energy,4)))
                logger.info("\t Found lower energy:{}".format(energy))

            energy = round(energy, 6)

            # End episode if the structure is unstable
            if energy > self.initial_energy * 0.5:
                done = True
                self.current_state = np.zeros(self.embedded_state_size)
                write_csv(self.output_dir, mpi_settings.agent_comm.rank, [self.nclusters, mpi_settings.agent_comm.rank, self.episode, self.steps, cluster_id, rotation_z, translation, self.current_energy, self.current_state[0], reward, done])
                return self.current_state, reward, done, {}


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
            self.current_state, self.state_order = get_state_embedding(self.schnet_model,current_ase)
            # Check with Schnet predictions
            schnet_energy = self.current_state[0]
            energy_mape = np.abs(energy-schnet_energy)/(energy+schnet_energy)
            if energy_mape>0.05:
                logger.debug('Large difference model predict and Schnet MAPE :{}'.format(energy_mape))

            # Set reward to normalized SchNet energy (first value in state)
            reward = (self.current_energy - energy ) #/ self.initial_energy

            # Update current energy
            self.current_energy = energy

        except Exception as e:
            logger.debug('Error with energy value: {}'.format(e))
            #logger.debug('stdout:', stdout)
            #logger.debug('stderr:', stderr)
            # Return values
            done = True
            self.current_state = np.zeros(self.embedded_state_size)
            self.current_energy = -2
            #reward = 0

        write_structure(self.current_structure, current_ase, self.current_energy)
        fix_structure_file(self.current_structure)

        if lowest_energy_xyz!='':
            copyfile(self.current_structure,lowest_energy_xyz)


        write_csv(self.output_dir, mpi_settings.agent_comm.rank, [self.nclusters, mpi_settings.agent_comm.rank, self.episode, self.steps, cluster_id, rotation_z, translation, self.current_energy, self.current_state[0], reward, done])

        logger.debug('Reward:{}'.format(reward))
        logger.debug('Energy:{}'.format(energy))
        return self.current_state, reward, done, {}

    def reset(self):
        logger.info("Resetting the environemnts.")
        logger.info("Current lowest energy: {}".format(self.lowest_energy))

        # delete all files from last episode of the rank (not lowest energy files)
        files_in_directory = os.listdir(self.output_dir)
        filtered_files = [file for file in files_in_directory if (file.endswith(".xyz")) and ('rank{}_'.format(mpi_settings.agent_comm.rank) in file) and ('-' not in file)]
        for file in filtered_files:
            path_to_file = os.path.join(self.output_dir, file)
            os.remove(path_to_file)

        self.episode += 1
        self.steps = 0
        logger.debug('Env::reset(); episode[{0:4d}]'.format(self.episode, self.steps))
        (init_ase, self.nclusters) = self._load_structure(self.env_input)
        self.current_structure = self.init_structure

        state_embedding, self.state_order = get_state_embedding(self.schnet_model,init_ase)
        self.initial_energy = read_energy(self.env_input)#state_embedding[0] 
        self.current_energy = self.initial_energy
        self.current_state =  state_embedding
        logger.debug('self.current_state shape:{}'.format(self.current_state.shape))
        return self.current_state

    def render(self, mode='human'):
        return 0

    def close(self):
        return 0
