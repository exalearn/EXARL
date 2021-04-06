#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=30
#SBATCH --nodes=7
#SBATCH --ntasks=28
#SBATCH --cpus-per-task=16
#SBATCH --constraint=haswell
#SBATCH -J ExaRL-tensorboard-1g-10
#SBATCH -o ExaRL-tensorboard-1g-10.%j.out
#SBATCH -L cfs
#SBATCH --image=registry.nersc.gov/apg/exarl-ngc:0.1
export OMP_PROC_BIND=true
export OMP_PLACES=threads
export OMP_NUM_THREADS=8

PYTHONPATH="/global/cscratch1/sd/$USER/ExaRL_tutorial/ExaRL"
output_dir="${PWD}/exarl-output-${SLURM_JOB_ID}"
mkdir ${output_dir}
cd "/global/cscratch1/sd/$USER/ExaRL_tutorial/ExaRL"

# These are some other environments to try out
#  --env ExaCartPole-v1 --agent DQN-v0 --n_episodes 100 --n_steps 100 \
#  --env ExaCH-v0 --agent DQN-v0 --n_episodes 100 --n_steps 50 \
#  --env ExaBoosterDiscrete-v0 --agent DQN-v0 --n_episodes 2000 --n_steps 50 --booster_data_dir /global/cfs/cdirs/m3363/exarl_data/data_exabooster --data_file data_exabooster
#  --env ExaWaterCluster-v0 --agent DDPG-v0 --n_episodes 100 --n_steps 50 \

srun --cpu-bind=cores \
shifter \
python \
  driver/driver.py \
  --output_dir ${output_dir} \
  --env ExaCartPole-v1 --agent DQN-v0 --n_episodes 100 --n_steps 100 \
  --workflow async \
  --batch_size 32 \
  --learning_rate 0.005
