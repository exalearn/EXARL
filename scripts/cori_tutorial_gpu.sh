#!/bin/bash
#SBATCH -C gpu
#SBATCH -n 4
#SBATCH --gpus-per-task=1
#SBATCH -c 10
#SBATCH -t 120
#SBATCH -J ExaRL-tensorboard-1g-10
#SBATCH -o ExaRL-tensorboard-1g-10.%j.out
#SBATCH -L cfs
#SBATCH --image=registry.nersc.gov/apg/exarl-ngc:0.1
#SBATCH --gpu-bind=map_gpu:0,1,2,3
set -xe
export SLURM_CPU_BIND="cores"
PYTHONPATH="/global/cscratch1/sd/$USER/ExaRL_tutorial/ExaRL"
output_dir="${PWD}/exarl-output-${SLURM_JOB_ID}"
mkdir ${output_dir}
cd "/global/cscratch1/sd/$USER/ExaRL_tutorial/ExaRL"
#  --env ExaCartPole-v1 --agent DQN-v0 --n_episodes 100 --n_steps 100 \
#  --env ExaBoosterDiscrete-v0 --agent DQN-v0 --n_episodes 2000 --n_steps 50 --booster_data_dir /global/cfs/cdirs/m3363/exarl_data/data_exabooster --data_file data_exabooster
#  --env ExaWaterClusterDiscrete-v0 --agent DDPG-v0 --n_episodes 1000 --n_steps 100 \
#  --env ExaCH-v0 --agent DQN-v0 --n_episodes 500 --n_steps 10 \

srun \
shifter \
python \
  driver/driver.py \
  --output_dir ${output_dir} \
  --env ExaCartPole-v1 --agent DQN-v0 --n_episodes 10 --n_steps 10 \
  --workflow async \
  --batch_size 32 \
  --learning_rate 0.005
