#!/bin/bash

#Submit this script with: sbatch filename

#SBATCH --time=03:00:00   # walltime
#SBATCH --time-min=60:00   # minimum walltime
#SBATCH --partition=scaling   # partition name
#SBATCH --mail-user=schenna@lanl.gov   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --no-requeue   # do not requeue when preempted and on node failure
#SBATCH --signal=23@60  # send signal to job at [seconds] before end


# LOAD MODULEFILES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load openmpi/3.1.0-gcc_7.3.0
module load anaconda/Anaconda3.2019.10
#module load cuda/10.1
source activate exarl
export SLURM_CPU_BIND_VERBOSE=1
#srun --exclusive --nodes 2 --ntasks 17 --job-name $4 --output $5 --error $5 python exarl/driver --workflow $6 --n_steps $7 --n_episodes $8 --agent $9 --env $10 --output_dir $11 --learner_procs $12 --process_per_env $13 --model_type $14 --action_type $15 --batch_size $16

mpirun -np $1 python exarl/driver --workflow $2 --n_steps $3 --n_episodes $4 --agent $5 --env $6 --output_dir $7 --learner_procs $8 --process_per_env $9 --model_type ${10} --action_type ${11} --batch_size ${12}
  
