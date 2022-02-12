#!/bin/bash

######## Double check this works for your set up ########
module purge
module load cuda/11.0.3
module load open-ce/1.4.0-py37-0
module load git-lfs

######## This is user specific conda env! ########
conda activate exarl_summit
export PYTHONPATH=/ccs/home/vinayr/ExaLearn/EXARL:$PYTHONPATH
export EXARL_ROOT="/ccs/home/vinayr/ExaLearn/EXARL"

# Print time with timezone
export RUNDATE=`date +"%FT%H%M%z"`
# Make sure this is where you want your results
export OUT_DIR=/gpfs/alpine/scratch/${USER}/ast153/results_${RUNDATE}
mkdir -p $OUT_DIR

export ERR_LOG=${OUT_DIR}/error_${RUNDATE}.log
export OUT_LOG=${OUT_DIR}/output_${RUNDATE}.log

# These are the arguments for the environments
# a - EXA_ALL
# c - EXA_CARTPOLE
# p - EXA_BCP
# b - EXA_BOOSTER
# w - EXA_WATER_CLUSTER
WHICH_ENV="-c"

export TASKS_PER_RS=1
export CPUS_PER_RS=7
export GPUS_PER_RS=1
export RS_PER_HOST=6
export WORKFLOW=async

#for NODES in 1 2 4 8 16 32 64
for NODES in 1
do
export NUM_RES_SET=$((NODES*6))
bsub -P AST153 -J RunEXARL -W 2:00 -nnodes $NODES -e $ERR_LOG -o $OUT_LOG -alloc_flags "gpumps" "sh ${EXARL_ROOT}/scripts/submit_summit.lsf -n $NODES $WHICH_ENV &"
done
