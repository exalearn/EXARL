#!/bin/bash

######## Double check this works for your set up ########
module purge
module load gcc/10.2.0
module load spectrum-mpi/10.3.1.2-20200121
module load cuda/10.2.89
# Latest open-ce / default June 24 2021
module load open-ce/1.1.3-py38-0

######## This is user specific conda env! ########
conda activate exaRL_clone
export PYTHONPATH=/ccs/home/suet688/ExaRL/gym:$PYTHONPATH
export EXARL_ROOT="/ccs/home/suet688/ExaRL/EXARL"

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
WHICH_ENV="-b"

for NODES in 1 2 4 8 16 32 64
do
bsub -P AST153 -J RunEXARL -W 2:00 -nnodes $NODES -e $ERR_LOG -o $OUT_LOG -alloc_flags "gpumps" "sh ${EXARL_ROOT}/scripts/submit_summit.lsf -n $NODES $WHICH_ENV &"
done
