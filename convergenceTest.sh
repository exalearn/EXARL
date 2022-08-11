#!/usr/bin/bash
# source ../env3.sh
# set +x

# Set the results directory
resultsDir="/people/suet688/exaLearn/ExaRL/convergenceTest/cartpole_batch"

# Used to indicate what nodes we are running on
# nodes=($(scontrol show hostnames))
# numNodes=${#nodes[@]}

partition="slurm"
extra_slurm_args="-x node42"
numNodes=1
max_episode=100000
max_step=100
env="--env ExaCartPoleStatic-v0"

base_batch_size=32
episode_block=( True False )
batch_frequency=( 1 2 5 10 50 -1 )
train_frequency=( 1 2 5 10 100)
ranks=( 2 5 9 17 24 )
profile="--profile intro"
mkdir -p ${resultsDir}

# Runs a sweep of the interesting parameters
for ep in "${episode_block[@]}"
do
    for ba in "${batch_frequency[@]}"
    do
        for t in "${train_frequency[@]}"
        do
            for r in "${ranks[@]}"
            do
                # batch_size=$((base_batch_size * 1))
                batch_size=$(( base_batch_size * (r - 1) ))
                command="srun -N $numNodes -n ${r} $extra_slurm_args python ./exarl/driver ${profile} --output_dir ${resultsDir}/async_${ep}_${ba}_${r}_${t} --agent BSUITE-BASE-v1 --workflow async --episode_block ${ep} --batch_frequency ${ba} --train_frequency ${t} --n_episodes $max_episode --n_steps $max_step --batch_size $batch_size $env &> ${resultsDir}/async_${ep}_${ba}_${r}_${t}_${batch_size}.txt &"
                ./bsuite/throttle.pl $partition "$command"
                # srun -N $numNodes -n ${r} $extra_slurm_args python ./exarl/driver ${profile} --output_dir ${resultsDir}/async_${ep}_${ba}_${r}_${t} --agent BSUITE-BASE-v1 --workflow async --episode_block ${ep} --batch_frequency ${ba} --train_frequency ${t} --n_episodes $max_episode --n_steps $max_step --batch_size $batch_size $env &> ${resultsDir}/async_${ep}_${ba}_${r}_${t}_${batch_size}.txt & 
            done
        done
    done
done
