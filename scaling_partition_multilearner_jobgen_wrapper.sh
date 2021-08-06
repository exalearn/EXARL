#!/bin/bash

agent="MLDQN-v0"
nsteps="200"
nepisodes="10000"
workflow_list="mlasync mlrma mlrma-v1"
env="ExaBoosterDiscrete-v0"
ppe="1"
modeltype="LSTM"
actiontype="fixed"
bz="32"

#For scaling experiments we are using 1:3 learner to actor ratio and a total of 32 processes per node
#Hence it will be 32 processes on 1 node (8 learners, 24 actors), 64 on 2 node (16l;28a) so on
#Note: if you want to compare these workflows with single learner workflows, keep the actor count same and total learners as 1

#Processor count 
np_list="32 64 128 256 512 1024 2048 4096"

npernode="32" #processes per node

basedir="./results_dir/test"    #specify the name of the base folder or directory under which you want to generate output directory

for workflow in $workflow_list
do
	for np in $np_list
	do
		(( lp=( np )/4 ))
		(( actor=( lp )*3 ))
		(( node=( np )/32 ))
		outputdir="${basedir}/multi-node/scaling_partition/${workflow}/${agent}_${modeltype}_${bz}_${env}_${actiontype}_${nepisodes}_${nsteps}_lp${lp}_actors${actor}_ppe${ppe}"
		mkdir -p "$outputdir"
		echo "Creating output directory ${outputdir}"
		jobname="${workflow}_${agent}_${modeltype}_${bz}_${env}_${actiontype}_${nepisodes}_${nsteps}_lp${lp}_actors${actor}_ppe${ppe}"
        	log="${outputdir}/output"
	        #echo $np
        	#echo $lp
	        #echo $actor
	        #echo $npernode
		#echo $node
		#sbatch ./job_submission_script.sh $node $np $npernode $jobname $log $workflow $nsteps $nepisodes $agent $env $outputdir $lp $ppe $modeltype $actiontype $bz
		sbatch --partition scaling --exclusive --nodes $node --ntasks $np --ntasks-per-node $npernode --job-name $jobname --output $log --error $log ./job_submission_script.sh $np $workflow $nsteps $nepisodes $agent $env $outputdir $lp $ppe $modeltype $actiontype $bz

	done
done
