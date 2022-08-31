#!/usr/bin/env python
#SBATCH --time=48:00:00
#SBATCH --exclusive

import sys
import subprocess
import optuna

class command:
    def __init__(self, cmd, wait=False):
        self.cmd = cmd
        self.sp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.done = False
        self.ret = None
        self.out = None
        self.err = None
        
        if wait:
            self.wait()

    def wait(self):
        self.ret = self.sp.wait()
        self.out, self.err = self.sp.communicate()

def get_nodes():
    cmd = command("scontrol show hostnames", wait=True)
    return [x for x in cmd.out.split('\n') if len(x) > 0]

def srun_prefix(cmd, nodeId, nodes=1, procs=1):
    return " ".join(["srun -N", str(nodes), "-n", str(procs), "-w", nodeId, cmd])

def test_cmd(trial_number, node_id):
    return srun_prefix("printenv | grep SLURM >> out_" + str(node_id) + "_" + str(trial_number) + ".txt", all_nodes[node_id]) 

def objective(trial):
    my_node = trial.number % len(all_nodes)
    command(test_cmd(trial.number, my_node), wait=True)
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2

all_nodes = get_nodes()
print("Python", sys.version, "All Nodes", all_nodes, flush=True)

study = optuna.create_study()
study.optimize(objective, n_trials=5, n_jobs=len(all_nodes))

best_params = study.best_params
found_x = best_params["x"]
print("Found x: {}, (x - 2)^2: {}".format(found_x, (found_x - 2) ** 2))