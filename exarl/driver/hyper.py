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

def test_cmd(trial_number, node_id, epsilon):
    return srun_prefix("python exarl/driver --epsilon_decay_rate_denominator " + str(epsilon), all_nodes[node_id]) 

def parse(output):
    lines = output.split("\n")
    print(lines)
    for line in lines:
        if "Maximum elapsed time" in line:
            temp = line.split(" ")
            return temp[-1]

def inner(trial_number, epsilon):
    my_node = trial_number % len(all_nodes)
    cmd = command(test_cmd(trial_number, my_node, epsilon=epsilon), wait=True)
    time = parse(cmd.out)
    print(time)

def objective(trial):
    my_node = trial.number % len(all_nodes)
    epsilon = trial.suggest_float("epsilon_decay_rate_denominator", 1, 10)
    cmd = command(test_cmd(trial.number, my_node, epsilon=epsilon), wait=True)
    time = parse(cmd.out)
    return time

all_nodes = get_nodes()
print("Python", sys.version, "All Nodes", all_nodes, flush=True)

inner(0, 0.1)

study = optuna.create_study()
study.optimize(objective, n_trials=100, n_jobs=len(all_nodes))

# best_params = study.best_params
# found_ep = best_params["epsilon"]
# print("Found ep: {}".format(found_ep))