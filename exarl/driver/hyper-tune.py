#!/usr/bin/env python
#SBATCH --time=48:00:00
#SBATCH --exclusive

import sys
import subprocess
import optuna
import json
import numpy as np
import matplotlib.pyplot as plt


objectives = {
    "last_rolling_reward": lambda x: x[0],
    "reward_per_time": lambda x: x[0]/x[1],
    "reward_per_episode": lambda x: x[0]/x[2]
}

samplers = {
    "tpe": optuna.samplers.TPESampler,
    "random": optuna.samplers.RandomSampler,
    "cmaes": optuna.samplers.CmaEsSampler,
    "qmc": optuna.samplers.QMCSampler,
}

class Optimizer:
    def __init__(self, params, objective_func, sampler, n_trials=100, exp="EXP000"):
        self.params = params
        self.objective_func = objective_func
        self.sampler = sampler
        self.n_trials = n_trials
        self.exp = exp
        self.all_nodes = self.get_nodes()
        # JS: Leave the first node for optuna
        if len(self.all_nodes) > 1:
            self.all_nodes = self.all_nodes[1:]
        self.log = [[0, 0]] 

    def optimize(self):
        #print("Python", sys.version, "All Nodes", all_nodes, flush=True)
        print("STUDY: ", self.sampler, self.n_trials)
        study = optuna.create_study(direction="maximize", sampler=self.sampler())
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=len(self.all_nodes))

        best_params = study.best_params
        return [(p, best_params[p]) for p in best_params], self.log
        

    def get_nodes(self):
        cmd = command("scontrol show hostnames", wait=True)
        return [x for x in cmd.out.split('\n') if len(x) > 0]

    def srun_prefix(self, cmd, nodeId, nodes=1, procs=1):
        return " ".join(["srun -N", str(nodes), "-n", str(procs), "-w", str(nodeId), cmd])

    def run_cmd(self, params, nodeId, trial=0):
        text = f'python exarl/driver --run_id RUN{trial:04d} --experiment_id ' + self.exp
        dash = [("--" + x[0], str(x[1])) for x in params]
        flat_list = [item for sublist in dash for item in sublist]
        text = ' '.join([text, *flat_list])
        return self.srun_prefix(text, nodeId) 

    def parse(self, output):
        lines = output.split("\n")
        #print("BEGIN_LINES: \n", lines, "\nEND_LINES", flush=True)
        num_conv_eps = None
        for line in lines:
            if "Total Reward:" in line:
                rolling_reward = line.split(" ")[-1]
            elif "Maximum elapsed time" in line:
                time = line.split(" ")[-1]
            elif "Num eps" in line:
                #print("LINE: ", line)
                num_eps = line.split(" ")[2]
            elif "Converged:" in line:
                num_conv_eps = line.split(" ")[1]
        try:
            if num_conv_eps is not None:
                num_eps = num_conv_eps
            return (rolling_reward, time, num_eps)
        except UnboundLocalError:
            return False

    def objective(self, trial):
        my_node = trial.number % len(self.all_nodes)
        suggestions = []
        for p in self.params:
            if len(self.params[p]) == 2:
                minimum, maximum = self.params[p]
                assert type(minimum) == type(maximum), "Minimum and Maximum types should be the same"
                if type(minimum) == float:
                    suggestions.append((p, trial.suggest_float(p, minimum, maximum)))
                elif type(minimum) == int:
                    suggestions.append((p, trial.suggest_int(p, minimum, maximum)))
                else:
                    raise Exception("Could not determine hyperparameter type")
                
            else:
                suggestions.append((p, trial.suggest_categorical(p, minimum, maximum)))
        #print(suggestions)
        cmd = command(self.run_cmd(suggestions, self.all_nodes[my_node], trial=trial.number), wait=True)
        res = self.parse(cmd.out)
        try: 
            res = [float(x) for x in res]
            res = self.objective_func(res)
            self.log.append([trial.number, max(res, max([x[1] for x in self.log]))])
            return res
        except:
            print("Failed:", cmd.cmd)
            print("PARSING ERROR:\n", cmd.out)
            print("Error: \n", cmd.err, flush=True)
            self.log.append([trial.number, self.log[-1][1]])

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
        #self.ret = self.sp.wait()
        self.out, self.err = self.sp.communicate()

with open("/people/suet688/exaLearn/ExaRL/exarl/config/hyper_params.json") as file:
    js = json.load(file)
    parameters_from_json = js["parameters"]
    my_sampler = js["sampler"]
    my_objective = js["objective"]
    n_trials = js["trials"]

# op = Optimizer(parameters_from_json, objectives[my_objective], samplers[my_sampler], n_trials=n_trials, exp=my_sampler + "_" + my_objective)
# res, _ = op.optimize()
# for param in res:
#     print(param, res[param])

for sampler in samplers:
    for objective in objectives:
        op = Optimizer(parameters_from_json, objectives[objective], samplers[sampler], n_trials=n_trials, exp="_".join([sampler, objective]))
        res, _ = op.optimize()
        for param in res:
            print(sampler, objective, param[0], param[1])

# for sampler in samplers:
#     xpoints = []
#     ypoints = []
#     for _ in range(2):
#         op = Optimizer(parameters_from_json, my_objective, samplers[sampler], n_trials=100)
#         res, log = op.optimize()
#         xpoints.append(np.array([x[0] for x in log]))
#         ypoints.append(np.array([x[1] for x in log]))
#     xpoints = np.array(xpoints)
#     ypoints = np.array(ypoints)
#     xpoints = np.mean(xpoints, axis=0)
#     ypoints = np.mean(ypoints, axis=0)

#     plt.plot(xpoints, ypoints, label=sampler)
# plt.legend()
# plt.savefig("Rolling_rewards.jpeg")

