from collections import deque
import multiprocessing
import subprocess
import optuna
import json

"""
To use:
    sbatch -N [number of nodes] ./exarl/driver/hyper-tune.py
Make sure the configuration files have the correct number of steps/episodes, workflow, agent, and environment
set correctly.  The hyper_params.json file will contain the parameters to explore.  Each parameter should
have the mins and max ranges set.  The following is the format:
    {
        "parameters": {
            "param1": [0.001, 0.1],
            "param2": [5, 128]
        },
        "objective": "all",
        "sampler": "tpe",
        "trials": 1
    }
The objective and sampler fields configure which objective function to maximize and which optuna function to
use respectively.  Passing all for either will cycle through all objective/sampler.  This file should be
launched from the directory just before exarl.  Also we try to reserve a single node for optuna so if you want
to launch 10 trials in parallel, sbatch -N 11.
The results dir set in the learner_cfg will contain ALL the results for each trial marked by optimizer
and trial number.
"""

"""
This is the path to the hyper parameters to optimize!
"""
path_to_json = "/qfs/projects/ecp_exalearn/hsharma/Exrl_pars/EXARL/exarl/config/hyper_params.json"

"""
These are the objective functions we will be optimizing:
    Last Rolling Reward: This is the rolling reward on exit
    Rolling Reward per Time: This is the rolling reward normalized to the total runtime
    Rolling Reward per Episode: This is the rolling reward normalized to the total number of episodes
    Total Reward per Time: This is the total reward normalized to the total runtime
    Total Reward per Episode: This is the total reward normalized to the total number of episodes
"""
objectives = {
    "rolling_reward": lambda x: x[0],
    "total_reward": lambda x: x[1],
    "rolling_reward_per_time": lambda x: x[0] / x[1],
    "rolling_reward_per_episode": lambda x: x[0] / x[2],
    "total_reward_per_time": lambda x: x[0] / x[1],
    "total_reward_per_episode": lambda x: x[0] / x[2]
}

"""
These are the samples that are readily available from optuna:
    https://optuna.readthedocs.io/en/stable/reference/samplers/index.html
"""
samplers = {
    "tpe": optuna.samplers.TPESampler,
    "random": optuna.samplers.RandomSampler,
    "cmaes": optuna.samplers.CmaEsSampler,
    "qmc": optuna.samplers.QMCSampler,
}

class Optimizer:
    """
    This class is a utility built around optuna's study to support parallel study execution using Slurm.
    Optuna parallelizes the study using built in python threading.  We leaverage this "threading" by
    launching a srun command per thread.  These commands are then run in parallel on seperate nodes.
    The results of the commands are captured and parsed and then passed into the study.
    See the following for information on parallelizing the study:
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
    Notice this is how they expect to parallize the study...
    https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html#distributed
    Attributes
    ----------
    params : dictionary
        The parameters with their upper and lower bounds to explore
    objective_func : function
        This is the function that parses the returned values of a trial giving a score
    sampler : optuna.sampler
        The type of sampler to use in the study
    n_trials : int
        How many trials to run
    exp : string
        The name of the experiment folder
    all_nodes : list
        List of node names to send srun commands to.
    """
    def __init__(self, params, objective_func, sampler, n_trials=100, exp="EXP000"):
        self.params = params
        self.objective_func = objective_func
        self.sampler = sampler
        self.n_trials = n_trials
        self.exp = exp
        self.manager = multiprocessing.Manager()
        self.lock = self.manager.Lock()
        self.all_nodes = self.get_nodes()
        # JS: Leave the first node for optuna
        if len(self.all_nodes) > 1:
            self.all_nodes = self.all_nodes[1:]

    def optimize(self):
        """
        This function performs the study.
        Returns
        -------
        dictionary
            This is the list of params and the best achieved value
        """
        print("Study: ", self.sampler, self.n_trials)
        # JS: Notice we are maximizing!
        study = optuna.create_study(direction="maximize", sampler=self.sampler())
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=len(self.all_nodes))
        return study.best_params

    def get_nodes(self):
        """
        This function uses scontrol to get a list of the nodes in sbatch job.
        They are then return as a list.
        Returns
        -------
        list
            Names of all nodes in job
        """
        cmd = command("scontrol show hostnames", wait=True)
        return [x for x in cmd.out.split('\n') if len(x) > 0]

    def srun_prefix(self, cmd, nodeId, nodes=1, procs=1):
        """
        Adds srun and srun options to the command to run.
        TODO: Propagate nodes > 1 and procs > 1 option through the rest
        of the script.  Maybe add to the hyperparameter json options.
        Returns
        -------
        string
            srun command with options
        """
        return " ".join(["srun -N", str(nodes), "-n", str(procs), "-w", str(nodeId), cmd])

    def run_cmd(self, params, nodeId, trial=0):
        """
        Creates a srun command from the trial parameters and nodeId.
        All trials will create a directory EXPXXX/RUNXXX.  The exp dir
        is configurable at the creation of this class.  The run is given
        by the trial id.
        Parameters
        ----------
        params : list
            A list of tuples containing name, value for the parameters to run
        nodeId : string
            The name of the node to run on
        trial : int
            The trial id (given by the study)
        Returns
        -------
        string
            srun command with options
        """
        text = f'python exarl/driver --run_id RUN{trial:04d} --experiment_id ' + self.exp
        dash = [("--" + x[0], str(x[1])) for x in params]
        flat_list = [item for sublist in dash for item in sublist]
        text = ' '.join([text, *flat_list])
        return self.srun_prefix(text, nodeId)

    def parse(self, output):
        """
        This parses the output of the srun command returning the reward, time, and total episodes.
        Parameters
        ----------
        output : string
            The output of the srun command
        Returns
        -------
        tuple :
            Rolling reward, total time, total episodes
        """
        lines = output.split("\n")
        for line in lines:
            if "Total reward =" in line:
                total_reward = float(line.split(" ")[-1])
            elif "Final rolling reward =" in line:
                rolling_reward = float(line.split(" ")[-1])
            elif "Maximum elapsed time =" in line:
                time = float(line.split(" ")[-1])
            elif "Final number of episodes =" in line:
                num_eps = int(line.split(" ")[-1])
        return (rolling_reward, total_reward, time, num_eps)

    def objective(self, trial):
        """
        This is the objection function.  Here we get the suggested trial parameters and pass them along
        to create a srun command.  We run this command and parse its output.  The node used to run the
        command is based on the trial number.  We assume that each python threads is running this
        function (https://github.com/optuna/optuna/blob/1a520bd5daa9ff0af09fb060464bb157f8af891b/optuna/study/_optimize.py#L64).
        We also assume each trial takes about the same time such that each nodes starts a stride making
        the trial number good to use as a node offset. Also srun should treat job step allocations "exclusively."
        From https://slurm.schedmd.com/srun.html :
            This option applies to job and job step allocations, and has two slightly different meanings for each one...
            The exclusive allocation of CPUs applies to job steps by default, but --exact is NOT the default. In other words,
            the default behavior is this: job steps will not share CPUs, but job steps will be allocated all CPUs available
            to the job on all nodes allocated to the steps.
        The function will try to run the srun command and parse the output.  If the parsing fails,
        we catch the error and print the output and the error.  We still continue on with the study.
        Parameters
        ----------
        trial : int
            Current trial
        Returns
        -------
        float :
            The score of the run trial, or -1 on failure
        """
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
        cmd = command(self.run_cmd(suggestions, self.all_nodes[my_node], trial=trial.number), wait=True)
        try:
            res = self.objective_func(self.parse(cmd.out))
            if res != res:
                raise ValueError("Result is NaN!")
            return res
        except Exception as e:
            print(e)
            print("Failed:", cmd.cmd)
            print("Output:\n", cmd.out)
            print("Error:\n", cmd.err, flush=True)
            # JS: We return -1 since we are maximizing!!!
            return -1

class command:
    """
    This class is a wrapper around subprocesses storing its results.
    Attributes
    ----------
    cmd : string
        Command to run
    sp : subprocess
        Subprocess handle to wait on
    out : string
        stdout of the command
    err : string
        stderr of the command
    """
    def __init__(self, cmd, wait=False):
        self.cmd = cmd
        self.sp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.out = None
        self.err = None
        if wait:
            self.wait()

    def wait(self):
        """
        This will wait for the command to finish populating out and err.
        """
        # self.ret = self.sp.wait()
        self.out, self.err = self.sp.communicate()


if __name__ == "__main__":
    with open(path_to_json) as file:
        js = json.load(file)
        parameters_from_json = js["parameters"]
        my_sampler = js["sampler"]
        my_objective = js["objective"]
        n_trials = js["trials"]

    if my_sampler == "all":
        my_samplers = samplers.keys()
    else:
        my_samplers = [my_sampler]

    if my_objective == "all":
        my_objectives = objectives.keys()
    else:
        my_objectives = [my_objective]

    for sampler in my_samplers:
        for objective in my_objectives:
            op = Optimizer(parameters_from_json, objectives[objective], samplers[sampler], n_trials=n_trials, exp="_".join([sampler, objective]))
            res = op.optimize()
            for param in res:
                print(sampler, objective, param, res[param])