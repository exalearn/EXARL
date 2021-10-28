## Configuration Files

Configuration files such as `exarl/config/learner_cfg.json` are searched for in the
following directories:

1. (current working directory)/exarl/config
2. ~/.exarl/config
3. (site-packages dir)/exarl/config

If you would like to run EXARL from outside the source directory, you may
install the config files with exarl or copy them into EXARL's config directory
in your home directory like so:

```console
$ mkdir -p ~/.exarl/config
$ cd EXARL
$ cp config/* ~/.exarl/config
```

## [CANDLE](https://github.com/ECP-CANDLE/Candle) functionality is built into EXARL

- Add/modify the learner parameters in `EXARL/exarl/config/learner_cfg.json`\
  E.g.:-

```
{
    "agent": "DQN-v0",
    "env": "ExaLearnCartpole-v1",
    "workflow": "async",
    "n_episodes": 1,
    "n_steps": 10,
    "output_dir": "./exa_results_dir"
}
```

- The description of each parameter:

```
agent: the algorithm name of the agent
env: the RL environment name
workflow: the workflow of EXARL
n_episodes: the number of episodes in the RL training
n_steps: the number of time steps in the RL training
output_dir: the output directory
```

- Add/modify the agent parameters in `EXARL/exarl/config/agent_cfg/<AgentName>.json`\
  E.g.:-

```
{
    "gamma": 0.75,
    "epsilon": 1.0,
    "epsilon_min" : 0.01,
    "epsilon_decay" : 0.999,
    "learning_rate" : 0.001,
    "batch_size" : 5,
    "tau" : 0.5
}
```

- The description of each parameter:

```
gamma: the discount rate
epsilon: the RL environment name
epsilon_decay: the epsilon-greedy (exploration vs. exploitation tradeoff) parameter
learning_rate: the learning rate
batch_size: the batch size
tau: the weight of the current model for the target weight update ( tau * model_weights[i] + ( 1 - tau ) * target_weights[i] )
```

Currently, DQN agent takes either MLP or LSTM as model_type.

- Add/modify the model parameters in `EXARL/exarl/config/model_cfg/<ModelName>.json`\
  E.g.:-

```
{
    "dense" : [64, 128],
    "activation" : "relu",
    "optimizer" : "adam",
    "out_activation" : "linear",
    "loss" : "mse"
}
```

- The description of each parameter:

```
dense: the architecture of the hidden layers
activation: the activation function for the hidden layers
optimizer: the optimizer for the model
out_activation: the activation function for the output layer
loss: the loss function
```

- Add/modify the environment parameters in `EXARL/exarl/config/env_cfg/<EnvName>.json`\
  E.g.:-

```
{
        "worker_app": "./exarl/envs/env_vault/cpi.py"
}
```

- The description of each parameter:

```
worker_app: the worker application
```

- Add/modify the workflow parameters in `EXARL/exarl/config/workflow_cfg/<WorkflowName>.json`\
  E.g.:-

```
{
        "process_per_env": "1"
}
```

- The description of each parameter:

```
process_per_env: the number of processes per environment
```

- Please note the agent, model, environment, and workflow configuration file (json file) name must match the agent, model, environment, and workflow ID specified in `EXARL/exarl/config/learner_cfg.json`. \
  E.g.:- `EXARL/exarl/config/agent_cfg/DQN-v0.json`, `EXARL/exarl/config/model_cfg/MLP.json`, `EXARL/exarl/config/env_cfg/ExaCartPole-v1.json`, and `EXARL/exarl/config/workflow_cfg/async.json`

## Running EXARL using MPI

- Existing environment can be paired with an available agent
- The following script is provided for convenience: `EXARL/exarl/driver/__main__.py`

```
from mpi4py import MPI
import utils.analyze_reward as ar
import time
import exarl as erl
import mpi4py.rc
mpi4py.rc.threads = False
mpi4py.rc.recv_mprobe = False

# MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Get run parameters using CANDLE
# run_params = initialize_parameters()

# Create learner object and run
exa_learner = erl.ExaLearner(comm)

# Run the learner, measure time
start = time.time()
exa_learner.run()
elapse = time.time() - start

# Compute and print average time
max_elapse = comm.reduce(elapse, op=MPI.MAX, root=0)
elapse = comm.reduce(elapse, op=MPI.SUM, root=0)

if rank == 0:
    print("Average elapsed time = ", elapse / size)
    print("Maximum elapsed time = ", max_elapse)
    # Save rewards vs. episodes plot
    ar.save_reward_plot()
```

- Write your own script or modify the above as needed
- Run the following command:

```
mpiexec -np <num_parent_processes> python exarl/driver/__main__.py --<run_params>=<param_value>
```

- If running a multi-process environment or agent, the communicators are available in `exarl/mpi_settings.py`.
  E.g.:-

```
import exarl.mpi_settings as mpi_settings
self.env_comm = mpi_settings.env_comm
self.agent_comm = mpi_settings.agent_comm
```

### Using parameters set in CANDLE configuration/get parameters from terminal

- To obtain the parameters from JSON file/set in terminal using CANDLE, use the following lines:

```
import exarl.utils.candleDriver as cd
cd.run_params # dictionary containing all parameters
```

- Individual parameters are accessed using the corresponding key \
  E.g.-

```
self.search_method =  cd.run_params['search_method']
self.gamma =  cd.run_params['gamma']
```
