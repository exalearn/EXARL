CANDLE Integration
******************

`CANDLE <https://github.com/ECP-CANDLE/Candle>`_ functionality is built into EXARL.
- Add/modify the learner parameters in ``ExaRL/learner_cfg.json``

E.g.:-

.. code-block:: json

    {
        "agent": "DQN-v0",
        "env": "ExaLearnCartpole-v1",
        "workflow": "async",
        "n_episodes": 1,
        "n_steps": 10,
        "output_dir": "./exa_results_dir"
    }

- Add/modify the agent parameters in ``ExaRL/agents/agent_vault/agent_cfg/<AgentName>_<model_type>.json``

E.g.:-

.. code-block:: json

    {
        "gamma": 0.75,
        "epsilon": 1.0,
        "epsilon_min" : 0.01,
        "epsilon_decay" : 0.999,
        "learning_rate" : 0.001,
        "batch_size" : 32,
        "tau" : 0.5,
        "model_type" : "MLP",
        "dense" : [64, 128],
        "activation" : "relu",
        "optimizer" : "adam",
        "loss" : "mse"
    }

Currently, DQN agent takes either MLP or LSTM as model_type.
- Add/modify the environment parameters in ``ExaRL/envs/env_vault/env_cfg/<EnvName>.json``

E.g.:-

.. code-block:: json

    {
            "worker_app": "./envs/env_vault/cpi.py"
    }

- Add/modify the workflow parameters in ``ExaRL/workflows/workflow_vault/workflow_cfg/<WorkflowName>.json``

E.g.:-

.. code-block:: json

    {
            "process_per_env": "1"
    }

- Please note the agent, environment, and workflow configuration file (json file) name must match the agent, environment, and workflow ID specified in ``ExaRL/learner_cfg.json``.

E.g.:- ``ExaRL/agents/agent_vault/agent_cfg/DQN-v0_LSTM.json``, ``ExaRL/envs/env_vault/env_cfg/ExaCartPole-v1.json``, and ``ExaRL/workflows/workflow_vault/workflow_cfg/async.json``
