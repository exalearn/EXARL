import argparse
import json
import utils.log as log
from pprint import pformat
import keras
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', 'candlelib'))
sys.path.append(lib_path2)
import candle
# from pprint import pprint


# required = ['agent', 'env', 'n_episodes', 'n_steps']
required = ['agent', 'env']
<<<<<<< HEAD


=======
dir = '../../'
>>>>>>> unit tests for dqn.py
class BenchmarkDriver(candle.Benchmark):

    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        print('Additional definitions built from json files')
        additional_definitions = get_driver_params()
        # pprint(additional_definitions, flush=True)
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


def initialize_parameters():
    # Build agent object

    driver = BenchmarkDriver(file_path, '', 'keras',
                             prog='CANDLE_example', desc='CANDLE example driver script')

    # Initialize parameters
    gParameters = candle.finalize_parameters(driver)
    # benchmark.logger.info('Params: {}'.format(gParameters))
    logger = log.setup_logger(__name__, gParameters['log_level'])
    logger.info("Finalized parameters:\n" + pformat(gParameters))
    global run_params
    run_params = gParameters


def base_parser(params):
    # checks for env or agent command line override before reasing json files
    parser = argparse.ArgumentParser(description="Base parser")
    parser.add_argument("--agent")
    parser.add_argument("--env")
    parser.add_argument("--workflow")
    args, leftovers = parser.parse_known_args()

    if args.agent is not None:
        params['agent'] = args.agent
        print("Agent overwitten from command line: ", args.agent)

    if args.env is not None:
        params['env'] = args.env
        print("Environment overwitten from command line: ", args.env)

    if args.workflow is not None:
        params['workflow'] = args.workflow
        print("Workflow overwitten from command line: ", args.workflow)

    return params


def parser_from_json(json_file):
    file = open(json_file,)
    params = json.load(file)
    new_defs = []
    for key in params:
        if params[key] == "True" or params[key] == "False":
            new_def = {'name': key, 'type': (type(candle.str2bool(params[key]))), 'default': candle.str2bool(params[key])}
        else:
            new_def = {'name': key, 'type': (type(params[key])), 'default': params[key]}
        new_defs.append(new_def)

    return new_defs


def get_driver_params():
    learner_cfg = 'learner_cfg.json'
    learner_defs = parser_from_json(learner_cfg)
    print('Learner parameters from ', learner_cfg)
    params = json.load(open(learner_cfg))
<<<<<<< HEAD
    params = base_parser(params)
    agent_cfg = 'agents/agent_vault/agent_cfg/' + params['agent'] + '_' + params['model_type'] + '.json'
=======

    agent_cfg = dir+'agents/agent_vault/agent_cfg/'+params['agent']+'_'+params['model_type']+'.json'
    print(agent_cfg)
>>>>>>> unit tests for dqn.py
    if os.path.exists(agent_cfg):
        print('Agent parameters from ', agent_cfg)
    else:
        agent_cfg = dir+'agents/agent_vault/agent_cfg/default_agent_cfg.json'
        print('Agent configuration does not exist, using default configuration')
    agent_defs = parser_from_json(agent_cfg)

<<<<<<< HEAD
    env_cfg = 'envs/env_vault/env_cfg/' + params['env'] + '.json'
=======
    env_cfg = dir+'envs/env_vault/env_cfg/'+params['env']+'.json'
    print(env_cfg)
>>>>>>> unit tests for dqn.py
    if os.path.exists(env_cfg):
        print('Environment parameters from ', env_cfg)
    else:
        env_cfg = dir+'envs/env_vault/env_cfg/default_env_cfg.json'
        print('Environment configuration does not exist, using default configuration')
    env_defs = parser_from_json(env_cfg)

<<<<<<< HEAD
    workflow_cfg = 'workflows/workflow_vault/workflow_cfg/' + params['workflow'] + '.json'
=======
    workflow_cfg = dir+'workflows/workflow_vault/workflow_cfg/'+params['workflow']+'.json'
>>>>>>> unit tests for dqn.py
    if os.path.exists(workflow_cfg):
        print('Workflow parameters from ', workflow_cfg)
    else:
        workflow_cfg = dir+'workflows/workflow_vault/workflow_cfg/default_workflow_cfg.json'
        print('Workflow configuration does not exist, using default configuration')
    workflow_defs = parser_from_json(workflow_cfg)

<<<<<<< HEAD
    return learner_defs + agent_defs + env_defs + workflow_defs
=======
    return learner_defs+agent_defs+env_defs+workflow_defs
>>>>>>> unit tests for dqn.py
