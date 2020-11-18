import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', 'candlelib'))
sys.path.append(lib_path2)

import keras
import candle
from pprint import pprint

import json
import argparse


#required = ['agent', 'env', 'n_episodes', 'n_steps']
required = ['agent', 'env']

class BenchmarkDriver(candle.Benchmark):

    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        print('Additional definitions built from json files')
        additional_definitions = get_driver_params()
        #pprint(additional_definitions, flush=True)
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
    #benchmark.logger.info('Params: {}'.format(gParameters))

    return gParameters

def base_parser(params):
    # checks for env or agent command line override before reasing json files
    parser = argparse.ArgumentParser(description = "Base parser")
    parser.add_argument("--agent")
    parser.add_argument("--env")
    args, leftovers = parser.parse_known_args()

    if args.agent is not None:
        params['agent'] = args.agent
        print("Agent overwitten from command line: ", args.agent)

    if args.env is not None:
        params['env'] = args.env
        print("Environment overwitten from command line: ", args.env)

    return params

def parser_from_json(json_file):
    file = open(json_file,)
    params = json.load(file)
    new_defs = []
    for key in params:
        if params[key] == "True" or params[key] == "False":
            new_def = {'name':key, 'type':(type(candle.str2bool(params[key]))), 'default':candle.str2bool(params[key])}
        else:
            new_def = {'name':key, 'type':(type(params[key])), 'default':params[key]}
        new_defs.append(new_def)
    #print(new_defs)
    return new_defs

def get_driver_params():
    lrn_cfg = 'learner_cfg.json'
    lrn_defs = parser_from_json(lrn_cfg)
    print('Learner parameters from ', lrn_cfg)
    pprint(lrn_defs)
    params = json.load(open(lrn_cfg))
    params = base_parser(params)
    agent_cfg = 'agents/agent_vault/agent_cfg/'+params['agent']+'_'+params['model_type']+'.json'
    if os.path.exists(agent_cfg):
        print('Agent parameters from ', agent_cfg)
    else:
        agent_cfg = 'agents/agent_vault/agent_cfg/default_agent_cfg.json'
        print('Agent configuration does not exist, using default configuration')
    agent_defs = parser_from_json(agent_cfg)
    pprint(agent_defs)

    env_cfg = 'envs/env_vault/env_cfg/'+params['env']+'.json'
    if os.path.exists(env_cfg):
        print('Environment parameters from ', env_cfg)
    else:
        env_cfg = 'envs/env_vault/env_cfg/default_env_cfg.json'
        print('Environment configuration does not exist, using default configuration')
    env_defs = parser_from_json(env_cfg)
    pprint(env_defs)

    return lrn_defs+agent_defs+env_defs