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

'''
# These are just added to the command line options
additional_definitions =  [
# learner params
{'name':'n_episodes', 'type':int,'help':'Number of episodes to run'},
{'name':'n_steps', 'type':int,'help':'Number of steps to run'},
{'name':'agent', 'type':int,'help':'agent'},
{'name':'env', 'type':int,'help':'environment'},
{'name':'run_type', 'type':str, 'choices':['static','dynamic'],'help':'run_type'},
# agent params
#{'name':'search_method',  'default':'epsilon', 'help':'Search method'},
#{'name':'gamma', 'type':float, 'default': 0.95,'help':'discount rate'},
#{'name':'epsilon', 'type':float,'default': 1.0  ,'help':'exploration rate'},
#{'name':'epsilon_min', 'type':float,'default': 0.05},
#{'name':'epsilon_decay', 'type':float, 'default': 0.995},
#{'name':'learning_rate', 'type':float, 'default':  0.001},
#{'name':'batch_size', 'type':int,'default': 32},
#{'name':'tau', 'type':float,'default': 0.5}
{'name':'search_method', 'type':str, 'help':'Search method'},
{'name':'gamma', 'type':float, 'help':'discount rate'},
{'name':'epsilon', 'type':float, 'help':'exploration rate'},
{'name':'epsilon_min', 'type':float},
{'name':'epsilon_decay', 'type':float},
{'name':'learning_rate', 'type':float},
{'name':'batch_size', 'type':int},
{'name':'tau', 'type':float}
]
'''

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
    agent_cfg = 'agents/agent_vault/agent_cfg/'+params['agent']+'.json'
    if os.path.exists(agent_cfg):
        print('Agent parameters from ', agent_cfg)
    else:
        env_cfg = 'envs/env_vault/env_cfg/defaagent_env_cfg.json'
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
