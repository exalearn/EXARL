import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', 'candlelib'))
sys.path.append(lib_path2)

import keras
import candle
from pprint import pprint

import json

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
    driver = BenchmarkDriver(file_path, '../combo_setup.small', 'keras',
                            prog='CANDLE_example', desc='CANDLE example driver script')

    # Initialize parameters
    gParameters = candle.finalize_parameters(driver)
    #benchmark.logger.info('Params: {}'.format(gParameters))

    return gParameters

def parser_from_json(json_file):
    file = open(json_file,)
    params = json.load(file)
    new_defs = []
    for key in params:
        new_def = {'name':key, 'type':(type(params[key])), 'default':params[key]}
        new_defs.append(new_def)
    #print(new_defs)
    return new_defs

def get_driver_params():
    run_cfg = open('run_params.json')
    params = json.load(run_cfg)
    run_defs = parser_from_json('run_params.json')
    print('Driver parameters')
    pprint(run_defs)
    agent_cfg = 'agents/agent_vault/agent_cfg/'+params['agent']+'.json'
    agent_defs = parser_from_json(agent_cfg)
    print('Agent parameters')
    pprint(agent_defs)
    env_cfg = 'envs/env_vault/env_cfg/'+params['env']+'.json'
    env_defs = parser_from_json(env_cfg)
    print('Environment parameters')
    pprint(env_defs)
    lrn_cfg = 'learner_cfg.json'
    lrn_defs = parser_from_json(lrn_cfg)
    print('Learner parameters')
    pprint(lrn_defs)
    return run_defs+agent_defs+env_defs+lrn_defs

