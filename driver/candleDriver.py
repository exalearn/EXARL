import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', 'candlelib'))
sys.path.append(lib_path2)

import keras
import candle

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

#required = ['agent', 'env', 'n_episodes', 'n_steps']
required = ['agent', 'env']

class BenchmarkDriver(candle.Benchmark):

    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions

def initialize_parameters():

    # Build agent object
    driver = BenchmarkDriver(file_path, '../combo_setup.txt', 'keras',
                            prog='CANDLE_example', desc='CANDLE example driver script')

    # Initialize parameters
    gParameters = candle.finalize_parameters(driver)
    #benchmark.logger.info('Params: {}'.format(gParameters))

    return gParameters

run_params = initialize_parameters()
