# This material was prepared as an account of work sponsored by an agency of the
# United States Government.  Neither the United States Government nor the United
# States Department of Energy, nor Battelle, nor any of their employees, nor any
# jurisdiction or organization that has cooperated in the development of these
# materials, makes any warranty, express or implied, or assumes any legal
# liability or responsibility for the accuracy, completeness, or usefulness or
# any information, apparatus, product, software, or process disclosed, or
# represents that its use would not infringe privately owned rights. Reference
# herein to any specific commercial product, process, or service by trade name,
# trademark, manufacturer, or otherwise does not necessarily constitute or imply
# its endorsement, recommendation, or favoring by the United States Government
# or any agency thereof, or Battelle Memorial Institute. The views and opinions
# of authors expressed herein do not necessarily state or reflect those of the
# United States Government or any agency thereof.
#                 PACIFIC NORTHWEST NATIONAL LABORATORY
#                            operated by
#                             BATTELLE
#                             for the
#                   UNITED STATES DEPARTMENT OF ENERGY
#                    under Contract DE-AC05-76RL01830
import argparse
import json
import utils.log as log
from pprint import pformat
import keras
import os
import sys
import site
file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', 'candlelib'))
sys.path.append(lib_path2)
import candle
# from pprint import pprint


# required = ['agent', 'env', 'n_episodes', 'n_steps']
required = ['agent', 'env']

def resolve_path(*path_components) -> str:
    '''Resolve path to configuration files.
    Priority is as follows:

      1. <current working directory>/config
      2. ~/.exarl/config
      3. <site-packages dir>/exarl/config
    '''
    if len(path_components) == 1:
        path = path_components[0]
    else:
        path = os.path.join(*path_components)

    cwd_path = os.path.join(os.getcwd(), 'config', path)
    if os.path.exists(cwd_path):
        return cwd_path
    home_path = os.path.join(os.path.expanduser('~'), '.exarl', 'config', path)
    if os.path.exists(home_path):
        return home_path
    for site_dir in site.getsitepackages():
        install_path = os.path.join(site_dir, 'exarl', 'config', path)
        if os.path.exists(install_path):
            return install_path
    raise FileNotFoundError("Could not find file {0}!".format(path))

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
    learner_cfg = resolve_path('learner_cfg.json')
    learner_defs = parser_from_json(learner_cfg)
    print('Learner parameters from ', learner_cfg)
    params = json.load(open(learner_cfg))
    params = base_parser(params)
    try:
        agent_cfg = resolve_path('agent_cfg',
                                 params['agent'] + '_' + params['model_type'] + '.json')
        print('Agent parameters from ', agent_cfg)
    except FileNotFoundError:
        agent_cfg = resolve_path('agent_cfg', 'default_agent_cfg.json')
        print('Agent configuration does not exist, using default configuration')
    agent_defs = parser_from_json(agent_cfg)

    try:
        env_cfg = resolve_path('env_cfg', params['env'] + '.json')
        print('Environment parameters from ', env_cfg)
    except FileNotFoundError:
        env_cfg = resolve_path('env_cfg', 'default_env_cfg.json')
        print('Environment configuration does not exist, using default configuration')
    env_defs = parser_from_json(env_cfg)

    try:
        workflow_cfg = resolve_path('workflow_cfg', params['workflow'] + '.json')
        print('Workflow parameters from ', workflow_cfg)
    except FileNotFoundError:
        workflow_cfg = resolve_path('workflow_cfg', 'default_workflow_cfg.json')
        print('Workflow configuration does not exist, using default configuration')
    workflow_defs = parser_from_json(workflow_cfg)

    return learner_defs + agent_defs + env_defs + workflow_defs
