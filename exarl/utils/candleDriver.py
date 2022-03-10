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
from pprint import pformat
from tensorflow import keras
import os
import sys
import site
file_path = os.path.dirname(os.path.realpath(__file__))
import exarl.candlelib.candle as candle


# required = ['agent', 'env', 'n_episodes', 'n_steps']
required = ['agent', 'model_type', 'env', 'workflow']

def resolve_path(*path_components) -> str:
    """ Resolve path to configuration files.
    Priority is as follows:

      0. ${CONFIG_DIR}
      1. <current working directory>/exarl/config
      2. ~/.exarl/config
      3. <site-packages dir>/exarl/config
    """
    if len(path_components) == 1:
        path = path_components[0]
    else:
        path = os.path.join(*path_components)

    if "CONFIG_DIR" in os.environ:
        config_dir = os.environ.get('CONFIG_DIR')
        config_file = os.path.join(config_dir, path)
        print(config_file)
        return config_file
    cwd_path = os.path.join(os.getcwd(), 'exarl', 'config', path)
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
        """ Functionality to set variables specific for the benchmark
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
    from exarl.utils.log import setup_logger
    logger = setup_logger(__name__, gParameters['log_level'])
    logger.info("Finalized parameters:\n" + pformat(gParameters))
    global run_params
    global kerasDefaults
    run_params = gParameters
    kerasDefaults = candle.keras_default_config()


def base_parser(params):
    """
        The base_parser is needed to intercept command line overwrites of the
        basic configuration files only. All other additional keywords are
        generated automatically by the parser_from_json function.
        The configuration files which can be set here correspond to the
        essential components of an EXARL run: agent, env (environment),
        model (model_type) and workflow.

        Parameters
        ----------
        params : dictionary object
                Dictionary of parameters

        Returns
        -------
        params : dictionary object
            Updated dictionary of parameters
    """

    # checks for env or agent command line override before reading json files
    parser = argparse.ArgumentParser(description="Base parser")
    parser.add_argument("--agent")
    parser.add_argument("--env")
    parser.add_argument("--model_type")
    parser.add_argument("--workflow")
    parser.add_argument("--data_structure")
    parser.add_argument("--batch_size")

    args, leftovers = parser.parse_known_args()

    if args.agent is not None:
        params['agent'] = args.agent
        print("Agent overwitten from command line: ", args.agent)

    if args.env is not None:
        params['env'] = args.env
        print("Environment overwitten from command line: ", args.env)

    if args.model_type is not None:
        params['model_type'] = args.model_type
        print("Model overwitten from command line: ", args.model_type)

    if args.workflow is not None:
        params['workflow'] = args.workflow
        print("Workflow overwitten from command line: ", args.workflow)

    return params


def parser_from_json(json_file):
    """
        Custom parser to read a json file and return the list of included keywords.
        Special case for True/False since these are not handled correctly by the default
        python command line parser.
        All keywords defined in json files are subsequently available to be overwritten
        from the command line, using the CANDLE command line parser.

        Parameters
        ----------
        json_file : str
            File to be parsed

        Returns
        -------
        new_defs : dictionary
            Dictionary of parameters

    """
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
    """ Build the full set of run parameters by sequentially parsing the config files
        for agent, model, env and workflow.
        Unless overwritten from the command line (via base_parser), the names for
        these config files are defined in the learner_cfg.json file.
    """

    learner_cfg = resolve_path('learner_cfg.json')
    print('Looking for ', learner_cfg)
    learner_defs = parser_from_json(learner_cfg)
    print('Learner parameters from ', learner_cfg)
    params = json.load(open(learner_cfg))
    params = base_parser(params)
    print('_________________________________________________________________')
    print("Running - {}, {}, {} and {}".format(params['agent'], params['model_type'], params['env'], params['workflow']))
    print('_________________________________________________________________', flush=True)
    try:
        agent_cfg = resolve_path('agent_cfg',
                                 params['agent'] + '.json')
        print('Agent parameters from ', agent_cfg)
    except FileNotFoundError:
        agent_cfg = resolve_path('agent_cfg', 'default_agent_cfg.json')
        print('Agent configuration does not exist, using default configuration')
    agent_defs = parser_from_json(agent_cfg)

    try:
        model_cfg = resolve_path('model_cfg',
                                 params['model_type'] + '.json')
        print('Model parameters from ', model_cfg)
    except FileNotFoundError:
        model_cfg = resolve_path('model_cfg', 'default_model_cfg.json')
        print('Model configuration does not exist, using default configuration')
    model_defs = parser_from_json(model_cfg)

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

    return learner_defs + agent_defs + model_defs + env_defs + workflow_defs
