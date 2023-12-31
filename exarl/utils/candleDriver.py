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
import os
import sys
import site
import json
import argparse
from exarl.utils.globals import ExaGlobals
import exarl.candlelib.candle as candle

file_path = os.path.dirname(os.path.realpath(__file__))
required = ['agent', 'env', 'workflow', 'model_type']

def resolve_path(*path_components, config_path=None, alternate_path=None) -> str:
    """
    Resolve path to configuration files.  The alternate path is a second choice
    based on the externally loaded modules.  We look to see if any config files
    exist in these dirs.  For env and agent files they must still follow the
    dir structure (i.e. agent_cfg/DQN-v0.json or env_cfg/ExaCartPoleStatic-v0).
    Priority is as follows:
      1. Path passed in at command line using --config_file
      2. Alternate path given by --load_agent/env_path
      3. <current working directory>/exarl/config
      4. ~/.exarl/config
      5. <site-packages dir>/exarl/config
    """
    if len(path_components) == 1:
        path = path_components[0]
    else:
        path = os.path.join(*path_components)
    if config_path is not None:
        config_path = os.path.abspath(config_path)
        config_path = os.path.join(config_path, path)
        if os.path.exists(config_path):
            return config_path
    if alternate_path is not None:
        alternate_path = os.path.abspath(alternate_path)
        alternate_path = os.path.join(alternate_path, path)
        if os.path.exists(alternate_path):
            return alternate_path
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

def initialize_parameters(params=None):
    if params is None:
        # Build agent object
        class BenchmarkDriver(candle.Benchmark):
            def set_locals(self):
                """ Functionality to set variables specific for the benchmark
                - required: set of required parameters for the benchmark.
                - additional_definitions: list of dictionaries describing the additional parameters for the
                benchmark.
                """

                print('Additional definitions built from json files')
                additional_definitions = get_driver_params()
                if required is not None:
                    self.required = set(required)
                if additional_definitions is not None:
                    self.additional_definitions = additional_definitions

        driver = BenchmarkDriver(file_path, '', 'keras',
                                 prog='CANDLE_example',
                                 desc='CANDLE example driver script')
        params = candle.finalize_parameters(driver)
    ExaGlobals(params, candle.keras_default_config())

def config_parser():
    """
    This parsers runs first to get a config_path if present.
    It removes the argument by resetting the sys.argv with the remaining args.

    Returns
    -------
    String :
        The path to search for config file
    """
    parser = argparse.ArgumentParser(description="Config parser")
    parser.add_argument("--config_path")
    args, leftovers = parser.parse_known_args()
    if args.config_path is not None and not os.path.exists(args.config_path):
        raise FileNotFoundError("Path {0} does not exists!".format(args.config_path))
    sys.argv = sys.argv[:1] + leftovers
    return args.config_path

def external_env_and_agents_parser():
    """
    This checks command line for external agents and envs.  If the load_*_path is
    set for either, they will by added to the system path.  The load_agent and
    load_env will be added to the candle params.

    Returns
    -------
    List :
        List of load agent/env params
    String :
        agent load path
    String :
        env load path
    """
    parser = argparse.ArgumentParser(description="External source parser")
    parser.add_argument("--load_agent_module")
    parser.add_argument("--load_agent_path")
    parser.add_argument("--load_env_module")
    parser.add_argument("--load_env_path")
    args, leftovers = parser.parse_known_args()

    if args.load_agent_path is not None:
        args.load_agent_path = os.path.abspath(args.load_agent_path)
        if not os.path.exists(args.load_agent_path):
            raise FileNotFoundError("Load Agent Path {0} does not exists!".format(args.load_agent_path))
        if args.load_agent_path not in sys.path:
            sys.path.append(args.load_agent_path)

    if args.load_env_path is not None:
        args.load_env_path = os.path.abspath(args.load_env_path)
        if not os.path.exists(args.load_env_path):
            raise FileNotFoundError("Load Env Path {0} does not exists!".format(args.load_env_path))
        if args.load_env_path not in sys.path:
            sys.path.append(args.load_env_path)

    ret = []
    if args.load_agent_module is not None:
        ret.append({'name': 'load_agent_module', 'type': str, 'default': args.load_agent_module})

    if args.load_env_module is not None:
        ret.append({'name': 'load_env_module', 'type': str, 'default': args.load_env_module})

    sys.argv = sys.argv[:1] + leftovers
    return ret, args.load_agent_module, args.load_env_module

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
    parser.add_argument("--workflow")
    parser.add_argument("--model_type")

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

    if args.model_type is not None:
        params['model_type'] = args.model_type
        print("Model overwitten from command line: ", args.model_type)
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

def check_keyword_and_config(params, keyword, config_path, alternate_path=None):
    """
        This function performs a check for specific keywords to see if
        they are set in the config file and also checks if there is a
        corresponding configuration file.
    """
    if keyword in params.keys():
        if keyword == 'model_type':
            cfg = 'model_cfg'
        else:
            cfg = keyword + "_cfg"
        try:
            cfg_file = resolve_path(cfg, params[keyword] + '.json', config_path=config_path, alternate_path=alternate_path)
            print('Agent parameters from ', cfg)
        except FileNotFoundError:
            cfg_file = resolve_path(cfg, 'default_' + cfg + '.json', config_path=config_path)
            print(keyword + ' configuration does not exist, using default configuration')
        return parser_from_json(cfg_file)
    else:
        sys.exit("CANDLELIB::ERROR The learner config file is malformed. There is no " + keyword + " selected.")

def get_driver_params():
    """
        Build the full set of run parameters by sequentially parsing the config files
        for agent, model, env and workflow.
        Unless overwritten from the command line (via base_parser), the names for
        these config files are defined in the learner_cfg.json file.
    """
    config_path = config_parser()
    external_defs, load_agent_path, load_env_path = external_env_and_agents_parser()

    learner_cfg = resolve_path('learner_cfg.json', config_path=config_path)
    learner_defs = parser_from_json(learner_cfg)
    print('Learner parameters from ', learner_cfg)
    params = json.load(open(learner_cfg))
    params = base_parser(params)

    agent_defs = check_keyword_and_config(params, "agent", config_path, alternate_path=load_agent_path)
    env_defs = check_keyword_and_config(params, "env", config_path, alternate_path=load_env_path)
    workflow_defs = check_keyword_and_config(params, "workflow", config_path)
    model_defs = check_keyword_and_config(params, "model_type", config_path)

    print('_________________________________________________________________')
    print("Running - {}, {}, {}, and {}".format(params['agent'], params['env'], params['workflow'], params['model_type']))
    # print("Running - {}, {}, {} and {}".format(params['agent'], params['model_type'], params['env'], params['workflow']))
    print('_________________________________________________________________', flush=True)

    return learner_defs + agent_defs + env_defs + workflow_defs + model_defs + external_defs
