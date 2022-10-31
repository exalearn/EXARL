from exarl.utils import candleDriver
try:
    candleDriver.initialize_parameters()
except FileNotFoundError as e:
    print(e, flush=True)
    print("Could not load candle parameters.", flush=True)

from exarl.base import ExaComm
from exarl.base import ExaAgent
from exarl.base import ExaEnv
from exarl.base import ExaWorkflow
from exarl.base import ExaLearner
from exarl.base import ExaData

from exarl.utils.globals import ExaGlobals
import importlib
import sys
try:
    load_agent_module = ExaGlobals.lookup_params('load_agent_module')
    print(sys.path)
    importlib.import_module(load_agent_module)
    print("Loaded agents:", load_agent_module)
except ExaGlobals.GlobalDoesNotExist:
    pass

try:
    load_env_module = ExaGlobals.lookup_params('load_env_module')
    importlib.import_module(load_env_module)
    print("Loaded envs:", load_env_module)
except ExaGlobals.GlobalDoesNotExist:
    pass