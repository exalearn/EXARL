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
import logging
from pprint import pformat

class ExaGlobals:
    """
    This class houses all of the globals required for Exarl.
    This includes run_params, keras_defaults, and logger.
    It has been built with helpful exceptions/messages for
    when something has not been initialized yet.
    To initialize parameter create the ExaGlobals object. The
    class is a singleton so it can only be created once.
    Keys can be modified however by using the set_param method.

    Attributes
    ----------
    __keras_default : dictionary
        Private member that houses defaults from candle for keras
    __run_params : dictionary
        Private member for all the parameters read in from the config
        files via candleDriver
    init_logger : bool
        Indicates if the logs have been initialized
    __logger : dictionary
        Private member that holds all loggers
    __global_log_level : int
        Private member that give the log level from candleDriver
    """

    __keras_defaults = None
    __run_params = None
    __logger = {}
    __global_log_level = None

    class GlobalsNotInitialized(Exception):
        """
        Exception raised when trying to access globals that have not
        been initialized.
        """
        def __init__(self, key, value=None):
            self.key = key
            self.value = value

        def __str__(self):
            message = "ExaRL globals have not been set. Trying " + str(self.key)
            if self.value is not None:
                return message + " : " + str(self.value)
            else:
                return message

    class GlobalDoesNotExist(Exception):
        """
        Exception raised when trying to access global that has not
        been set.
        """
        def __init__(self, which, key, value=None):
            self.key = key
            self.value = value
            self.which = which

        def __str__(self):
            message = "ExaRL globals key does not exist. Trying " + str(self.key)
            if self.value is not None:
                message += " : " + str(self.value)
            message += "\n" + pformat(self.which)
            return message

    def __init__(self, run_params, keras_defaults):
        if ExaGlobals.__run_params is None and ExaGlobals.__keras_defaults is None:
            if isinstance(run_params, dict):
                ExaGlobals.__run_params = run_params
            if isinstance(keras_defaults, dict):
                ExaGlobals.__keras_defaults = keras_defaults

            log_level = ExaGlobals.lookup_params('log_level')
            ExaGlobals.__init_loggers(*log_level)
            logger = ExaGlobals.setup_logger(__name__)
            logger().info("Finalized parameters:\n" + pformat(ExaGlobals.__run_params))

    def is_init():
        """
        Returns if globals have been initialized.
        """
        return ExaGlobals.__run_params is not None and ExaGlobals.__keras_defaults is not None

    def lookup_params(key):
        """
        Returns if key is in run_params otherwise throws an exception.
        """
        if ExaGlobals.__run_params is None:
            raise ExaGlobals.GlobalsNotInitialized(key)
        if key not in ExaGlobals.__run_params:
            raise ExaGlobals.GlobalDoesNotExist(ExaGlobals.__run_params, key)
        return ExaGlobals.__run_params[key]

    def set_param(key, value):
        """
        Sets a parameter in run_params if it has been initialized.
        """
        if ExaGlobals.__run_params is None:
            raise ExaGlobals.GlobalsNotInitialized(key, value=value)
        ExaGlobals.__run_params[key] = value

    def set_params(dic):
        """
        Updates parameters with a dictionary of new params.
        """
        if ExaGlobals.__run_params is None:
            raise ExaGlobals.GlobalsNotInitialized("")
        assert isinstance(dic, dict), "Params must be a dictionary."
        ExaGlobals.__run_params.update(dic)

    def keras_default(key):
        """
        Returns if key is in keras_defaults otherwise throws an exception.
        """
        if ExaGlobals.__keras_defaults is None:
            raise ExaGlobals.GlobalsNotInitialized(key)
        if key is None:
            return ExaGlobals.__keras_defaults
        if key not in ExaGlobals.__keras_defaults:
            raise ExaGlobals.GlobalDoesNotExist(ExaGlobals.__keras_defaults, key)
        return ExaGlobals.__keras_defaults[key]

    def log_helper(name):
        """
        This function is returned when called by setup_logger
        as a way to raise an error if not initialized.
        """
        if ExaGlobals.is_init():
            return ExaGlobals.__logger[name]
        raise ExaGlobals.GlobalsNotInitialized('log_level')

    def __init_log(name, level):
        """
        Initialize a single logger.
        """
        if level is None:
            level = ExaGlobals.__global_log_level
        formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger = logging.getLogger(name)

        if level == 0:
            logger.setLevel(logging.DEBUG)
        elif level == 1:
            logger.setLevel(logging.INFO)
        elif level == 2:
            logger.setLevel(logging.WARNING)
        elif level == 3:
            logger.setLevel(logging.ERROR)

        logger.addHandler(handler)
        return logger

    def __init_loggers(tensorflow_log_level, global_log_level):
        """
        Initialize all loggers and set the tensorflow logger level.
        """
        # Set TensorFlow log level
        # 0: debug, 1: info, 2: warning, 3: error
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tensorflow_log_level)
        ExaGlobals.__global_log_level = global_log_level
        for key in ExaGlobals.__logger:
            log_level = ExaGlobals.__logger[key]
            ExaGlobals.__logger[key] = ExaGlobals.__init_log(key, log_level)

    def setup_logger(name=None, level=None):
        """
        This is function called by individual files to provide a
        promise for logging.
        """
        if name not in ExaGlobals.__logger:
            if ExaGlobals.is_init():
                ExaGlobals.__logger[name] = ExaGlobals.__init_log(name, level)
            else:
                ExaGlobals.__logger[name] = level
        return lambda: ExaGlobals.log_helper(name)
