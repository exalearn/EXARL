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
from abc import ABC, abstractmethod

import tensorflow as tf
from exarl.utils.globals import ExaGlobals
from exarl.base.comm_base import ExaComm

class Tensorflow_Model(ABC):
    """
    This class is a base class for tensorflow models like mlp and lstm.  The purpose is to
    abstract the tensorflow specifics to reduce boiler plate for new models.  The second
    reason is to provide a factory method for building models.  This reduces the total code
    to change out the model for a given agent.

    Attributes
    ----------
    _builders : dictionary
        This dictionary has the name and constructor of inherited models.

    use_gpu : bool
        Flag that indicates model should use gpu

    enable_xla : bool
        Flag indicating is xla should be used when compiling model.  This should
        come from the configuration file for the TF model.

    mixed_precision : bool
        Flag to turn on or off mixed preceision trading off speed and memory for accuracy.
        This comes from configuration file for the TF model.

    rank : int
        Rank used for getting gpu

    observation_space : gym space
        From the environment, required input to the model

    action_space : gym space
        From the environment, required input to the model

    _device : string
        Id of the device the model will use

    _model : tensorflow model
        This is the raw tf model
    """

    _builders = {}

    def __init__(self, observation_space, action_space, use_gpu=True):
        self.use_gpu = use_gpu
        self.rank = ExaComm.global_comm.rank
        self.observation_space = observation_space
        self.action_space = action_space

        # Set the device to use
        self.dev_affinity = ExaGlobals.lookup_params('dev_affinity')
        self._set_device()

        # Optimization using XLA (1.1x speedup)
        self.enable_xla = True if ExaGlobals.lookup_params('xla') in ["true", "True", 1] else False
        
        # Optimization using mixed precision (1.5x speedup)
        self.mixed_precision = True if ExaGlobals.lookup_params('mixed_precision') in ["true", "True", 1] else False
        # https://www.tensorflow.org/guide/mixed_precision
        if self.mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        self._device = None
        self._model = None

    def register(key, builder):
        """
        This function registers a model to the tf model generator.
        Models are registered in the exarl/agents/models/__init__
        when the python file is imported.  The builder must pass
        the observation and action spaces as the first two arguments
        respectively.

        Parameters
        ----------
            key : string
                Name of the tf model
            
            builder : constructor
                This is constructor of the tf model to build 
        """
        Tensorflow_Model._builders[key] = builder

    def create(key=None, **kwargs):
        """
        This function is used to create registerd models.  This should be
        used in an agent.  The observation and action spaces should be
        passed in as keyword arguments.

        Parameters
        ----------
            key : string
                Name of the registered model to build

            kwargs : list
                The parameters to pass to the registered builder
        """
        # JS: Lookup which model if not passed
        if key is None:
            key = ExaGlobals.lookup_params('model_type')
        # JS: Ensure buffer is listed
        builder = Tensorflow_Model._builders.get(key)
        if not builder:
            raise ValueError(key)
        # JS: Must pass observation and action spaces
        observation_space = kwargs.pop("observation_space", None)
        action_space = kwargs.pop("action_space", None)
        return builder(observation_space, action_space, **kwargs)

    @abstractmethod
    def _build(self):
        """
        This function is a placeholder for the code to build the tf model.
        """
        pass

    def _compile(self):
        """
        This internal function compiles a tf model.
        """
        with tf.device(self._device):
            self._build()
            self._model.compile(loss=self.loss, optimizer=self.optimizer, jit_compile=self.enable_xla)

    def _get_device(self):
        """
        Get device type (CPU/GPU).
        There is a weird bug in TF:
        https://github.com/tensorflow/tensorflow/issues/39857
        If fixed we could use tf.config.list_physical_devices().

        Returns:
            string: device type
        """
        gpus = []
        if self.use_gpu:
            gpus = tf.config.list_logical_devices('GPU')
            # gpus = tf.config.list_physical_devices('GPU')
        cpus = tf.config.list_logical_devices('CPU')
        # cpus = tf.config.list_physical_devices('CPU')
        assert len(gpus) + len(cpus) > 0, "There are no devices listed for TF -- gpus: {} cpus: {}".format(len(gpus), len(cpus))
        
        if self.dev_affinity in ["share", "Share", "none", "None", 0, False, "False", "false"]:
            # JS: "share" mode will round robin all agents across the gpus
            if len(gpus) > 0:
                return gpus[ExaComm.agent_comm.rank % len(gpus)], 'GPU'
        else:
            # JS: Otherwise, we dedicate gpus for the learners and make the rest share
            if ExaComm.is_learner():
                if len(gpus) > 0:
                    return gpus[ExaComm.learner_comm.rank % len(gpus)], 'GPU'
            elif ExaComm.is_actor():
                learners_on_node = ExaComm.affinity.learners_on_node()
                if len(gpus) > learners_on_node:
                    remGpus = gpus[learners_on_node:]
                    return remGpus[ExaComm.agent_comm.rank % len(remGpus)], 'GPU'
                else:
                    return cpus[ExaComm.agent_comm.rank % len(cpus)], 'CPU'
        
        # JS: Fall back on cpu
        return cpus[ExaComm.agent_comm.rank % len(cpus)], 'CPU'
    
    def _set_device(self):
        self._device, dev_type = self._get_device()
        print("Setting agent rank:", ExaComm.agent_comm.rank, self._device, dev_type)
        # JS: https://www.tensorflow.org/guide/gpu
        # This limits the proc to one GPU
        # There may be cases we want to override
        # i.e. mirror strategy...
        # tf.config.set_visible_devices(self._device, dev_type)
        tf.config.set_visible_devices(self._device.name, dev_type)
        # JS: This minimizes the memory footprint
        tf.config.experimental.set_memory_growth(self._device, True)
    
    def init_model(self):
        if self._model is None:
            self._compile()

    @property
    def model(self):
        self.init_model() 
        return self._model
    
    @property
    def trainable_variables(self):
        return self._model.trainable_variables

    @property
    def variables(self):
        return self._model.variables

    def __call__(self, input, **kwargs):
        model = self._model
        with tf.device(self._device):
            ret = model(input, kwargs)
        return ret

    def get_weights(self):
        return self._model.get_weights()
    
    def set_weights(self, weights):
        with tf.device(self._device):
            self._model.set_weights(weights)

    def print(self):
        self._model.summary()
        print('', flush=True)
    