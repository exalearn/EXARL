from __future__ import absolute_import


from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from tensorflow.keras import losses

from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.metrics import binary_crossentropy, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
import tensorflow.keras as keras

from scipy.stats.stats import pearsonr

from exarl.candlelib.default_utils import set_seed as set_seed_defaultUtils

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.metrics import r2_score

import os


def set_parallelism_threads():
    """ Set the number of parallel threads according to the number available on the hardware
    """

    if K.backend() == 'tensorflow' and 'NUM_INTRA_THREADS' in os.environ and 'NUM_INTER_THREADS' in os.environ:
        import tensorflow as tf
        # print('Using Thread Parallelism: {} NUM_INTRA_THREADS, {} NUM_INTER_THREADS'.format(os.environ['NUM_INTRA_THREADS'], os.environ['NUM_INTER_THREADS']))
        session_conf = tf.ConfigProto(inter_op_parallelism_threads=int(os.environ['NUM_INTER_THREADS']),
                                      intra_op_parallelism_threads=int(os.environ['NUM_INTRA_THREADS']))
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)


def set_seed(seed):
    """ Set the random number seed to the desired value

        Parameters
        ----------
        seed : integer
            Random number seed.
    """

    set_seed_defaultUtils(seed)

    if K.backend() == 'tensorflow':
        import tensorflow as tf
        if tf.__version__ < "2.0.0":
            tf.set_random_seed(seed)
        else:
            tf.random.set_seed(seed)


def get_function(name):
    mapping = {}

    mapped = mapping.get(name)
    if not mapped:
        raise Exception('No keras function found for "{}"'.format(name))

    return mapped


def build_optimizer(type, lr, kerasDefaults):
    """ Set the optimizer to the appropriate Keras optimizer function
        based on the input string and learning rate. Other required values
        are set to the Keras default values

        Parameters
        ----------
        type : string
            String to choose the optimizer

            Options recognized: 'sgd', 'rmsprop', 'adagrad', adadelta', 'adam'
            See the Keras documentation for a full description of the options

        lr : float
            Learning rate

        kerasDefaults : list
            List of default parameter values to ensure consistency between frameworks

        Returns
        ----------
        The appropriate Keras optimizer function
    """

    if type == 'sgd':
        return optimizers.SGD(learning_rate=lr, decay=kerasDefaults['decay_lr'],
                              momentum=kerasDefaults['momentum_sgd'],
                              nesterov=kerasDefaults['nesterov_sgd'])  # ,
        # clipnorm=kerasDefaults['clipnorm'],
        # clipvalue=kerasDefaults['clipvalue'])

    elif type == 'rmsprop':
        return optimizers.RMSprop(learning_rate=lr, rho=kerasDefaults['rho'],
                                  epsilon=kerasDefaults['epsilon'],
                                  decay=kerasDefaults['decay_lr'])  # ,
        # clipnorm=kerasDefaults['clipnorm'],
# clipvalue=kerasDefaults['clipvalue'])

    elif type == 'adagrad':
        return optimizers.Adagrad(learning_rate=lr,
                                  epsilon=kerasDefaults['epsilon'],
                                  decay=kerasDefaults['decay_lr'])  # ,
        # clipnorm=kerasDefaults['clipnorm'],
        # clipvalue=kerasDefaults['clipvalue'])

    elif type == 'adadelta':
        return optimizers.Adadelta(learning_rate=lr, rho=kerasDefaults['rho'],
                                   epsilon=kerasDefaults['epsilon'],
                                   decay=kerasDefaults['decay_lr'])  # ,
        # clipnorm=kerasDefaults['clipnorm'],
# clipvalue=kerasDefaults['clipvalue'])

    elif type == 'adam':
        return optimizers.Adam(learning_rate=lr, beta_1=kerasDefaults['beta_1'],
                               beta_2=kerasDefaults['beta_2'],
                               epsilon=kerasDefaults['epsilon'],
                               decay=kerasDefaults['decay_lr'])  # ,
        # clipnorm=kerasDefaults['clipnorm'],
        # clipvalue=kerasDefaults['clipvalue'])

# Not generally available
#    elif type == 'adamax':
#        return optimizers.Adamax(learning_rate=lr, beta_1=kerasDefaults['beta_1'],
#                               beta_2=kerasDefaults['beta_2'],
#                               epsilon=kerasDefaults['epsilon'],
#                               decay=kerasDefaults['decay_lr'])

#    elif type == 'nadam':
#        return optimizers.Nadam(learning_rate=lr, beta_1=kerasDefaults['beta_1'],
#                               beta_2=kerasDefaults['beta_2'],
#                               epsilon=kerasDefaults['epsilon'],
#                               schedule_decay=kerasDefaults['decay_schedule_lr'])


def build_initializer(type, kerasDefaults, seed=None, constant=0.):
    """ Set the initializer to the appropriate Keras initializer function
        based on the input string and learning rate. Other required values
        are set to the Keras default values

        Parameters
        ----------
        type : string
            String to choose the initializer

            Options recognized: 'constant', 'uniform', 'normal',
            'glorot_uniform', 'lecun_uniform', 'he_normal'

            See the Keras documentation for a full description of the options

        kerasDefaults : list
            List of default parameter values to ensure consistency between frameworks

        seed : integer
            Random number seed

        constant : float
            Constant value (for the constant initializer only)

        Return
        ----------
        The appropriate Keras initializer function
    """

    if type == 'constant':
        return initializers.Constant(value=constant)

    elif type == 'uniform':
        return initializers.RandomUniform(minval=kerasDefaults['minval_uniform'],
                                          maxval=kerasDefaults['maxval_uniform'],
                                          seed=seed)

    elif type == 'normal':
        return initializers.RandomNormal(mean=kerasDefaults['mean_normal'],
                                         stddev=kerasDefaults['stddev_normal'],
                                         seed=seed)

# Not generally available
#    elif type == 'glorot_normal':
#        return initializers.glorot_normal(seed=seed)

    elif type == 'glorot_uniform':
        return initializers.glorot_uniform(seed=seed)

    elif type == 'lecun_uniform':
        return initializers.lecun_uniform(seed=seed)

    elif type == 'he_normal':
        return initializers.he_normal(seed=seed)


def build_loss(loss_type, kerasDefaults, reduction='auto'):

    # set the reduction according to passed in text string
    if reduction == 'auto':
        reduction = keras.utils.ReductionV2.AUTO
    elif reduction == 'none':
        reduction = keras.losses.Reduction.NONE
    elif reduction == 'sum':
        reduction = keras.losses.Reduction.SUM
    elif reduction == 'sum_over_batch_size':
        reduction = keras.losses.Reduction.SUM_OVER_BATCH_SIZE

    # set loss function according to passed in text string
    if loss_type == 'binary_crossentropy':
        return keras.losses.BinaryCrossEntropy(reduction=reduction)
    elif loss_type == 'categorical_crossentropy':
        return keras.losses.CategoricalCrossEntropy(reduction=reduction)
    elif loss_type == 'categorical_hinge':
        return keras.losses.CategoricalHinge(reduction=reduction)
    elif loss_type == 'cosine_similarity':
        return keras.losses.CosineSimilarity(reduction=reduction)
    elif loss_type == 'hinge':
        return keras.losses.Hinge(reduction=reduction)
    elif loss_type == 'huber':
        return keras.losses.Huber(reduction=reduction)
    elif loss_type == 'kl_divergence':
        return keras.losses.KLDivergence(reduction=reduction)
    elif loss_type == 'log_cosh':
        return keras.losses.LogCosh(reduction=reduction)
    elif loss_type == 'mae':
        return keras.losses.MeanAbsoluteError(reduction=reduction)
    elif loss_type == 'mape':
        return keras.losses.MeanAbsolutePercentageError(reduction=reduction)
    elif loss_type == 'mse':
        return keras.losses.MeanSquaredError(reduction=reduction)
    elif loss_type == 'msle':
        return keras.losses.MeanSquaredLogarithmicError(reduction=reduction)
    elif loss_type == 'poisson':
        return keras.losses.Poisson(reduction=reduction)
    elif loss_type == 'sparse_categorical_crossentropy':
        return keras.losses.SparseCategoricalCrossentropy(reduction=reduction)
    elif loss_type == 'squared_hinge':
        return keras.losses.SquaredHinge(reduction=reduction)


def xent(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)


def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def covariance(x, y):
    return K.mean(x * y) - K.mean(x) * K.mean(y)


def corr(y_true, y_pred):
    cov = covariance(y_true, y_pred)
    var1 = covariance(y_true, y_true)
    var2 = covariance(y_pred, y_pred)
    return cov / (K.sqrt(var1 * var2) + K.epsilon())


def evaluate_autoencoder(y_pred, y_test):
    mse = mean_squared_error(y_pred, y_test)
    r2 = r2_score(y_test, y_pred)
    corr, _ = pearsonr(y_pred.flatten(), y_test.flatten())
    # print('Mean squared error: {}%'.format(mse))
    return {'mse': mse, 'r2_score': r2, 'correlation': corr}


class PermanentDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super(PermanentDropout, self).__init__(rate, **kwargs)
        self.uses_learning_phase = False

    def call(self, x, mask=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(x)
            x = K.dropout(x, self.rate, noise_shape)
        return x


def register_permanent_dropout():
    get_custom_objects()['PermanentDropout'] = PermanentDropout


class LoggingCallback(Callback):
    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
        msg = "[Epoch: %i] %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in sorted(logs.items())))
        self.print_fcn(msg)


class MultiGPUCheckpoint(ModelCheckpoint):

    def set_model(self, model):
        if isinstance(model.layers[-2], Model):
            self.model = model.layers[-2]
        else:
            self.model = model
