from __future__ import absolute_import

# __version__ = '0.0.0'

# import from data_utils
from exarl.candlelib.data_utils import load_csv_data
from exarl.candlelib.data_utils import load_Xy_one_hot_data2
from exarl.candlelib.data_utils import load_Xy_data_noheader
from exarl.candlelib.data_utils import drop_impute_and_scale_dataframe
from exarl.candlelib.data_utils import discretize_dataframe
from exarl.candlelib.data_utils import discretize_array
from exarl.candlelib.data_utils import lookup

# import from file_utils
from exarl.candlelib.file_utils import get_file

# import from default_utils
from exarl.candlelib.default_utils import ArgumentStruct
from exarl.candlelib.default_utils import Benchmark
from exarl.candlelib.default_utils import str2bool
from exarl.candlelib.default_utils import finalize_parameters
from exarl.candlelib.default_utils import fetch_file
from exarl.candlelib.default_utils import verify_path
from exarl.candlelib.default_utils import keras_default_config
from exarl.candlelib.default_utils import set_up_logger

from exarl.candlelib.generic_utils import Progbar

# import from viz_utils
from exarl.candlelib.viz_utils import plot_history
from exarl.candlelib.viz_utils import plot_scatter
from exarl.candlelib.viz_utils import plot_density_observed_vs_predicted
from exarl.candlelib.viz_utils import plot_2d_density_sigma_vs_error
from exarl.candlelib.viz_utils import plot_histogram_error_per_sigma
from exarl.candlelib.viz_utils import plot_calibration_and_errors
from exarl.candlelib.viz_utils import plot_percentile_predictions


# import from uq_utils
from exarl.candlelib.uq_utils import compute_statistics_homoscedastic
from exarl.candlelib.uq_utils import compute_statistics_homoscedastic_all
from exarl.candlelib.uq_utils import compute_statistics_heteroscedastic
from exarl.candlelib.uq_utils import compute_statistics_quantile
from exarl.candlelib.uq_utils import split_data_for_empirical_calibration
from exarl.candlelib.uq_utils import compute_empirical_calibration
from exarl.candlelib.uq_utils import bining_for_calibration
from exarl.candlelib.uq_utils import computation_of_valid_calibration_interval
from exarl.candlelib.uq_utils import applying_calibration
from exarl.candlelib.uq_utils import overprediction_check
from exarl.candlelib.uq_utils import generate_index_distribution

# profiling
from exarl.candlelib.profiling_utils import start_profiling
from exarl.candlelib.profiling_utils import stop_profiling

# exarl
from exarl.candlelib.exarl_utils import get_default_exarl_parser
from exarl.candlelib.helper_utils import search

# import benchmark-dependent utils
import sys

if search(sys.modules, 'keras'):
    print('Importing candle utils for keras')

    # import from keras_utils
    # from keras_utils import dense
    # from keras_utils import add_dense
    from exarl.candlelib.keras_utils import build_initializer
    from exarl.candlelib.keras_utils import build_optimizer
    from exarl.candlelib.keras_utils import build_loss
    from exarl.candlelib.keras_utils import get_function
    from exarl.candlelib.keras_utils import set_seed
    from exarl.candlelib.keras_utils import set_parallelism_threads
    from exarl.candlelib.keras_utils import PermanentDropout
    from exarl.candlelib.keras_utils import register_permanent_dropout
    from exarl.candlelib.keras_utils import LoggingCallback
    from exarl.candlelib.keras_utils import MultiGPUCheckpoint
    from exarl.candlelib.keras_utils import r2
    from exarl.candlelib.keras_utils import mae
    from exarl.candlelib.keras_utils import mse

    from exarl.candlelib.viz_utils import plot_metrics

    from exarl.candlelib.solr_keras import CandleRemoteMonitor
    from exarl.candlelib.solr_keras import compute_trainable_params
    from exarl.candlelib.solr_keras import TerminateOnTimeOut

elif search(sys.modules, 'torch'):
    print('Importing candle utils for pytorch')
    from exarl.candlelib.pytorch_utils import set_seed
    from exarl.candlelib.pytorch_utils import build_optimizer
    from exarl.candlelib.pytorch_utils import build_activation
    from exarl.candlelib.pytorch_utils import get_function
    from exarl.candlelib.pytorch_utils import initialize
    from exarl.candlelib.pytorch_utils import xent
    from exarl.candlelib.pytorch_utils import mse
    from exarl.candlelib.pytorch_utils import set_parallelism_threads  # for compatibility

else:
    raise Exception('No backend has been specified.')
