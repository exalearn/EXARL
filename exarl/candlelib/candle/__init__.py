from __future__ import absolute_import

# __version__ = '0.0.0'

# import from data_utils
from exarl.candlelib.data_utils import (
    load_csv_data,
    load_Xy_one_hot_data2,
    load_Xy_data_noheader,
    drop_impute_and_scale_dataframe,
    discretize_dataframe,
    discretize_array,
    lookup,
)

# import from file_utils
from exarl.candlelib.file_utils import get_file

# import from default_utils
from exarl.candlelib.default_utils import (
    ArgumentStruct,
    Benchmark,
    str2bool,
    finalize_parameters,
    fetch_file,
    verify_path,
    keras_default_config,
    set_up_logger,
)

from exarl.candlelib.generic_utils import Progbar

# import from viz_utils
from exarl.candlelib.viz_utils import (
    plot_history,
    plot_scatter,
    plot_density_observed_vs_predicted,
    plot_2d_density_sigma_vs_error,
    plot_histogram_error_per_sigma,
    plot_calibration_and_errors,
    plot_percentile_predictions,
)


# import from uq_utils
from exarl.candlelib.uq_utils import (
    compute_statistics_homoscedastic,
    compute_statistics_homoscedastic_all,
    compute_statistics_heteroscedastic,
    compute_statistics_quantile,
    split_data_for_empirical_calibration,
    compute_empirical_calibration,
    bining_for_calibration,
    computation_of_valid_calibration_interval,
    applying_calibration,
    overprediction_check,
    generate_index_distribution,
)

# profiling
from exarl.candlelib.profiling_utils import (
    start_profiling,
    stop_profiling
)

# exarl
from exarl.candlelib.exarl_utils import get_default_exarl_parser

# import benchmark-dependent utils
import sys
if 'keras' in sys.modules:
    print('Importing candle utils for keras')
    # import from keras_utils
    # from keras_utils import dense
    # from keras_utils import add_dense
    from exarl.candlelib.keras_utils import (
        build_initializer,
        build_optimizer,
        get_function,
        set_seed,
        set_parallelism_threads,
        PermanentDropout,
        register_permanent_dropout,
        LoggingCallback,
        MultiGPUCheckpoint,
        r2,
        mae,
        mse,
    )

    from exarl.candlelib.viz_utils import plot_metrics

    from exarl.candlelib.solr_keras import (
        CandleRemoteMonitor,
        compute_trainable_params,
        TerminateOnTimeOut,
    )

elif 'torch' in sys.modules:
    print('Importing candle utils for pytorch')
    from exarl.candlelib.pytorch_utils import (
        set_seed,
        build_optimizer,
        build_activation,
        get_function,
        initialize,
        xent,
        mse,
        set_parallelism_threads  # for compatibility,
    )

else:
    raise Exception('No backend has been specified.')
