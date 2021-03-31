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


def setup_logger(name, level):
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)

    # Set TensorFlow log level
    # 0: debug, 1: info, 2: warning, 3: error
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(level[0])
    if level[1] == 0:
        # Set Python logging level to debug
        logger.setLevel(logging.DEBUG)
    elif level[1] == 1:
        # Set Python logging level to info
        logger.setLevel(logging.INFO)
    elif level[1] == 2:
        # Set Python logging level to warning
        logger.setLevel(logging.WARNING)
    elif level[1] == 3:
        # Set Python logging level to error
        logger.setLevel(logging.ERROR)

    logger.addHandler(handler)
    return logger
