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
