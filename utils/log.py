import os
import logging

def setup_logger(name, level):
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)

    if level == 0:
        # Set TF logging level to debug
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        # Set Python logging level to debug
        logger.setLevel(logging.DEBUG)   
    elif level == 1:
        # Set TF logging level to info
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        # Set Python logging level to info
        logger.setLevel(logging.INFO)
    elif level == 2:
        # Set TF logging level to warning
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # Set Python logging level to warning
        logger.setLevel(logging.WARNING)
    elif level == 3:
        # Set TF logging level to error
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        # Set Python logging level to error
        logger.setLevel(logging.ERROR)

    logger.addHandler(handler)
    return logger