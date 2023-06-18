import logging
import os

def setup_logger(module_name):
    # Create a logger with the module name
    logger = logging.getLogger(module_name)

    # Set the log level to INFO
    logger.setLevel(logging.INFO)

    # Check if log directory exists for the module, if not create it
    if not os.path.exists(f'logs/{module_name}'):
        os.makedirs(f'logs/{module_name}')

    # Create a file handler
    file_handler = logging.FileHandler(f'logs/{module_name}/{module_name}.log')

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add the formatter to the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger
