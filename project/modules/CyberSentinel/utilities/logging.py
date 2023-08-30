import logging
from datetime import datetime

# Format the current datetime
current_time = datetime.now().strftime("%d%m%Y_%H%M%S")

# Concatenate the datetime with the log filename
log_filename = f'preprocessing{current_time}.log'

logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
