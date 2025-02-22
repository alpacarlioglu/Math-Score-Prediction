import logging
import os, sys
from datetime import datetime
from src.exception import CustomException

"""
This script sets up a logging system that saves log messages to a dynamically 
named log file within a 'logs' directory. The log file's name is based on the 
current date and time to ensure uniqueness.

The steps followed by the script are:
1. Generates a log file name based on the current timestamp (format: MM_DD_YYYY_HH_MM_SS).
2. Constructs the full file path for the log file, placing it inside the 'logs' directory.
3. Creates the 'logs' directory if it does not already exist.
4. Configures the logging system to log messages with a severity of INFO or higher 
   (INFO, WARNING, ERROR, CRITICAL).
5. Sets the log message format to include the timestamp, line number, logger name, 
   log level, and the log message itself.

The resulting log file will store log messages in the following format:
    [ timestamp ] line_number logger_name - log_level - log_message
"""

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_dir = os.path.join(os.getcwd(), "logs")  # Path to the 'logs' directory
os.makedirs(logs_dir, exist_ok=True)  # Create 'logs' directory if it doesn't exist

LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH, # Where logs will be saved
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)