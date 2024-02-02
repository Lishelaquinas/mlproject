import logging
import os
from datetime import datetime

# Naming convention of the log file.
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# This will creat a path getting the current directory and creates a subdirectory named logs and then appends LOG_FILE.
logs_path = os.path.join(os.getcwd(),"logs", LOG_FILE)
#creates the directory,The exist_ok=True parameter ensures that if the directory already exists, it won't raise an error.
os.makedirs(logs_path,exist_ok=True)


LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

#overridding the logging class
logging.basicConfig(
    filename =  LOG_FILE_PATH,
    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO
)
