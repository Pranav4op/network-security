import logging
import os
from datetime import datetime

CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")

LOG_DIR = os.path.join("logs", CURRENT_DATE)
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = datetime.now().strftime("%H_%M_%S") + ".log"

LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)
