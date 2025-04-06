import os
import sys
import logging

logging_str = "[%(asctime)s]: %(levelname)s: %(module)s: %(message)s"

log_dir = "logs"
log_dir_path = os.path.join(os.getcwd(), log_dir)
os.makedirs(log_dir_path, exist_ok=True)

log_file_path = os.path.join(log_dir_path, "log_file.log")

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("TextSummarizerLogger")
