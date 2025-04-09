import os
import urllib.request as request
import zipfile
from text_summarizer.logging import logger
from text_summarizer.utils.common import get_size
from text_summarizer.entity import DataIngestionConfig
from pathlib import Path


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            try:
                filename, headers = request.urlretrieve(
                    url=self.config.source_URL, filename=self.config.local_data_file
                )
                logger.info(
                    f"{filename} downloaded with the following info:\n{headers}"
                )
            except Exception as e:
                logger.error(f"Failed to download file. Error: {e}")
                raise e
        else:
            size = get_size(Path(self.config.local_data_file))
            logger.info(f"File already exists. Size: {size}")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        try:
            with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info(f"Extracted files to {unzip_path}")
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid ZIP file. Error: {e}")
            raise e
