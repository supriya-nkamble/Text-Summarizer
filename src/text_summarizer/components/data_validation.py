import os
from text_summarizer.entity import DataValidationConfig
from text_summarizer.logging import logger


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        try:
            all_files = os.listdir(
                os.path.join("artifacts", "data_ingestion", "samsum_dataset")
            )
            missing = [f for f in self.config.ALL_REQUIRED_FILES if f not in all_files]
            validation_status = len(missing) == 0

            if missing:
                logger.warning("Missing required files: %s", missing)

            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            raise e
