from text_summarizer.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from text_summarizer.pipeline.stage_02_data_validation import (
    DataValidationTrainingPipeline,
)
from text_summarizer.pipeline.stage_03_data_transformation import (
    DataTransformationTrainingPipeline,
)
from text_summarizer.pipeline.stage_04_mode_trainer import (
    ModelTrainingPipeline,
)
from text_summarizer.pipeline.stage_05_model_evaluation import (
    ModelEvaluationPipeline,
)
from text_summarizer.logging import logger
import torch


STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion_training_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_training_pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_validation_training_pipeline = DataValidationTrainingPipeline()
    data_validation_training_pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Transformation stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_transformation_training_pipeline = DataTransformationTrainingPipeline()
    data_transformation_training_pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Trainer stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_training_pipeline = ModelTrainingPipeline()
    model_training_pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Evaluation stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_evaluation_pipeline = ModelEvaluationPipeline()
    model_evaluation_pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
