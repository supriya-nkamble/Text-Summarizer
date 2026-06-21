import os
import pytest
from unittest.mock import MagicMock, patch
from text_summarizer.components.data_validation import DataValidation
from text_summarizer.entity import DataValidationConfig


@pytest.fixture
def validation_config(tmp_path):
    status_file = str(tmp_path / "status.txt")
    return DataValidationConfig(
        root_dir=str(tmp_path),
        STATUS_FILE=status_file,
        ALL_REQUIRED_FILES=["train", "test", "validation"],
    )


def test_validation_passes_when_all_files_present(validation_config, tmp_path):
    dataset_dir = tmp_path / "artifacts" / "data_ingestion" / "samsum_dataset"
    dataset_dir.mkdir(parents=True)
    for name in ["train", "test", "validation"]:
        (dataset_dir / name).mkdir()

    validator = DataValidation(config=validation_config)
    with patch("os.listdir", return_value=["train", "test", "validation"]):
        result = validator.validate_all_files_exist()

    assert result is True
    assert "True" in open(validation_config.STATUS_FILE).read()


def test_validation_fails_when_file_missing(validation_config):
    validator = DataValidation(config=validation_config)
    with patch("os.listdir", return_value=["train", "test"]):  # missing validation
        result = validator.validate_all_files_exist()

    assert result is False
    assert "False" in open(validation_config.STATUS_FILE).read()


def test_validation_fails_when_directory_empty(validation_config):
    validator = DataValidation(config=validation_config)
    with patch("os.listdir", return_value=[]):
        result = validator.validate_all_files_exist()

    assert result is False


def test_validation_writes_status_file(validation_config):
    validator = DataValidation(config=validation_config)
    with patch("os.listdir", return_value=["train", "test", "validation"]):
        validator.validate_all_files_exist()

    assert os.path.exists(validation_config.STATUS_FILE)


def test_validation_raises_on_missing_directory(validation_config):
    validator = DataValidation(config=validation_config)
    with patch("os.listdir", side_effect=FileNotFoundError("no such directory")):
        with pytest.raises(FileNotFoundError):
            validator.validate_all_files_exist()
