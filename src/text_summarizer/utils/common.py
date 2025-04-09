import os
from box.exceptions import BoxValueError
import yaml
from text_summarizer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox object.

    Args:
        path_to_yaml (Path): The path to the YAML file to be read.

    Returns:
        ConfigBox: A ConfigBox object containing the parsed YAML content.

    Raises:
        ValueError: If the YAML file is empty.
        Exception: For any other exceptions that occur during file reading.
    """

    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=False):
    """
    Creates directories at the given list of paths if they do not exist.

    Args:
        path_to_directories (list): A list of paths to directories to be created.
        verbose (bool): If True, logs a message for each directory created.

    Returns:
        None
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def get_size(path: Path) -> str:
    """
    Returns the size of the file at the given path in megabytes.

    Args:
        path (Path): The path to the file whose size is to be obtained.

    Returns:
        str: The size of the file in megabytes, rounded to two decimal places,
             formatted as a string prefixed with '~ '.
    """

    size_in_mb = round(os.path.getsize(path) / 1024**2, 2)
    return f"~ {size_in_mb} MB"
