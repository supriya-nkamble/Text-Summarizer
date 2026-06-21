import pytest
from unittest.mock import MagicMock, patch


SAMPLE_DIALOGUE = (
    "Amanda: I baked cookies. Do you want some? "
    "Jerry: Sure! Amanda: I'll bring you tomorrow :-) "
    "Jerry: I'm so happy!!! Amanda: Me too."
)

SAMPLE_SUMMARY = "Amanda baked cookies and will bring some to Jerry tomorrow."


@pytest.fixture(autouse=True)
def reset_pipeline_cache():
    """Reset the module-level pipeline cache between tests."""
    import text_summarizer.pipeline.prediction as pred_module
    original = pred_module._cached_pipeline
    pred_module._cached_pipeline = None
    yield
    pred_module._cached_pipeline = original


@pytest.fixture
def mock_hf_pipeline():
    with patch("text_summarizer.pipeline.prediction.hf_pipeline") as mock_factory:
        mock_pipe = MagicMock(return_value=[{"summary_text": SAMPLE_SUMMARY}])
        mock_factory.return_value = mock_pipe
        yield mock_factory, mock_pipe


@pytest.fixture
def mock_tokenizer():
    with patch("text_summarizer.pipeline.prediction.AutoTokenizer") as mock_tok:
        mock_tok.from_pretrained.return_value = MagicMock()
        yield mock_tok


@pytest.fixture
def mock_config(tmp_path):
    config = MagicMock()
    config.model_path = str(tmp_path / "model")
    config.tokenizer_path = str(tmp_path / "tokenizer")
    return config


@pytest.fixture
def api_client(mock_hf_pipeline, mock_tokenizer, mock_config):
    with patch(
        "text_summarizer.pipeline.prediction.ConfigurationManager"
    ) as mock_cm:
        mock_cm.return_value.get_model_evaluation_config.return_value = mock_config
        from fastapi.testclient import TestClient
        from app import app
        yield TestClient(app)
