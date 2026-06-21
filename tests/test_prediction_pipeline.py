import pytest
from unittest.mock import patch, MagicMock
from tests.conftest import SAMPLE_DIALOGUE, SAMPLE_SUMMARY


def _make_pipeline(mock_hf_pipeline, mock_tokenizer, mock_config):
    from text_summarizer.pipeline.prediction import PredictionPipeline
    with patch(
        "text_summarizer.pipeline.prediction.ConfigurationManager"
    ) as mock_cm:
        mock_cm.return_value.get_model_evaluation_config.return_value = mock_config
        return PredictionPipeline()


def test_predict_returns_string(mock_hf_pipeline, mock_tokenizer, mock_config):
    pipeline = _make_pipeline(mock_hf_pipeline, mock_tokenizer, mock_config)
    with patch(
        "text_summarizer.pipeline.prediction.ConfigurationManager"
    ) as mock_cm:
        mock_cm.return_value.get_model_evaluation_config.return_value = mock_config
        result = pipeline.predict(SAMPLE_DIALOGUE)
    assert isinstance(result, str)
    assert result == SAMPLE_SUMMARY


def test_predict_calls_hf_pipeline_with_gen_kwargs(mock_hf_pipeline, mock_tokenizer, mock_config):
    _, mock_pipe = mock_hf_pipeline
    pipeline = _make_pipeline(mock_hf_pipeline, mock_tokenizer, mock_config)
    with patch(
        "text_summarizer.pipeline.prediction.ConfigurationManager"
    ) as mock_cm:
        mock_cm.return_value.get_model_evaluation_config.return_value = mock_config
        pipeline.predict(SAMPLE_DIALOGUE)
    call_kwargs = mock_pipe.call_args[1]
    assert call_kwargs["length_penalty"] == 0.8
    assert call_kwargs["num_beams"] == 8
    assert call_kwargs["max_length"] == 128


def test_model_loaded_only_once_across_calls(mock_hf_pipeline, mock_tokenizer, mock_config):
    """Pipeline factory should be called once; subsequent predict() calls reuse cache."""
    mock_factory, mock_pipe = mock_hf_pipeline
    pipeline = _make_pipeline(mock_hf_pipeline, mock_tokenizer, mock_config)
    with patch(
        "text_summarizer.pipeline.prediction.ConfigurationManager"
    ) as mock_cm:
        mock_cm.return_value.get_model_evaluation_config.return_value = mock_config
        pipeline.predict(SAMPLE_DIALOGUE)
        pipeline.predict(SAMPLE_DIALOGUE)
        pipeline.predict(SAMPLE_DIALOGUE)
    assert mock_factory.call_count == 1


def test_predict_passes_text_to_pipeline(mock_hf_pipeline, mock_tokenizer, mock_config):
    _, mock_pipe = mock_hf_pipeline
    pipeline = _make_pipeline(mock_hf_pipeline, mock_tokenizer, mock_config)
    with patch(
        "text_summarizer.pipeline.prediction.ConfigurationManager"
    ) as mock_cm:
        mock_cm.return_value.get_model_evaluation_config.return_value = mock_config
        pipeline.predict(SAMPLE_DIALOGUE)
    assert mock_pipe.call_args[0][0] == SAMPLE_DIALOGUE
