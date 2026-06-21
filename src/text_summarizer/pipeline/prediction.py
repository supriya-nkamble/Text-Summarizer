from transformers import AutoTokenizer
from transformers import pipeline as hf_pipeline

from text_summarizer.config.configuration import ConfigurationManager
from text_summarizer.logging import logger

_cached_pipeline = None


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text: str) -> str:
        global _cached_pipeline
        if _cached_pipeline is None:
            logger.info("Loading summarization model from %s", self.config.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
            _cached_pipeline = hf_pipeline(
                "summarization",
                model=str(self.config.model_path),
                tokenizer=tokenizer,
            )
            logger.info("Model loaded and cached")

        gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}
        output: str = _cached_pipeline(text, **gen_kwargs)[0]["summary_text"]
        logger.info("Generated summary (%d chars)", len(output))
        return output
