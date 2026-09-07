from text_summarizer.components.model_evaluation import ModelEvaluation
from text_summarizer.config.configuration import ConfigurationManager


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self, model_path=None, tokenizer_path=None, label=None):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        return model_evaluation.evaluate(
            model_path=model_path, tokenizer_path=tokenizer_path, label=label
        )
