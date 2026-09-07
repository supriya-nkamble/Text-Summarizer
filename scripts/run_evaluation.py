"""Standalone entrypoint for re-running just the evaluation stage.

Re-scores the checkpoint configured in config.yaml against the full SAMSum
test split without repeating data ingestion/validation/transformation/training.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --model-path lidiya/bart-large-xsum-samsum --label bart-large-xsum-samsum
"""

import argparse

from text_summarizer.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        help="Override model checkpoint (local path or HF Hub repo id). Defaults to config.yaml's model_evaluation.model_path.",
    )
    parser.add_argument(
        "--tokenizer-path",
        help="Override tokenizer (local path or HF Hub repo id). Defaults to --model-path if given, else config.yaml's tokenizer_path.",
    )
    parser.add_argument(
        "--label",
        help="Label recorded in metric.csv/BASELINE_RESULTS.md for this run. Defaults to the resolved model path.",
    )
    args = parser.parse_args()

    tokenizer_path = args.tokenizer_path or args.model_path

    result = ModelEvaluationPipeline().main(
        model_path=args.model_path, tokenizer_path=tokenizer_path, label=args.label
    )
    print(result)


if __name__ == "__main__":
    main()
