from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from datasets import load_from_disk
from evaluate import load
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from text_summarizer.constants import PARAMS_FILE_PATH
from text_summarizer.entity import ModelEvaluationConfig
from text_summarizer.utils.common import read_yaml

ROUGE_NAMES = ["rouge1", "rouge2", "rougeL", "rougeLsum"]


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """split the dataset into smaller batches that we can process simultaneously
        Yield successive batch-sized chunks from list_of_elements."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(
        self,
        dataset,
        metrics,
        model,
        tokenizer,
        batch_size=16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        column_text="dialogue",
        column_summary="summary",
    ):
        article_batches = list(
            self.generate_batch_sized_chunks(dataset[column_text], batch_size)
        )
        target_batches = list(
            self.generate_batch_sized_chunks(dataset[column_summary], batch_size)
        )

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)
        ):

            inputs = tokenizer(
                article_batch,
                max_length=1024,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                length_penalty=0.8,
                num_beams=8,
                max_length=128,
            )
            """ parameter for length penalty ensures that the model does not generate sequences that are too long. """

            # Finally, we decode the generated texts,
            # replace the  token, and add the decoded texts with the references to the metric.
            decoded_summaries = [
                tokenizer.decode(
                    s, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for s in summaries
            ]

            decoded_summaries = [d.replace("<n>", " ") for d in decoded_summaries]

            for metric in metrics:
                metric.add_batch(predictions=decoded_summaries, references=target_batch)

    def evaluate(self, model_path=None, tokenizer_path=None, label=None):
        model_path = model_path or self.config.model_path
        tokenizer_path = tokenizer_path or self.config.tokenizer_path
        label = label or str(model_path)

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

        # loading data
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        rouge_metric = load("rouge")
        bertscore_metric = load("bertscore")

        self.calculate_metric_on_test_ds(
            dataset_samsum_pt["test"],
            [rouge_metric, bertscore_metric],
            model,
            tokenizer,
            batch_size=8,
            column_text="dialogue",
            column_summary="summary",
        )

        rouge_score = rouge_metric.compute()
        bertscore_result = bertscore_metric.compute(lang="en")

        row = {rn: rouge_score[rn] for rn in ROUGE_NAMES}
        row["bertscore_precision"] = sum(bertscore_result["precision"]) / len(
            bertscore_result["precision"]
        )
        row["bertscore_recall"] = sum(bertscore_result["recall"]) / len(
            bertscore_result["recall"]
        )
        row["bertscore_f1"] = sum(bertscore_result["f1"]) / len(bertscore_result["f1"])
        row["label"] = label
        row["model_path"] = str(model_path)
        row["evaluated_at"] = datetime.now(timezone.utc).isoformat()

        self._append_metric_row(row)
        return row

    def _append_metric_row(self, row):
        metric_path = Path(self.config.metric_file_name)
        columns = [
            "label",
            "model_path",
            "evaluated_at",
            *ROUGE_NAMES,
            "bertscore_precision",
            "bertscore_recall",
            "bertscore_f1",
        ]
        new_row = pd.DataFrame([row], columns=columns)
        if metric_path.exists():
            existing = pd.read_csv(metric_path)
            df = pd.concat([existing, new_row], ignore_index=True)
        else:
            df = new_row
        df.to_csv(metric_path, index=False)
        self._write_baseline_report(df, metric_path.parent / "BASELINE_RESULTS.md")

    def _write_baseline_report(self, df, report_path: Path):
        params = read_yaml(PARAMS_FILE_PATH).TrainingArguments
        lines = [
            "# Baseline Evaluation Results",
            "",
            f"Training hyperparameters (from `{PARAMS_FILE_PATH}`): "
            f"{params.num_train_epochs} epoch(s), "
            f"per_device_train_batch_size={params.per_device_train_batch_size}, "
            f"gradient_accumulation_steps={params.gradient_accumulation_steps}, "
            f"warmup_steps={params.warmup_steps}.",
            "",
            df.to_markdown(index=False),
            "",
        ]
        report_path.write_text("\n".join(lines))
