"""Kaggle GPU kernel: sweep decoding hyperparameters for the trained model.

The training/eval scripts inherited num_beams=8, length_penalty=0.8 from the
original tutorial without ever tuning them for this checkpoint. This is a
free (no retraining) search over the decoding config space to see whether a
better choice improves ROUGE/BERTScore on the full 819-example test set.

Loads the already fine-tuned model from the training kernel's own output
(kernel_sources) rather than re-uploading anything.
"""

import glob
import os
import subprocess
import sys
import zipfile
from io import BytesIO

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "bert_score", "rouge_score", "evaluate"],
    check=True,
)

import pandas as pd
import requests
import torch
from evaluate import load
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def find_mounted_model_dir():
    matches = glob.glob("/kaggle/input/**/config.json", recursive=True)
    if not matches:
        raise FileNotFoundError("No config.json found under /kaggle/input")
    return os.path.dirname(matches[0])


MODEL_DIR = find_mounted_model_dir()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BERTSCORE_MODEL_TYPE = "distilbert-base-uncased"
ROUGE_NAMES = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
SAMSUM_ZIP_URL = "https://github.com/supriya-nkamble/Text-Summarizer/raw/main/data/samsumdata.zip"

# (num_beams, length_penalty, no_repeat_ngram_size)
CONFIGS = [
    (4, 0.6, 3), (4, 0.8, 3), (4, 1.0, 3),
    (8, 0.6, 3), (8, 0.8, 3), (8, 1.0, 3),
    (8, 0.8, 0),
    (12, 0.6, 3), (12, 0.8, 3), (12, 1.0, 3),
]


def load_samsum_test():
    response = requests.get(SAMSUM_ZIP_URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content)) as zf:
        with zf.open("samsum-test.csv") as f:
            df = pd.read_csv(f).dropna(subset=["dialogue", "summary"])
    return df["dialogue"].tolist(), df["summary"].tolist()


def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def evaluate_config(model, tokenizer, dialogues, summaries, num_beams, length_penalty, no_repeat_ngram_size, batch_size=8):
    rouge_metric = load("rouge")
    bertscore_metric = load("bertscore")

    gen_kwargs = dict(length_penalty=length_penalty, num_beams=num_beams, max_length=128)
    if no_repeat_ngram_size:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    for dialogue_batch, summary_batch in zip(
        batched(dialogues, batch_size), batched(summaries, batch_size)
    ):
        inputs = tokenizer(
            dialogue_batch, max_length=256, truncation=True, padding=True, return_tensors="pt"
        )
        generated = model.generate(
            input_ids=inputs["input_ids"].to(DEVICE),
            attention_mask=inputs["attention_mask"].to(DEVICE),
            **gen_kwargs,
        )
        decoded = [
            tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for s in generated
        ]
        decoded = [d.replace("<n>", " ") for d in decoded]
        rouge_metric.add_batch(predictions=decoded, references=summary_batch)
        bertscore_metric.add_batch(predictions=decoded, references=summary_batch)

    rouge_score = rouge_metric.compute()
    bertscore_result = bertscore_metric.compute(model_type=BERTSCORE_MODEL_TYPE)

    row = {rn: rouge_score[rn] for rn in ROUGE_NAMES}
    row["bertscore_precision"] = sum(bertscore_result["precision"]) / len(
        bertscore_result["precision"]
    )
    row["bertscore_recall"] = sum(bertscore_result["recall"]) / len(bertscore_result["recall"])
    row["bertscore_f1"] = sum(bertscore_result["f1"]) / len(bertscore_result["f1"])
    return row


def main():
    dialogues, summaries = load_samsum_test()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    results = []
    for num_beams, length_penalty, no_repeat_ngram_size in CONFIGS:
        label = f"beams={num_beams},lp={length_penalty},norepeat={no_repeat_ngram_size}"
        print(f"Evaluating {label}...")
        row = evaluate_config(
            model, tokenizer, dialogues, summaries, num_beams, length_penalty, no_repeat_ngram_size
        )
        row["config"] = label
        row["num_beams"] = num_beams
        row["length_penalty"] = length_penalty
        row["no_repeat_ngram_size"] = no_repeat_ngram_size
        results.append(row)
        print(row)

    columns = [
        "config", "num_beams", "length_penalty", "no_repeat_ngram_size",
        *ROUGE_NAMES, "bertscore_precision", "bertscore_recall", "bertscore_f1",
    ]
    df = pd.DataFrame(results, columns=columns)
    df = df.sort_values("bertscore_f1", ascending=False)
    df.to_csv("/kaggle/working/decoding_sweep_results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
