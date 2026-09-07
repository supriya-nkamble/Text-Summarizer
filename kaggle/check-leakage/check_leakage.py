"""Kaggle GPU kernel: quantify the effect of train/test dialogue overlap.

41 of SAMSum's 819 test examples have dialogue text byte-identical to an
entry in the 14,732-example training set (verified locally: zero `id`
overlap between splits, but real text overlap - likely duplicate/near-
duplicate conversations baked into the original corpus, not a split bug).

Evaluates the already fine-tuned bart-large-xsum-samsum-v2 model separately
on the "leaked" (41) and "clean" (778) subsets of the test set, to see
whether the leaked subset's scores are inflated by memorization.
"""

import subprocess
import sys
import zipfile
from io import BytesIO

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "bert_score", "rouge_score", "evaluate"],
    check=True,
)

import glob
import os

import pandas as pd
import requests
import torch
from evaluate import load
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def find_mounted_model_dir():
    print("Contents of /kaggle/input:")
    for path in glob.glob("/kaggle/input/**", recursive=True):
        print(" ", path)
    matches = glob.glob("/kaggle/input/**/config.json", recursive=True)
    if not matches:
        raise FileNotFoundError("No config.json found under /kaggle/input - check kernel_sources mount path above")
    return os.path.dirname(matches[0])


MODEL_ID = find_mounted_model_dir()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BERTSCORE_MODEL_TYPE = "distilbert-base-uncased"
ROUGE_NAMES = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
SAMSUM_ZIP_URL = "https://github.com/supriya-nkamble/Text-Summarizer/raw/main/data/samsumdata.zip"


def load_samsum_splits():
    response = requests.get(SAMSUM_ZIP_URL, timeout=60)
    response.raise_for_status()
    splits = {}
    with zipfile.ZipFile(BytesIO(response.content)) as zf:
        for split, filename in [
            ("train", "samsum-train.csv"),
            ("test", "samsum-test.csv"),
        ]:
            with zf.open(filename) as f:
                splits[split] = pd.read_csv(f).dropna(subset=["dialogue", "summary"])
    return splits


def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def evaluate_subset(model, tokenizer, dialogues, summaries, batch_size=8):
    rouge_metric = load("rouge")
    bertscore_metric = load("bertscore")

    for dialogue_batch, summary_batch in zip(
        batched(dialogues, batch_size), batched(summaries, batch_size)
    ):
        inputs = tokenizer(
            dialogue_batch, max_length=256, truncation=True, padding=True, return_tensors="pt"
        )
        generated = model.generate(
            input_ids=inputs["input_ids"].to(DEVICE),
            attention_mask=inputs["attention_mask"].to(DEVICE),
            length_penalty=0.8,
            num_beams=8,
            max_length=128,
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
    row["n"] = len(dialogues)
    return row


def main():
    splits = load_samsum_splits()
    train_dialogues = set(splits["train"]["dialogue"])
    test_df = splits["test"]

    is_leaked = test_df["dialogue"].isin(train_dialogues)
    leaked_df = test_df[is_leaked]
    clean_df = test_df[~is_leaked]
    print(f"leaked: {len(leaked_df)}, clean: {len(clean_df)}, total: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()

    results = []
    for label, df in [("leaked (in train)", leaked_df), ("clean (not in train)", clean_df), ("full test set", test_df)]:
        print(f"Evaluating {label} ({len(df)} examples)...")
        row = evaluate_subset(model, tokenizer, df["dialogue"].tolist(), df["summary"].tolist())
        row["subset"] = label
        results.append(row)
        print(row)

    columns = ["subset", "n", *ROUGE_NAMES, "bertscore_precision", "bertscore_recall", "bertscore_f1"]
    out_df = pd.DataFrame(results, columns=columns)
    out_df.to_csv("/kaggle/working/leakage_check_results.csv", index=False)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
