"""Kaggle GPU kernel: benchmark SAMSum summarization checkpoints.

Evaluates three checkpoints on the full SAMSum test split (819 examples)
with ROUGE + BERTScore, using the same fixed methodology as
src/text_summarizer/components/model_evaluation.py in the Text-Summarizer
repo (github.com/supriya-nkamble/Text-Summarizer):

1. sk1709/pegasus-samsum-baseline  - our current fine-tune (public HF repo)
2. lidiya/bart-large-xsum-samsum   - published reference, ROUGE-1/2/L = 53.3/28.4/44.1
3. philschmid/flan-t5-base-samsum - efficiency reference, published ROUGE-1 = 47.2

All three checkpoints are public, so no Hugging Face auth is required.
"""

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BERTSCORE_MODEL_TYPE = "distilbert-base-uncased"
ROUGE_NAMES = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
# Same SAMSum mirror config.yaml's data_ingestion.source_URL points at, so the
# test split here is byte-identical to the one the local pipeline evaluates
# against (the official HF "samsum" dataset has had loader/removal issues).
SAMSUM_ZIP_URL = "https://github.com/supriya-nkamble/Text-Summarizer/raw/main/data/samsumdata.zip"

CHECKPOINTS = [
    {"label": "pegasus-samsum-baseline (ours)", "model_id": "sk1709/pegasus-samsum-baseline"},
    {"label": "bart-large-xsum-samsum (lidiya)", "model_id": "lidiya/bart-large-xsum-samsum"},
    {"label": "flan-t5-base-samsum (philschmid)", "model_id": "philschmid/flan-t5-base-samsum"},
]


def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def evaluate_checkpoint(model_id, test_dialogues, test_summaries, batch_size=8):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(DEVICE)

    rouge_metric = load("rouge")
    bertscore_metric = load("bertscore")

    dialogue_batches = list(batched(test_dialogues, batch_size))
    summary_batches = list(batched(test_summaries, batch_size))

    for dialogue_batch, summary_batch in zip(dialogue_batches, summary_batches):
        inputs = tokenizer(
            dialogue_batch,
            max_length=256,
            truncation=True,
            padding=True,
            return_tensors="pt",
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
        # PEGASUS emits a literal "<n>" sentence separator; harmless no-op for other tokenizers.
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

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return row


def load_samsum_test_split():
    response = requests.get(SAMSUM_ZIP_URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content)) as zf:
        with zf.open("samsum-test.csv") as f:
            df = pd.read_csv(f)
    return df["dialogue"].tolist(), df["summary"].tolist()


def main():
    dialogues, summaries = load_samsum_test_split()

    results = []
    for checkpoint in CHECKPOINTS:
        print(f"Evaluating {checkpoint['label']} ({checkpoint['model_id']})...")
        row = evaluate_checkpoint(checkpoint["model_id"], dialogues, summaries)
        row["label"] = checkpoint["label"]
        row["model_id"] = checkpoint["model_id"]
        results.append(row)
        print(row)

    columns = [
        "label",
        "model_id",
        *ROUGE_NAMES,
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
    ]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv("/kaggle/working/samsum_benchmark_results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
