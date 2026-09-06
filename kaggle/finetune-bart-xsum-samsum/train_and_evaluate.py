"""Kaggle GPU kernel: fine-tune facebook/bart-large-xsum on SAMSum.

Base checkpoint choice: bart-large-xsum, not bart-large-cnn/pegasus-cnn_dailymail
(see PLAN.md's model selection analysis) - XSum's short, highly-abstractive
summary style matches SAMSum far better than CNN/DailyMail's longer,
extractive-leaning style. Benchmarking (kaggle/evaluate-samsum-baselines)
already confirmed lidiya/bart-large-xsum-samsum outperforms our
pegasus-cnn_dailymail baseline and philschmid/flan-t5-base-samsum on both
ROUGE and BERTScore.

Trains, then runs the same full-test-set ROUGE+BERTScore evaluation as
kaggle/evaluate-samsum-baselines/evaluate_models.py, and saves the trained
model+tokenizer as kernel output (pulled back locally afterward - no HF Hub
auth needed for this step).
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
from datasets import Dataset
from evaluate import load
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

MODEL_CKPT = "facebook/bart-large-xsum"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BERTSCORE_MODEL_TYPE = "distilbert-base-uncased"
ROUGE_NAMES = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
SAMSUM_ZIP_URL = "https://github.com/supriya-nkamble/Text-Summarizer/raw/main/data/samsumdata.zip"
OUTPUT_DIR = "/kaggle/working"


def load_samsum_splits():
    response = requests.get(SAMSUM_ZIP_URL, timeout=60)
    response.raise_for_status()
    splits = {}
    with zipfile.ZipFile(BytesIO(response.content)) as zf:
        for split, filename in [
            ("train", "samsum-train.csv"),
            ("validation", "samsum-validation.csv"),
            ("test", "samsum-test.csv"),
        ]:
            with zf.open(filename) as f:
                df = pd.read_csv(f)
            splits[split] = df.dropna(subset=["dialogue", "summary"])
    return splits


def tokenize(tokenizer, dialogues, summaries):
    input_encodings = tokenizer(dialogues, max_length=256, truncation=True)
    target_encodings = tokenizer(text_target=summaries, max_length=64, truncation=True)
    return {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": target_encodings["input_ids"],
    }


def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def evaluate_on_test(model, tokenizer, test_dialogues, test_summaries, batch_size=8):
    rouge_metric = load("rouge")
    bertscore_metric = load("bertscore")

    for dialogue_batch, summary_batch in zip(
        batched(test_dialogues, batch_size), batched(test_summaries, batch_size)
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
    return row


def main():
    splits = load_samsum_splits()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CKPT).to(DEVICE)

    train_ds = Dataset.from_pandas(splits["train"]).map(
        lambda batch: tokenize(tokenizer, batch["dialogue"], batch["summary"]), batched=True
    )
    val_ds = Dataset.from_pandas(splits["validation"]).map(
        lambda batch: tokenize(tokenizer, batch["dialogue"], batch["summary"]), batched=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/checkpoints",
        num_train_epochs=3,
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        warmup_steps=500,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("Starting training...")
    trainer.train()

    model_dir = f"{OUTPUT_DIR}/bart-large-xsum-samsum-v2"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Saved fine-tuned model to {model_dir}")

    print("Running final test-set evaluation...")
    model.eval()
    row = evaluate_on_test(model, tokenizer, splits["test"]["dialogue"].tolist(), splits["test"]["summary"].tolist())
    row["label"] = "bart-large-xsum-samsum-v2 (ours, fine-tuned)"
    row["model_id"] = "local"

    columns = ["label", "model_id", *ROUGE_NAMES, "bertscore_precision", "bertscore_recall", "bertscore_f1"]
    df = pd.DataFrame([row], columns=columns)
    df.to_csv(f"{OUTPUT_DIR}/finetune_results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
