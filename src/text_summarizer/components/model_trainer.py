from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from text_summarizer.entity import ModelTrainerConfig
import torch
import os


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_ckpt
        ).to(device)

        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

        dataset_samsum_pt = load_from_disk(self.config.data_path)

        training_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )

        trainer = Trainer(
            model=model_pegasus,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["train"],
            eval_dataset=dataset_samsum_pt["validation"],
        )

        trainer.train()
        # save model
        model_pegasus.save_pretrained(
            os.path.join(self.config.root_dir, "pegasus-model")
        )
        # save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))
