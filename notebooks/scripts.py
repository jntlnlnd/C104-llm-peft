import torch
from peft import LoraConfig, TaskType
from peft.peft_model import PeftModelForSequenceClassification
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import load_dataset
import datetime

import evaluate


def train_peft(peft_config):
    tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-3-1.8b")
    model = AutoModelForSequenceClassification.from_pretrained("llm-jp/llm-jp-3-1.8b")
    model.config.pad_token_id = tokenizer.pad_token_id

    class PatchedPeftModelForSequenceClassification(PeftModelForSequenceClassification):
        def add_adapter(self, adaper_name, peft_config, low_cpu_mem_usage: bool = False):
            super().add_adapter(adapter_name, peft_config)    

    peft_model = PatchedPeftModelForSequenceClassification(model, peft_config)
            
            
    ds = load_dataset(
        "csv",
        data_files={
            "train": "../data/train.tsv",
            "valid": "../data/valid.tsv",
            "test": "../data/test.tsv",
        },
        delimiter="\t",
    ).rename_column("label", "labels")

    class TokenizeCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, examples):
            encoding = self.tokenizer(
                [ex["poem"] for ex in examples],
                padding="longest",
                truncation=True,
                max_length=200,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": torch.tensor([ex["labels"] for ex in examples]),
            }

    roc_auc_evaluate = evaluate.load("roc_auc")
    acc_evaluate = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = map(torch.tensor, eval_pred)
        probs = torch.nn.functional.softmax(logits, dim=1)[:, 1]  # label=1の確率
        pred_labels = torch.argmax(logits, dim=1)  # 予測ラベル
        return {
            **roc_auc_evaluate.compute(prediction_scores=probs, references=labels),
            **acc_evaluate.compute(predictions=pred_labels, references=labels),
        }
    
    training_args = TrainingArguments(
        output_dir="../results/{:%Y%m%d%H%M%S}".format(datetime.datetime.now()),
        num_train_epochs=10,
        learning_rate=1e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=1.0,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["valid"],
        tokenizer=tokenizer,
        data_collator=TokenizeCollator(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()    
    # test
    print(compute_metrics((trainer.predict(ds["test"]).predictions, ds["test"]["labels"])))