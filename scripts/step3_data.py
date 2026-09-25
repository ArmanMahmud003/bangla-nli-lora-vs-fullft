# ---- STEP 3: dataset + metrics (train, validation AND test) ----
# Identical preprocessing to the first-pass notebook: same normalizer, same tokenizer,
# max_length=128, padding to max_length. The test split is tokenized here too,
# because Step 10 scores it inside every run. Imported by
# scripts/step10_train_with_test.py when it runs outside the notebook. Takes a few minutes the first time,
# then it is cached under the HuggingFace cache folder.
from datasets import load_dataset
from transformers import AutoTokenizer
from normalizer import normalize
import numpy as np
import evaluate

dataset = load_dataset("csebuetnlp/xnli_bn", revision="refs/convert/parquet")
tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")

def normalize_and_tokenize(examples):
    s1 = [normalize(t) for t in examples["sentence1"]]
    s2 = [normalize(t) for t in examples["sentence2"]]
    return tokenizer(s1, s2, truncation=True, padding="max_length", max_length=128)

tokenized_ds = {
    split: dataset[split].map(normalize_and_tokenize, batched=True)
    for split in ["train", "validation", "test"]
}

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1_metric.compute(predictions=preds, references=labels,
                                      average="macro")["f1"],
    }

print({k: len(v) for k, v in tokenized_ds.items()})
# expected: train 381449, validation 2419, test 4895
