from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch

# Load tokenizer (we only need to load this once)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# --- Load Pre-trained Model (Before Fine-tuning) ---
pretrained_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# --- Load Fine-tuned Model (After Fine-tuning) ---
fine_tuned_model = AutoModelForSequenceClassification.from_pretrained("./novel_classifier_prototype_final")

# --- Data Loading and Preparation (Same as before) ---
try:
    with open(r"NASの実装方法の例\ツキナミ\このすば\input.txt", "r", encoding="utf-8") as f:
        novel_x = f.read()
except FileNotFoundError:
    print(f"Error: File 'このすば\\input.txt' not found.")
    exit()

try:
    with open(r"NASの実装方法の例\ツキナミ\ひげひろ\input.txt", "r", encoding="utf-8") as f:
        novel_y = f.read()
except FileNotFoundError:
    print(f"Error: File 'ひげひろ\\input.txt' not found.")
    exit()

texts = []
labels = []

paragraphs_x = novel_x.split("\n\n")
texts.extend(paragraphs_x)
labels.extend([0] * len(paragraphs_x))

paragraphs_y = novel_y.split("\n\n")
texts.extend(paragraphs_y)
labels.extend([1] * len(paragraphs_y))

train_texts, val_test_texts, train_labels, val_test_labels = train_test_split(
    texts, labels, test_size=0.3, stratify=labels, random_state=42
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    val_test_texts, val_test_labels, test_size=0.333, stratify=val_test_labels, random_state=42
)

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

datasets = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

def tokenize(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True
    )
datasets = datasets.map(tokenize, batched=True)

def compute_metrics(p: dict):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": accuracy, "f1": f1}

# --- Training Arguments (same as before for fine-tuning, but we won't train again in this run) ---
training_args = TrainingArguments(
    output_dir="./novel_classifier_prototype",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=50,
    report_to="none"
)

# --- Trainer for Fine-tuned Model (Same Trainer setup as before) ---
trainer_fine_tuned = Trainer(
    model=fine_tuned_model, # Use the fine-tuned model here
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    compute_metrics=compute_metrics
)

# --- Trainer for Pre-trained Model ---
trainer_pretrained = Trainer(
    model=pretrained_model, # Use the pre-trained model here
    args=training_args, # We can reuse the same training arguments (evaluation is relevant)
    eval_dataset=datasets["test"], # Evaluate on the test set
    compute_metrics=compute_metrics
)

# --- Evaluate Pre-trained Model ---
print("--- Evaluating Pre-trained Model (Before Fine-tuning) ---")
pretrained_results = trainer_pretrained.evaluate(datasets["test"])
print("Pre-trained Model Test Results:")
print(pretrained_results)

# --- Evaluate Fine-tuned Model ---
print("--- Evaluating Fine-tuned Model (After Fine-tuning) ---")
finetuned_results = trainer_fine_tuned.evaluate(datasets["test"])
print("Fine-tuned Model Test Results:")
print(finetuned_results)

print("\n--- Comparison ---")
print("Pre-trained Model F1 Score:", pretrained_results['eval_f1'])
print("Fine-tuned Model F1 Score:", finetuned_results['eval_f1'])
print("Pre-trained Model Accuracy:", pretrained_results['eval_accuracy'])
print("Fine-tuned Model Accuracy:", finetuned_results['eval_accuracy'])

print("\n--- Summary and Explanation ---")
print("The pre-trained model is a general-purpose language model. When evaluated on our novel text classification task *before* fine-tuning, it provides a baseline performance.")
print("Fine-tuning adapts the pre-trained model to the specific nuances of the writing styles in our novels.")
print("By comparing the metrics, you should ideally observe an improvement in the F1-score and Accuracy after fine-tuning. This improvement demonstrates that the model has learned to better distinguish between the writing styles of the two novels.")
print("The extent of the improvement depends on factors like the size and distinctiveness of your dataset, and the effectiveness of the fine-tuning process.")
