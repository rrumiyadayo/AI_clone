from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch

# Load pretrained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

try:
    with open(r"このすば\input.txt", "r", encoding="utf-8") as f:
        novel_x = f.read()
except FileNotFoundError:
    print(f"Error: File 'このすば\\input.txt' not found. Please ensure the file is in the correct directory (currently assumed to be the same directory as your script).")
    exit()

try:
    with open(r"ひげひろ\input.txt", "r", encoding="utf-8") as f:
        novel_y = f.read()
except FileNotFoundError:
    print(f"Error: File 'ひげひろ\\input.txt' not found. Please ensure the file is in the correct directory (currently assumed to be the same directory as your script).")
    exit()


texts = []
labels = []

# Split Novel text into paragraphs and add to dataset
paragraphs_x = novel_x.split("\n\n")
# --- REDUCED DATASET SIZE ---
paragraphs_x = paragraphs_x[:10]  # Take limited paragraphs from novel x
texts.extend(paragraphs_x)
labels.extend([0] * len(paragraphs_x))

paragraphs_y = novel_y.split("\n\n")
# --- REDUCED DATASET SIZE ---
paragraphs_y = paragraphs_y[:10]  # Take limited paragraphs from novel y
texts.extend(paragraphs_y)
labels.extend([1] * len(paragraphs_y))

# Split data into training, validation, and test sets
train_texts, val_test_texts, train_labels, val_test_labels = train_test_split(
    texts, labels, test_size=0.3, stratify=labels, random_state=42
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    val_test_texts, val_test_labels, test_size=0.333, stratify=val_test_labels, random_state=42
)

# Create Hugging Face datasets
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

datasets = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

# Tokenize datasets
def tokenize(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128 # --- REDUCE MAX SEQUENCE LENGTH --- Limit token length to 128
    )
datasets = datasets.map(tokenize, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./result/novel_classifier_prototype_final(small_ver)",
    evaluation_strategy="epoch",
    # --- REDUCED EPOCHS ---
    num_train_epochs=1, # Train for only a limited epoch
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    # --- REMOVE checkpoint saving arguments ---
    #save_strategy="epoch", # No need for epoch saving for final model in root dir
    #save_total_limit=1,  # No need for epoch saving for final model in root dir
    # --- REMOVE BEST MODEL LOADING FOR QUICK DEMO ---
    # load_best_model_at_end=True,
    # metric_for_best_model="f1",
    # greater_is_better=True,
    logging_steps=100, # Log less frequently
    report_to="none"
)

# Define compute metrics function
def compute_metrics(p: dict):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": accuracy, "f1": f1}

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    compute_metrics=compute_metrics
)

# --- Important GPU/CPU Check and Information ---
print("--- GPU / CPU Check ---")
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU is available! Training will be performed on: {torch.cuda.get_device_name(0)}") # Print GPU name
else:
    device = torch.device('cpu')
    print("No GPU detected. Training will be performed on CPU. This will be slower.")
    print("If you expect to an AMD GPU, please ensure you have installed PyTorch with ROCm support correctly.")
print("--- Starting Training ---")

# Fine-tune the model
trainer.train()

# Evaluate on test set
test_results = trainer.evaluate(datasets["test"])
print("Test Results:")
print(test_results)


print("--- Training and Evaluation Complete ---")
trainer.save_model("./result/novel_classifier_prototype_final(small_ver)")
print("Model saved to ./result/novel_classifier_prototype_final(small_ver)")
print("Please check the './result/novel_classifier_prototype_final(small_ver)' directory for the final model files.")