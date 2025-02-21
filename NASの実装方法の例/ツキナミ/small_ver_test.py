from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
import random # Import the random module

# Load tokenizer (we only need to load this once)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# --- Load Pre-trained Model (Before Fine-tuning) ---
pretrained_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# --- Load Fine-tuned Model (After Fine-tuning) ---
fine_tuned_model = AutoModelForSequenceClassification.from_pretrained("./reuslt/novel_classifier_prototype_final(small_ver)")

# --- Data Loading and Preparation (Same as before) ---
try:
    with open(r"NASの実装方法の例\ツキナミ\このすば\input.txt", "r", encoding="utf-8") as f:
        novel_x_text = f.read() # Keep original full text for random sampling
except FileNotFoundError:
    print(f"Error: File 'このすば\\input.txt' not found.")
    exit()

try:
    with open(r"NASの実装方法の例\ツキナミ\ひげひろ\input.txt", "r", encoding="utf-8") as f:
        novel_y_text = f.read() # Keep original full text for random sampling
except FileNotFoundError:
    print(f"Error: File 'ひげひろ\\input.txt' not found.")
    exit()

# Split novels into paragraphs for training dataset creation (same as before)
novel_x_paragraphs = novel_x_text.split("\n\n")
novel_y_paragraphs = novel_y_text.split("\n\n")

texts = []
labels = []
texts.extend(novel_x_paragraphs)
labels.extend([0] * len(novel_x_paragraphs))
texts.extend(novel_y_paragraphs)
labels.extend([1] * len(novel_y_paragraphs))

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

# --- Training Arguments (same as before - not training again in this run) ---
training_args = TrainingArguments(
    output_dir="./reuslt/novel_classifier_prototype_final(small_ver)",
    evaluation_strategy="epoch",
    num_train_epochs=1,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=100,
    report_to="none"
)

# --- Trainer for Fine-tuned Model (Evaluation only) ---
trainer_fine_tuned = Trainer(
    model=fine_tuned_model,
    args=training_args,
    train_dataset=datasets["train"], # Not used for evaluation
    eval_dataset=datasets["validation"], # Not used for evaluation
    compute_metrics=compute_metrics # Not used for evaluation
)

# --- Trainer for Pre-trained Model (Evaluation only) ---
trainer_pretrained = Trainer(
    model=pretrained_model,
    args=training_args, # Reusing training args for evaluation
    eval_dataset=datasets["test"], # Not used for our manual test
    compute_metrics=compute_metrics # Not used for our manual test
)

# --- Function to extract random 5-line samples ---
def get_random_samples(novel_text, novel_label, num_samples=10, lines_per_sample=5):
    lines = novel_text.splitlines()
    samples = []
    for _ in range(num_samples):
        start_line_index = random.randint(0, max(0, len(lines) - lines_per_sample)) # Ensure start index is valid
        sample_lines = lines[start_line_index:start_line_index + lines_per_sample]
        sample_text = "\n".join(sample_lines)
        samples.append({"text": sample_text, "label": novel_label}) # Store text and correct label
    return samples

# --- Generate random samples from both novels ---
num_test_samples_per_novel = 5 # 5 samples from each novel, total 10
novel_x_samples = get_random_samples(novel_x_text, 0, num_samples=num_test_samples_per_novel)
novel_y_samples = get_random_samples(novel_y_text, 1, num_samples=num_test_samples_per_novel)
test_samples = novel_x_samples + novel_y_samples
random.shuffle(test_samples) # Shuffle the samples for a mixed test

# --- Prepare for table output and score tracking ---
table_data = []
ft_correct_count = 0 # Counter for fine-tuned model's correct predictions
pt_correct_count = 0 # Counter for pre-trained model's correct predictions

print("\n--- Random Sample Prediction Comparison ---")
print("| Sample Text (初5字)     | True Label | Fine-tuned Correct? | Pre-trained Correct? |") # Wider Sample Text Column
print("|--------------------------|------------|----------------------|-----------------------|") # Adjusted separator width

# --- Loop through test samples and make predictions ---
for sample in test_samples:
    sample_text = sample["text"]
    true_label = sample["label"]

    # Tokenize sample text
    inputs = tokenizer(sample_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    # --- Fine-tuned model prediction ---
    with torch.no_grad():
        outputs_ft = fine_tuned_model(**inputs)
        predicted_class_ft = outputs_ft.logits.argmax(-1).item()

    # --- Pre-trained model prediction ---
    with torch.no_grad():
        outputs_pt = pretrained_model(**inputs)
        predicted_class_pt = outputs_pt.logits.argmax(-1).item()

    # --- Determine correctness and symbols ---
    ft_correct = predicted_class_ft == true_label
    pt_correct = predicted_class_pt == true_label
    ft_symbol = "○" if ft_correct else "×"
    pt_symbol = "○" if pt_correct else "×"

    # --- Increment correct prediction counters ---
    if ft_correct:
        ft_correct_count += 1
    if pt_correct:
        pt_correct_count += 1

    # --- Create table row ---
    table_row = {
        "sample_text": sample_text[:5] + "...", # First 5 chars of sample text
        "true_label": true_label,
        "ft_correct": ft_symbol,
        "pt_correct": pt_symbol
    }
    table_data.append(table_row)

    # --- Print table row to console ---
    print(f"| {table_row['sample_text']:<24} | {table_row['true_label']:<10} | {table_row['ft_correct']:<20} | {table_row['pt_correct']:<21} |") # Wider Sample Text Column

print("|--------------------------|------------|----------------------|-----------------------|") # Adjusted separator width

# --- Print Total Scores ---
print(f"| Total Correct              |            | Fine-tuned: {ft_correct_count}/{len(test_samples):<2} | Pre-trained: {pt_correct_count}/{len(test_samples):<2} |") # Total score row, Adjusted width
print("|--------------------------|------------|----------------------|-----------------------|") # Adjusted separator width


print("\n--- Table Summary ---")
print("○: Correct Prediction, ×: Incorrect Prediction")
print("True Label: 0 = このすば (Novel X), 1 = ひげひろ (Novel Y)")
print("Fine-tuned Model: Model after training on novel writing styles.")
print("Pre-trained Model: Original distilbert-base-uncased model (no fine-tuning).")