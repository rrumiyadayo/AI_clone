from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch  # Import torch to check GPU availability

# Load pretrained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

try:
    with open(r"NASの実装方法の例\ツキナミ\このすば\input.txt", "r", encoding="utf-8") as f:
        novel_x = f.read()
except FileNotFoundError:
    print(f"Error: File 'このすば\\input.txt' not found. Please ensure the file is in the correct directory (currently assumed to be the same directory as your script).")
    exit()

try:
    with open(r"NASの実装方法の例\ツキナミ\ひげひろ\input.txt", "r", encoding="utf-8") as f:
        novel_y = f.read()
except FileNotFoundError:
    print(f"Error: File 'ひげひろ\\input.txt' not found. Please ensure the file is in the correct directory (currently assumed to be the same directory as your script).")
    exit()


texts = []
labels = []

# Split Novel text into paragraphs and add to dataset
paragraphs_x = novel_x.split("\n\n")
# --- REDUCED DATASET SIZE ---
paragraphs_x = paragraphs_x[:10]  # Take only the first 10 paragraphs from novel x
texts.extend(paragraphs_x)
labels.extend([0] * len(paragraphs_x))

paragraphs_y = novel_y.split("\n\n")
# --- REDUCED DATASET SIZE ---
paragraphs_y = paragraphs_y[:10]  # Take only the first 10 paragraphs from novel y
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
    output_dir="./reuslt/novel_classifier_prototype_final(small_ver)",
    evaluation_strategy="epoch",
    # --- REDUCED EPOCHS ---
    num_train_epochs=1, # Train for only 1 epoch
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_strategy="epoch",
    save_total_limit=1, # Keep only the last saved model
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

# Save the fine-tuned model
trainer.save_model("./reuslt/novel_classifier_prototype_final(small_ver)")

fine_tuned_model = AutoModelForSequenceClassification.from_pretrained("./reuslt/novel_classifier_prototype_final(small_ver)") # Load from your saved directory

# --- Choose a sample paragraph to test (you can pick one from your original novel_x or novel_y strings, or create a new one) ---
sample_text = """
夕食後、カズマはいつものように自室でゴロゴロしていた。
特に何をするでもなく、天井を眺めて時間を潰す。
今日は冒険者ギルドでクエストの依頼を探したが、ろくなものがなかった。
""" # Example paragraph from 'このすば' (novel_x - label 0)

# --- Tokenize the sample text ---
inputs = tokenizer(sample_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt") # 'pt' for PyTorch tensors

# --- Make a prediction with the fine-tuned model ---
with torch.no_grad(): # Disable gradient calculation for inference
    outputs = fine_tuned_model(**inputs) # Pass tokenized input to the model
    predictions = outputs.logits.argmax(-1) # Get the predicted class (0 or 1)

predicted_class = predictions.item() # Extract the class number from the tensor

# --- Interpret the prediction ---
if predicted_class == 0:
    predicted_novel = "このすば (Novel X)"
else:
    predicted_novel = "ひげひろ (Novel Y)"

print(f"\n--- Manual Prediction ---")
print(f"Sample Text: '{sample_text[:50]}...'") # Print the first 50 characters of the sample text
print(f"Predicted Novel: {predicted_novel} (Class: {predicted_class})")
