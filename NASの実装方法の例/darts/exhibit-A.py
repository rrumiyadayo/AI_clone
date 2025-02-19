import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
import time

# 警告の抑制
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

print(f"{'-'*50}\nScript started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'-'*50}")

# データセットの準備
def prepare_dataset():
    print("\n" + "="*40 + "\nLoading dataset...")
    start_time = time.time()
    # IMDBレビューのデータセットを読み込み
    dataset = load_dataset("imdb")
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
    
    print("\n" + "="*40 + "\nSplitting dataset...")
    start_time = time.time()

    # trainデータセットからtrainとvalidationに分割
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        dataset["train"]["text"], dataset["train"]["label"], test_size=0.2, random_state=42
    )
    print(f"Dataset split completed in {time.time() - start_time:.2f} seconds")
    
    print("\n" + "="*40 + "\nLoading tokenizer...")
    start_time = time.time()

    # トークナイザの読み込み
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    print(f"Tokenizer loaded in {time.time() - start_time:.2f} seconds")
    
    print("\n" + "="*40 + "\nTokenizing dataset...")
    start_time = time.time()

    # データセットのトークン化
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_train_dataset = list(map(tokenize_function, [{"text": text} for text in train_texts]))
    tokenized_val_dataset = list(map(tokenize_function, [{"text": text} for text in val_texts]))
    
    print(f"Tokenization completed in {time.time() - start_time:.2f} seconds")
    
    print("\n" + "="*40 + "\nAdding labels...")
    start_time = time.time()

    tokenized_train_dataset = list(map(tokenize_function, [{"text": text} for text in train_texts]))
    tokenized_val_dataset = list(map(tokenize_function, [{"text": text} for text in val_texts]))

    # ラベルの追加
    for i, label in enumerate(train_labels):
        tokenized_train_dataset[i]["labels"] = label

    for i, label in enumerate(val_labels):
        tokenized_val_dataset[i]["labels"] = label
    
    print(f"Labels added in {time.time() - start_time:.2f} seconds")
    
    print("\n" + "="*40 + "\nReturning datasets...")
    return tokenized_train_dataset, tokenized_val_dataset, tokenizer

    return tokenized_train_dataset, tokenized_val_dataset, tokenizer

# カスタムデータセットの作成
class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long)
        }

# モデルの準備
def prepare_model():
    print("\n" + "="*40 + "\nLoading model...")
    start_time = time.time()
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    return model

# モデルの評価
def evaluate_model(model, dataloader, device):
    print("\n" + "="*40 + "\nEvaluating model...")
    start_time = time.time()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
    return accuracy

# ファインチューニング
def train_model(model, train_loader, val_loader, device, num_epochs=3):
    print("\n" + "="*40 + "\nStarting training...")
    start_time = time.time()
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"\nEpoch {epoch + 1} complete. Average Loss: {avg_loss:.4f}")
        
        # バリデーションの評価
        print("\nRunning validation...")
        val_accuracy = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_accuracy)
        print(f"Validation Accuracy: {val_accuracy}%")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    return train_losses, val_accuracies

    return train_losses, val_accuracies

# 結果の可視化
def plot_results(train_losses, val_accuracies, initial_accuracy):
    print("\n" + "="*40 + "\nGenerating plots...")
    plt.figure(figsize=(12, 6))

    # 損失のプロット
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()

    # 正解率のプロット
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
    plt.axhline(y=initial_accuracy, color='gray', linestyle='--', label=f'Initial Accuracy ({initial_accuracy}%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()
    print("Plots generated successfully!")

# メイン関数
def main():
    print(f"\n{'-'*50}\nStarting main function at {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'-'*50}")
    # データセットの準備
    tokenized_train_dataset, tokenized_val_dataset, tokenizer = prepare_dataset()

    print("\nCreating datasets...")
    # カスタムデータセットの作成
    train_dataset = SimpleDataset(tokenized_train_dataset)
    val_dataset = SimpleDataset(tokenized_val_dataset)
    print("Datasets created successfully!")

    print("\nCreating dataloaders...")
    # DataLoaderの作成
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    print("Dataloaders created successfully!")

    # モデルの準備
    model = prepare_model()
    print("\nModel preparation complete!")

    print("\nSetting up device...")
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\nEvaluating initial model...")
    # 初期モデルの評価
    initial_accuracy = evaluate_model(model, val_loader, device)
    print(f"Initial Model Validation Accuracy: {initial_accuracy}%")

    # ファインチューニング
    train_losses, val_accuracies = train_model(model, train_loader, val_loader, device)

    print("\nEvaluating fine-tuned model...")
    # ファインチューニング後のモデルの評価
    final_accuracy = evaluate_model(model, val_loader, device)
    print(f"Fine-Tuned Model Validation Accuracy: {final_accuracy}%")

    # 結果の可視化
    plot_results(train_losses, val_accuracies, initial_accuracy)
    
    print(f"\n{'-'*50}\nScript completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'-'*50}")

if __name__ == "__main__":
    main()
