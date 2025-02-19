import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings

# 警告の抑制
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# データセットの準備
def prepare_dataset():
    # IMDBレビューのデータセットを読み込み
    dataset = load_dataset("imdb")

    # trainデータセットからtrainとvalidationに分割
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        dataset["train"]["text"], dataset["train"]["label"], test_size=0.2, random_state=42
    )

    # トークナイザの読み込み
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # データセットのトークン化
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_train_dataset = list(map(tokenize_function, [{"text": text} for text in train_texts]))
    tokenized_val_dataset = list(map(tokenize_function, [{"text": text} for text in val_texts]))

    # ラベルの追加
    for i, label in enumerate(train_labels):
        tokenized_train_dataset[i]["labels"] = label

    for i, label in enumerate(val_labels):
        tokenized_val_dataset[i]["labels"] = label

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
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    return model

# モデルの評価
def evaluate_model(model, dataloader, device):
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
    return accuracy

# ファインチューニング
def train_model(model, train_loader, val_loader, device, num_epochs=3):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

        # バリデーションの評価
        val_accuracy = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy}%")

    return train_losses, val_accuracies

# 結果の可視化
def plot_results(train_losses, val_accuracies, initial_accuracy):
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

# メイン関数
def main():
    # データセットの準備
    tokenized_train_dataset, tokenized_val_dataset, tokenizer = prepare_dataset()

    # カスタムデータセットの作成
    train_dataset = SimpleDataset(tokenized_train_dataset)
    val_dataset = SimpleDataset(tokenized_val_dataset)

    # DataLoaderの作成
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # モデルの準備
    model = prepare_model()

    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初期モデルの評価
    initial_accuracy = evaluate_model(model, val_loader, device)
    print(f"Initial Model Validation Accuracy: {initial_accuracy}%")

    # ファインチューニング
    train_losses, val_accuracies = train_model(model, train_loader, val_loader, device)

    # ファインチューニング後のモデルの評価
    final_accuracy = evaluate_model(model, val_loader, device)
    print(f"Fine-Tuned Model Validation Accuracy: {final_accuracy}%")

    # 結果の可視化
    plot_results(train_losses, val_accuracies, initial_accuracy)

if __name__ == "__main__":
    main()