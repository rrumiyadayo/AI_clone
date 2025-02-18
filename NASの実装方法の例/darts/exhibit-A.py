import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
from sklearn.model_selection import train_test_split

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

# ファインチューニング
def train_model(model, train_loader, val_loader, device, num_epochs=3):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

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
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}, Validation Accuracy: {accuracy}%")

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

    # ファインチューニング
    train_model(model, train_loader, val_loader, device)

if __name__ == "__main__":
    main()