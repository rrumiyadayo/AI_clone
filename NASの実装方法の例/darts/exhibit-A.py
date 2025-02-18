import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# データセットの準備
def prepare_dataset():
    # データセットの読み込み
    dataset = load_dataset("imdb")
    
    # データセットの分割
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        dataset["train"]["text"], dataset["train"]["label"], test_size=0.2, random_state=42
    )
    
    # トークナイザの読み込み
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # データセットのトークン化
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_train_dataset = dataset["train"].map(tokenize_function, batched=True)
    tokenized_val_dataset = dataset["validation"].map(tokenize_function, batched=True)
    
    return tokenized_train_dataset, tokenized_val_dataset, tokenizer

# カスタムデータセットの作成
class SimpleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# モデルの準備
def prepare_model():
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    return model

# ファインチューニング
def fine_tune_model(model, train_loader, val_loader, num_epochs=3, learning_rate=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
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
                outputs = model(input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}, Validation Accuracy: {accuracy}%")

# メイン関数
def main():
    # データセットの準備
    tokenized_train_dataset, tokenized_val_dataset, tokenizer = prepare_dataset()
    
    # データローダーの作成
    train_dataset = SimpleDataset(tokenized_train_dataset["text"], tokenized_train_dataset["label"], tokenizer)
    val_dataset = SimpleDataset(tokenized_val_dataset["text"], tokenized_val_dataset["label"], tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # モデルの準備
    model = prepare_model()
    
    # ファインチューニング
    fine_tune_model(model, train_loader, val_loader)

if __name__ == "__main__":
    main()