import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DistilBertTokenizerFast, DistilBertForMaskedLM, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.inputs = []
        for text in texts:
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt'
            )
            self.inputs.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': encoding['input_ids'].squeeze()
            })
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx]

def load_texts(data_dir):
    file_paths = glob.glob(os.path.join(data_dir, "*.txt"))
    texts = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            texts.append(text)
    return texts

data_directory = 'train'
all_texts = load_texts(data_directory)
print(f"Total texts loaded: {len(all_texts)}")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
dataset = TextDataset(all_texts, tokenizer, max_length=512)
val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
model.to(device)
epochs = 5
learning_rate = 5e-5
epsilon = 1e-8
total_steps = len(train_loader) * epochs
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)
best_val_loss = float('inf')
for epoch in range(epochs):
    print(f"\n======== Epoch {epoch + 1} / {epochs} ========")
    print("Training...")
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Average training loss: {avg_train_loss:.4f}")
    print("Running Validation...")
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Average validation loss: {avg_val_loss:.4f}")
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        output_dir = f'./model_epoch_{epoch + 1}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
    else:
        print("No improvement in validation loss.")
print("\nTraining complete!")
final_output_dir = './final_model'
model.save_pretrained(final_output_dir)
tokenizer.save_pretrained(final_output_dir)
print(f"\nFinal model saved to {final_output_dir}")

