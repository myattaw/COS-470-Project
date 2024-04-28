import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np  # Add this import for handling NaN values

# Step 1: Load and preprocess data
data = pd.read_csv('tweet_data.csv')

# Remove rows with missing values in the 'class' column
data = data.dropna(subset=['class'])

# Step 2: Tokenization
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['message'])
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        labels = self.data.iloc[idx]['class']
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


# Split data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Step 3: Model Setup
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# Step 4: Training Loop

print(f"Cuda available: {torch.cuda.is_available()}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_dataset = CustomDataset(train_data, tokenizer, max_len=500)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(3):  # Adjust number of epochs as needed
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Step 5: Evaluation
test_dataset = CustomDataset(test_data, tokenizer, max_len=500)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

accuracy = total_correct / total_samples
print(f'Accuracy: {accuracy}')

# Step 6: Save the model
model.save_pretrained("roberta_model")
