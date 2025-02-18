import re
import string
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.stats import loguniform
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("hf://datasets/jhan21/amazon-food-reviews-dataset/Reviews.csv")
df = df[df['Score'] != 3]  # Remove neutral scores
df['target'] = df['Score'].apply(lambda x: 1 if x >= 4 else 0)

# Handle class imbalance
positive = df[df['target'] == 1]
negative = df[df['target'] == 0]
positive_undersampled = positive.sample(n=len(negative), random_state=42)
balanced_df = pd.concat([positive_undersampled, negative], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# Text preprocessing
nltk.download('stopwords')
nltk.download('punkt')
english_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w ]+', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join([ps.stem(word) for word in nltk.word_tokenize(text) if word not in english_stopwords])
    return text

balanced_df['cleaned_text'] = balanced_df['Text'].apply(preprocess_text)
balanced_df.to_csv("pre_processed_amazon_reviews.csv", index=False)

# Train-test split
train_df, temp_df = train_test_split(balanced_df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Check class distribution
for df in [train_df, val_df, test_df]:
    print(df['target'].value_counts())

# Define LSTM Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        numericalized = [self.vocab.get(token, self.vocab['<unk>']) for token in text.split()]
        padded = numericalized[:self.max_length] + [self.vocab['<pad>']] * (self.max_length - len(numericalized))
        return torch.tensor(padded), torch.tensor(label)

# Build vocabulary
def build_vocab(texts, max_vocab=20000):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {'<pad>': 0, '<unk>': 1}
    for idx, (word, count) in enumerate(counter.most_common(max_vocab)):
        vocab[word] = idx + 2
    return vocab

# Prepare data
texts = balanced_df['cleaned_text'].values
labels = balanced_df['target'].values
X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build vocabulary
vocab = build_vocab(X_train)
vocab_size = len(vocab)
max_length = 128

# Create datasets
train_dataset = TextDataset(X_train, y_train, vocab, max_length)
val_dataset = TextDataset(X_val, y_val, vocab, max_length)
test_dataset = TextDataset(X_test, y_test, vocab, max_length)

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        return self.fc(hidden.squeeze(0))

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMClassifier(vocab_size, embedding_dim=128, hidden_dim=256, output_dim=2).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Evaluation function
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            all_preds.extend(predictions.argmax(dim=1).cpu())
            all_labels.extend(labels.cpu())
    return total_loss / len(loader), all_preds, all_labels

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer)
    val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion)
    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}')
    print(classification_report(val_labels, val_preds))

# Final test evaluation
_, test_preds, test_labels = evaluate(model, test_loader, criterion)
print("Test Performance:")
print(classification_report(test_labels, test_preds))
