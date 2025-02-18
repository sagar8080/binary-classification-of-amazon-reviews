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

# GPU check
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data split
texts = balanced_df['cleaned_text'].values
labels = balanced_df['target'].values
X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert to DataFrames
train_df = pd.DataFrame({'text': X_train, 'label': y_train})
val_df = pd.DataFrame({'text': X_val, 'label': y_val})
test_df = pd.DataFrame({'text': X_test, 'label': y_test})

# Tokenization
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
def tokenize_function(x):
    return tokenizer(x['text'], padding='max_length', truncation=True, max_length=128)

train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True, num_proc=4)
val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True, num_proc=4)
test_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True, num_proc=4)

# Model fine-tuning
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2).to(device)
training_args = TrainingArguments(
    output_dir='/results', num_train_epochs=4, per_device_train_batch_size=64, per_device_eval_batch_size=128,
    evaluation_strategy="steps", logging_steps=100, fp16=True, gradient_accumulation_steps=4,
    save_strategy="steps", save_steps=1000, learning_rate=5e-5, weight_decay=0.01,
    load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True, save_total_limit=1,
    dataloader_num_workers=8
)
trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
    compute_metrics=lambda pred: {"accuracy": accuracy_score(pred.label_ids, pred.predictions.argmax(-1)),
                                   "f1": f1_score(pred.label_ids, pred.predictions.argmax(-1), average="binary")},
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Training & evaluation
trainer.train()
trainer.evaluate(test_dataset)
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids
print("\nFinal Test Performance:")
print(classification_report(labels, preds))

# Training visualization
plt.figure(figsize=(6, 4))
roberta_history = trainer.state.log_history
train_loss = [x['loss'] for x in roberta_history if 'loss' in x]
val_loss = [x['eval_loss'] for x in roberta_history if 'eval_loss' in x]
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training Progress')
plt.ylabel('Loss')
plt.legend()
plt.show()
