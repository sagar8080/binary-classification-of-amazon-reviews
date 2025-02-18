import re
import string
from collections import Counter

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.stats import loguniform
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report


# Note: Can optionally use this if required
# df = pd.read_csv("hf://datasets/jhan21/amazon-food-reviews-dataset/Reviews.csv")

df = pd.read_csv("hf://datasets/jhan21/amazon-food-reviews-dataset/Reviews.csv")

df.shape

df.columns

df.Score.value_counts()

df = df[df['Score'] != 3]

df.shape

df.Score.value_counts()

"""## Convert the target variable to 0 and 1: 0 for Negative Reviews (1-2) and Positive Reviews (4-5)"""

df['target'] = df['Score'].apply(lambda x: 1 if x >= 4 else 0)
print(df['target'].value_counts())

"""## Handle Class Imbalance"""

# Split by class
positive = df[df['target'] == 1]
negative = df[df['target'] == 0]
positive_undersampled = positive.sample(n=len(negative), random_state=42)

# Combine and Shuffle
balanced_df = pd.concat([positive_undersampled, negative], axis=0)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
print(balanced_df['target'].value_counts())

"""## Preprocessing the Data"""

nltk.download('stopwords')
nltk.download('punkt_tab')

english_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^\w ]+','', text)
    text = re.sub(r'(http|https)?://\S+|www\.\S+','', text)
    text = ''.join(word for word in text if ord(word) < 128)
    text = text.translate(str.maketrans('','',string.punctuation))
    text = re.sub(r'[\d]+','', text)
    text = ' '.join(word for word in text.split() if len(word)>1)
    text = ' '.join(text.split())
    # stopword and punct removal
    text = ' '.join([i for i in nltk.word_tokenize(text) if i not in
    english_stopwords and i not in string.punctuation])
    # removal of anything other than English letters
    text = re.sub('[^a-z]+', ' ', text)
    text = ' '.join([ps.stem(i) for i in nltk.word_tokenize(text)]) #stemming
    return text

balanced_df['cleaned_text'] = balanced_df['Text'].apply(lambda x: preprocess_text(x))

# Optional: Save the dataset
balanced_df.to_csv("pre_processed_amazon_reviews.csv", index=False)

balanced_df['cleaned_text'].head()

"""## Do a Train Test Split: 70% Training, 15% Val and 15% Test"""

from sklearn.model_selection import train_test_split
train_df, temp_df = train_test_split(balanced_df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

for x in [train_df, test_df, val_df]:
    print(x['target'].value_counts())

"""## Implement SVM"""



# SVM Pipeline
svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('clf', SVC(class_weight='balanced', kernel='rbf'))  # Default to RBF kernel
])

# Hyperparameter tuning
param_grid = {
    'clf__C': [0.1, 1, 10],
    'tfidf__ngram_range': [(1,1), (2,2)]
}

gs_svm = GridSearchCV(svm_pipeline, param_grid, cv=3, n_jobs=-1)
gs_svm.fit(train_df['cleaned_text'], train_df['target'])

# Evaluation
svm_preds = gs_svm.predict(val_df['cleaned_text'])
print(classification_report(val_df['target'], svm_preds))