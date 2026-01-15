# ==========================================
# Fake News Detection – Full Pipeline
# ==========================================

# ----------- Imports -----------
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from wordcloud import WordCloud
import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional

import plotly.express as px

# ==========================================
# Load the data
# ==========================================
df_true = pd.read_csv("True.csv")
df_fake = pd.read_csv("Fake.csv")

df_true['isfake'] = 1
df_fake['isfake'] = 0

df = pd.concat([df_true, df_fake]).reset_index(drop=True)

if 'date' in df.columns:
    df.drop(columns=['date'], inplace=True)

df['original'] = df['title'] + ' ' + df['text']

# ==========================================
# Stopwords
# ==========================================
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# ==========================================
# Text preprocessing
# ==========================================
def preprocess(text):
    return [
        token for token in simple_preprocess(text)
        if token not in gensim.parsing.preprocessing.STOPWORDS
        and token not in stop_words
        and len(token) > 3
    ]

df['clean'] = df['original'].apply(preprocess)
df['clean_joined'] = df['clean'].apply(lambda x: " ".join(x))

# ==========================================
# Exploratory Data Analysis (EDA)
# ==========================================

plt.figure(figsize=(8,8))
sns.countplot(y='subject', data=df)
plt.title("Number of samples per subject")
plt.show()

wc_real = WordCloud(width=1600, height=800, stopwords=stop_words)\
    .generate(" ".join(df[df.isfake==1]['clean_joined']))
plt.imshow(wc_real); plt.axis("off"); plt.title("Real News WordCloud"); plt.show()

wc_fake = WordCloud(width=1600, height=800, stopwords=stop_words)\
    .generate(" ".join(df[df.isfake==0]['clean_joined']))
plt.imshow(wc_fake); plt.axis("off"); plt.title("Fake News WordCloud"); plt.show()

word_counts = [len(word_tokenize(x)) for x in df['clean_joined']]
px.histogram(x=word_counts, nbins=100, title="Word Count Distribution").show()

# ==========================================
# Phase 1 – Baseline Models (TF-IDF + ML)
# ==========================================

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_joined'])
y = df['isfake']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression (FINAL MODEL)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\nLogistic Regression Results:")
print(classification_report(y_test, lr_pred))

# Naive Bayes (Comparison)
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

print("\nNaive Bayes Results:")
print(classification_report(y_test, nb_pred))

# ==========================================
# Save Final Model
# ==========================================

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("logistic_regression_model.pkl", "wb") as f:
    pickle.dump(lr, f)

print("\nFinal model and vectorizer saved.")

# ==========================================
# Phase 1 – Deep Learning Model (BiLSTM – Experimental)
# ==========================================

x_train_dl, x_test_dl, y_train_dl, y_test_dl = train_test_split(
    df['clean_joined'], df['isfake'], test_size=0.2, random_state=42
)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train_dl)

train_seq = tokenizer.texts_to_sequences(x_train_dl)
test_seq = tokenizer.texts_to_sequences(x_test_dl)

maxlen = 200
padded_train = pad_sequences(train_seq, maxlen=maxlen, padding='post')
padded_test = pad_sequences(test_seq, maxlen=maxlen, padding='post')

total_words = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(total_words, 128))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    padded_train,
    np.asarray(y_train_dl),
    batch_size=64,
    validation_split=0.1,
    epochs=2
)

# Overfitting Analysis
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Training vs Validation Loss (BiLSTM)")
plt.show()

# LSTM Evaluation
pred_dl = model.predict(padded_test)
pred_dl = [1 if p > 0.5 else 0 for p in pred_dl]

print("\nBiLSTM Accuracy:", accuracy_score(y_test_dl, pred_dl))

cm = confusion_matrix(y_test_dl, pred_dl)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("BiLSTM Confusion Matrix")
plt.show()

# ==========================================
# Phase 2 – Real-World Testing
# ==========================================

with open("tfidf_vectorizer.pkl", "rb") as f:
    loaded_tfidf = pickle.load(f)

with open("logistic_regression_model.pkl", "rb") as f:
    loaded_lr = pickle.load(f)

real_world_news = [
    "Government announces new education policy",
    "You won’t believe what this actor did today",
    "Scientists confirm discovery of water on Mars"
]

vec = loaded_tfidf.transform(real_world_news)
preds = loaded_lr.predict(vec)
probs = loaded_lr.predict_proba(vec)

for i, text in enumerate(real_world_news):
    label = "REAL" if preds[i] == 1 else "FAKE"
    confidence = max(probs[i]) * 100
    print(f"\n{text}\nPrediction: {label} ({confidence:.2f}%)")

# ==========================================
# Phase 2 – Explainability
# ==========================================

feature_names = loaded_tfidf.get_feature_names_out()
coefficients = loaded_lr.coef_[0]

importance = pd.DataFrame({
    "word": feature_names,
    "weight": coefficients
})

print("\nTop FAKE words:")
print(importance.sort_values(by="weight").head(10))

print("\nTop REAL words:")
print(importance.sort_values(by="weight", ascending=False).head(10))
