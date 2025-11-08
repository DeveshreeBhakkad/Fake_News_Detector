# ==========================================
# Fake News Detection – Windows Friendly
# ==========================================

# Install required packages before running:
# pip install tensorflow pandas numpy matplotlib seaborn nltk wordcloud gensim plotly scikit-learn

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px

# ==========================================
# Load the data
# ==========================================
df_true = pd.read_csv("True.csv")
df_fake = pd.read_csv("Fake.csv")

# Assign target labels
df_true['isfake'] = 1
df_fake['isfake'] = 0

# Merge datasets
df = pd.concat([df_true, df_fake]).reset_index(drop=True)

# Drop unnecessary column
if 'date' in df.columns:
    df.drop(columns=['date'], inplace=True)

# Combine title + text
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
    return [token for token in simple_preprocess(text)
            if token not in gensim.parsing.preprocessing.STOPWORDS 
            and len(token) > 3
            and token not in stop_words]

df['clean'] = df['original'].apply(preprocess)
df['clean_joined'] = df['clean'].apply(lambda x: " ".join(x))

# ==========================================
# Visualizations
# ==========================================

# Plot number of samples per subject
plt.figure(figsize=(8,8))
sns.countplot(y='subject', data=df)
plt.title("Number of samples per subject")
plt.show()

# WordCloud – Real News
plt.figure(figsize=(20,10))
wc_real = WordCloud(max_words=2000, width=1600, height=800, stopwords=stop_words).generate(" ".join(df[df.isfake==1]['clean_joined']))
plt.imshow(wc_real, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud – Real News")
plt.show()

# WordCloud – Fake News
plt.figure(figsize=(20,10))
wc_fake = WordCloud(max_words=2000, width=1600, height=800, stopwords=stop_words).generate(" ".join(df[df.isfake==0]['clean_joined']))
plt.imshow(wc_fake, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud – Fake News")
plt.show()

# Word count distribution histogram
word_counts = [len(word_tokenize(x)) for x in df['clean_joined']]
fig = px.histogram(x=word_counts, nbins=100, title="Distribution of Number of Words per Document")
fig.show()

# ==========================================
# Tokenization & Padding
# ==========================================
total_words = len(set([word for doc in df['clean'] for word in doc]))

x_train, x_test, y_train, y_test = train_test_split(df['clean_joined'], df['isfake'], test_size=0.2, random_state=42)

tokenizer = Tokenizer(num_words=total_words)
tokenizer.fit_on_texts(x_train)

train_seq = tokenizer.texts_to_sequences(x_train)
test_seq = tokenizer.texts_to_sequences(x_test)

maxlen = 200  # maximum words per document
padded_train = pad_sequences(train_seq, maxlen=maxlen, padding='post', truncating='post')
padded_test = pad_sequences(test_seq, maxlen=maxlen, padding='post', truncating='post')

y_train = np.array(y_train)

# ==========================================
# Build LSTM Model
# ==========================================
model = Sequential()
model.add(Embedding(total_words, output_dim=128))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()

# ==========================================
# Train the Model
# ==========================================
model.fit(padded_train, y_train, batch_size=64, validation_split=0.1, epochs=5)

# ==========================================
# Evaluate the Model
# ==========================================
pred = model.predict(padded_test)
prediction = [1 if p>0.5 else 0 for p in pred]

accuracy = accuracy_score(y_test, prediction)
print("Model Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, prediction)
plt.figure(figsize=(8,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================================
# Samples
# ==========================================
print("\nSample Original vs Cleaned Texts:")
for i in range(3):
    print(f"\nOriginal [{i}]: {df['original'][i]}")
    print(f"Cleaned  [{i}]: {df['clean_joined'][i]}")

