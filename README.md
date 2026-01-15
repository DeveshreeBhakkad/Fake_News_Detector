# 📰 Fake News Detection using NLP & Deep Learning

> “Don’t believe everything you read — let AI assist in identifying misinformation.”

This project focuses on classifying news articles as **REAL** or **FAKE** using Natural Language Processing (NLP) and Machine Learning techniques.  
It demonstrates an **end-to-end ML workflow**, including preprocessing, exploratory analysis, model comparison, explainability, and production-aware practices.

---

## 📌 Problem Statement

Fake news spreads rapidly through digital platforms and can significantly impact public opinion, politics, and society.  
Manual verification of news content is slow, subjective, and not scalable.

This project aims to build an automated system that:
- Analyzes news article text  
- Learns linguistic patterns of fake vs real news  
- Predicts whether a given article is **REAL** or **FAKE**  

---

## 🎯 Project Objectives

- Perform text cleaning and preprocessing using NLP techniques  
- Explore and visualize patterns in real and fake news data  
- Compare classical machine learning and deep learning models  
- Select the most suitable model based on performance and generalization  
- Analyze real-world behavior and limitations of the model  

---

## 📂 Dataset Description

The dataset consists of two CSV files:

- **True.csv** – Contains real news articles  
- **Fake.csv** – Contains fake news articles  

Each record includes:
- `title` – News headline  
- `text` – Full article content  
- `subject` – Topic/category of the article  

A binary label is created:
- `1` → Real News  
- `0` → Fake News  

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:**
  - NLP: `nltk`, `gensim`, `wordcloud`
  - Data Processing: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`, `plotly`
  - Machine Learning: `scikit-learn`
  - Deep Learning: `TensorFlow (Keras)`

---

## 🔄 Project Workflow

1. Data loading and labeling  
2. Text preprocessing (tokenization, stopword removal, cleaning)  
3. Exploratory Data Analysis (EDA)  
4. Feature extraction using TF-IDF  
5. Model comparison:
   - Naive Bayes
   - Logistic Regression
   - Bidirectional LSTM  
6. Model evaluation and overfitting analysis  
7. Final model selection  
8. Model saving, loading, and real-world testing  

---

## 🧠 Models Used

### Classical Machine Learning
- **TF-IDF + Logistic Regression**
- **TF-IDF + Multinomial Naive Bayes**

### Deep Learning
- **Bidirectional LSTM**
  - Embedding Layer  
  - Bidirectional LSTM Layer  
  - Dense Layers  
  - Sigmoid activation for binary classification  

---

## 📊 Results

- **TF-IDF + Logistic Regression** achieved approximately **99% accuracy** with strong precision and recall.
- **Naive Bayes** achieved around **93% accuracy**.
- **Bidirectional LSTM** achieved very low training loss but showed **overfitting**, where validation loss increased across epochs.

These results indicate that classical ML models generalized better than deep learning for this dataset.

---

## ✅ Final Model Selection

Three models were evaluated for the fake news classification task:

- **Multinomial Naive Bayes**
- **TF-IDF + Logistic Regression**
- **Bidirectional LSTM (Deep Learning)**

Although the BiLSTM model performed well on training data, it showed signs of overfitting during validation.

The **TF-IDF + Logistic Regression** model achieved the best balance of performance, simplicity, and generalization.

Due to its:
- Strong generalization performance  
- Lower computational complexity  
- Faster training and inference  
- High interpretability  

**TF-IDF + Logistic Regression was selected as the final model.**

The BiLSTM model is retained for experimental comparison and learning purposes.

---

## 🌍 Real-World Behavior & Observations

When tested on manually written real-world news-style sentences, the model showed a tendency to classify many neutral or breaking-news headlines as **FAKE**.

This behavior highlights:
- Dataset bias toward sensational language  
- Conservative predictions in uncertain cases  

In practical applications, such a model should be used as a **supporting tool for preliminary screening**, not as a final decision-maker.

---

## 🔍 Model Explainability

Since the final model is Logistic Regression, word-level coefficients were analyzed to understand predictions.

- Words such as **video, watch, featured** were strongly associated with fake news.
- Words such as **reuters, said, monday** were strongly associated with real news.

This improves transparency and helps build trust in the model’s decisions.

---

## ⚠️ Limitations

- Dataset bias toward specific news sources and writing styles  
- Limited generalization to social media or informal text  
- No continuous retraining or live data integration  
- Not suitable for fully autonomous real-world deployment  

---

## 🚀 Future Improvements

- Train on more diverse and recent datasets  
- Improve balance between fake and real samples  
- Add human-in-the-loop verification  
- Deploy as a lightweight web application (Streamlit / Flask)  
- Extend explainability and monitoring for real-world use  

---

## ▶️ How to Run the Project

### 1. Clone the repository
```bash
git clone <repository-url>
cd Fake_News_Detection
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the project
```bash
python main.py
```
---

## 👩‍💻 Author

Deveshree Bhakkad
Final Year AIML Student
Interested in Machine Learning, NLP, and Applied AI Systems

---

## ⭐ Conclusion

This project demonstrates a responsible and explainable approach to fake news detection by comparing multiple models and selecting the most appropriate one based on evidence.

Rather than overclaiming real-world deployment, the project focuses on sound ML practices, transparency, and practical limitations, making it recruiter-friendly and interview-safe.
