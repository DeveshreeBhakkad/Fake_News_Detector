
# 📰 Fake News Detection using NLP & Deep Learning

> “Don’t believe everything you read — let AI tell you the truth.”

Detecting fake news articles using Natural Language Processing (NLP) and Deep Learning techniques.

This project focuses on classifying news articles as **REAL** or **FAKE** based on their textual content. It demonstrates an end-to-end machine learning workflow including data preprocessing, exploratory data analysis, model building, and evaluation.

---

## 📌 Problem Statement

Fake news spreads rapidly through digital platforms and can significantly impact public opinion, politics, and society. Manual verification is slow and error-prone.

This project aims to build an automated system that:
- Analyzes news article text
- Learns linguistic patterns of fake vs real news
- Predicts whether a given article is **REAL** or **FAKE**

---

## 🎯 Project Objectives

- Perform text cleaning and preprocessing using NLP techniques  
- Explore and visualize patterns in real and fake news data  
- Build a Deep Learning model using **Bidirectional LSTM**  
- Evaluate model performance using standard ML metrics  
- Identify limitations and scope for future improvements  

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
4. Feature engineering and sequence padding  
5. Model building using Bidirectional LSTM  
6. Model training with validation split  
7. Model evaluation using test data  

---

## 🧠 Model Architecture

- Embedding Layer  
- Bidirectional LSTM Layer  
- Dense Fully Connected Layers  
- Sigmoid Activation for binary classification  

---

## 📊 Results

- The model achieves high training accuracy  
- Test accuracy is lower, indicating possible overfitting  
- Confusion matrix is used to analyze classification performance  

> Note: Accuracy alone is not sufficient for real-world fake news detection. Additional evaluation metrics are required.

---

## ✅ Final Model Selection

Three models were evaluated for the fake news classification task:

- **Multinomial Naive Bayes**
- **TF-IDF + Logistic Regression**
- **Bidirectional LSTM (Deep Learning)**

Although the BiLSTM model achieved very low training loss, it showed signs of overfitting as validation loss increased across epochs.

The **TF-IDF + Logistic Regression** model achieved the best balance of performance and generalization, with approximately **99% accuracy** and strong precision–recall scores on unseen test data.

Due to its:
- Superior generalization
- Lower computational complexity
- Faster training and inference
- Better interpretability

**TF-IDF + Logistic Regression was selected as the final production-ready model.**

The BiLSTM model is retained in this project for experimental comparison and learning purposes.

---

## ⚠️ Limitations

- Dataset bias towards specific news domains  
- Limited generalization to social media or short text  
- Overfitting due to model complexity  
- No real-time or production deployment yet  

---

## 🚀 Future Improvements

- Add baseline ML models (Logistic Regression, Naive Bayes)  
- Compare classical ML vs deep learning approaches  
- Use Precision, Recall, and F1-Score for evaluation  
- Implement model explainability techniques  
- Deploy the model as a web application  
- Test performance on real-world news data  

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

👩‍💻 Author

Deveshree Bhakkad
Final Year AIML Student
Interested in Machine Learning, NLP, and Applied AI

⭐ Conclusion

This project represents a step toward building industry-ready NLP systems.
Future versions will focus on improving generalization, explainability, and deployment readiness.


---

