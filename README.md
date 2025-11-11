# Fake_News_Detector
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Project-Complete-success)

📰 Fake News Detector

“Don’t believe everything you read — let AI tell you the truth.”

A Python-based machine learning application that detects whether a news article is real or fake using Natural Language Processing (NLP) and classification models.
Train, test, and evaluate your dataset — all from a simple, powerful terminal interface.

✨ **Features**

🔑 User Authentication – Register & login securely (passwords hashed with bcrypt)  
💵 Income & Expense Tracking – Add, view, and delete transactions with categories  
📊 Reports – Generate Monthly & Yearly summaries (Income, Expenses, Savings)  
🎯 Budgeting – Set monthly budgets & receive warnings if exceeded  
💾 Data Persistence – All data stored in SQLite database  
🗄 Backup & Restore – Create timestamped backups & restore anytime  


📂 **Project Structure**

│Fake_News_Detector
│── main.py               
│── dataset.csv          
│── model/                
│── graphs/               
│── README.md             
└── requirements.txt    
## 🛠 Installation & Setup

1️⃣ Clone the repository:

```bash
git clone https://github.com/DeveshreeBhakkad/Finance-Management-Application.git
cd finance_manager
```

2️⃣ Install dependencies
```bash
       pip install -r requirements.txt
```
Or install manually:
```bash
       pip install tensorflow nltk pandas numpy matplotlib scikit-learn
```
3️⃣ Run the project
```bash
       python main.py
```
🎮 Usage Guide

🧩 Training
     Automatically preprocesses data (cleaning, stemming, vectorizing).
     Trains ML models and saves them as .pkl files.

🔍 Prediction
     Enter a news headline or paragraph.
     The system predicts whether it’s REAL or FAKE.

📊 Model Evaluation
     Displays accuracy score.

Shows confusion matrix and precision-recall metrics.

🧩 How It Works

1. Data Loading – Loads dataset containing news articles and labels (real/fake).
2. Data Cleaning – Removes stopwords, punctuation, and performs tokenization + lemmatization.
3. Exploratory Data Analysis (EDA) – Visualizes fake vs real news counts, word counts, and subjects.
4. Model Building – Uses Embedding Layer + Bidirectional LSTM + Dense layers for classification.
5. Training – Runs for multiple epochs with validation split.
6. Evaluation – Displays accuracy, loss, confusion matrix, and performance metrics.



## 📊 Sample Visualizations

- 🟦 **Distribution of Fake vs Real News**  
- 🟩 **Number of Articles per Subject**  
- 🟨 **Word Count per Article**  
- 📉 **Training vs Validation Accuracy Graph**  

_All these visualizations are generated during runtime using Matplotlib._


## 🧮 Example Output

Epoch 1/5
acc: 0.9877 - loss: 0.0330 - val_acc: 0.9989 - val_loss: 0.0044

Epoch 2/5
acc: 0.9991 - loss: 0.0011 - val_acc: 0.9998 - val_loss: 0.0023

Final Model Accuracy: 80%


---

## 🔒 Notes

Make sure NLTK data packages are downloaded before running:

```python
import nltk  
nltk.download('punkt')  
nltk.download('stopwords')  
nltk.download('wordnet')  
nltk.download('punkt_tab')
⚠️ If TensorFlow shows oneDNN optimization warnings — it’s safe to ignore.

🌟 Future Improvements

🧾 Save model and use it for real-time fake news prediction

🌐 Deploy as a web app (Flask/Streamlit)

📄 Add dataset link or upload to Kaggle

📊 Add more visualizations and performance comparisons