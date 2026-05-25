# sentiment-classifier
# Sentiment Classifier

A machine learning-based sentiment analysis application for classifying Amazon product reviews as positive or negative.

## Features

- Text preprocessing and cleaning
- TF-IDF feature extraction
- Binary sentiment classification
- Machine learning models:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
- FastAPI backend
- Streamlit frontend
- REST API inference

---

## Project Structure

```bash
sentiment_classifier/
│
├── app/
│   ├── main.py
│   ├── frontend.py
│
├── models/
│   ├── sentiment_model.pkl
│   ├── vectorizer.pkl
│
├── notebooks/
│   ├── sentiment_analysis.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
Dataset

This project uses Amazon product reviews for sentiment classification.

Sentiment labels:

Positive → Rating >= 4
Negative → Rating <= 3

The original multiclass classification problem was converted into binary classification due to severe class imbalance in the neutral class.

Technologies Used
Python
Scikit-learn
Pandas
NumPy
NLTK
FastAPI
Streamlit
Joblib
Installation

Clone the repository:

git clone https://github.com/06112004mqh/sentiment-classifier.git
cd sentiment-classifier

Create virtual environment:

python -m venv sentimel_classifier_venv
source sentimel_classifier_venv/bin/activate

Install dependencies:

pip install -r requirements.txt
Run FastAPI Backend
uvicorn app.main:app --reload

API documentation:

http://127.0.0.1:8000/docs
Run Streamlit Frontend
streamlit run app/frontend.py

Streamlit app:

http://localhost:8501
Example Review
<img width="1420" height="647" alt="Screenshot from 2026-05-15 10-13-09" src="https://github.com/user-attachments/assets/dddb1e78-8f18-4a27-b2c5-f39692325e0b" />

Positive review:

This product exceeded my expectations and works perfectly.

Negative review:

The quality is terrible and I regret buying this product.
Future Improvements
BERT / PhoBERT integration
Aspect-based sentiment analysis
RAG-based review summarization
Docker deployment
CI/CD pipeline
Cloud deployment
Author

GitHub: https://github.com/06112004mqh
