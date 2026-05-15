import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from app.utils import clean_text

# load data
reviews = pd.read_csv('data/Amazon_Reviews.csv', engine = 'python')

reviews = reviews[['Review Text', 'Rating']]
reviews.dropna(inplace=True)

# convert rating
reviews['Rating'] = reviews['Rating'].str.extract(r'(\d+)')[0].astype(int)

# sentiment label

def sentiment_label(rating):
    if rating >= 3:
        return 'positive'
    else:
        return 'negative'


reviews['sentiment'] = reviews['Rating'].apply(sentiment_label)

# preprocess
reviews['Review Text'] = reviews['Review Text'].apply(clean_text)

X = reviews['Review Text']
y = reviews['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# vectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# predict
pred = model.predict(X_test_vec)

print(classification_report(y_test, pred))

# save
joblib.dump(model, 'models/sentiment_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print('Saved model successfully')