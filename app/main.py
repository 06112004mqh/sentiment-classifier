from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load('./models/sentiment_model.pkl')
vectorizer = joblib.load('./models/vectorizer.pkl')


class Review(BaseModel):
    text: str


@app.get('/')
def home():
    return {'message': 'Sentiment API Running'}


@app.post('/predict')
def predict(review: Review):

    text_vec = vectorizer.transform([review.text])

    prediction = model.predict(text_vec)[0]

    return {
        'review': review.text,
        'sentiment': prediction
    }