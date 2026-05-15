import streamlit as st
import requests

st.title("Sentiment Analysis")

review = st.text_area("Enter review")

if st.button("Predict"):

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={
            "text": review
        }
    )

    result = response.json()

    st.success(result["sentiment"])