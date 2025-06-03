import streamlit as st
from src.predict import predict_sentiment

st.title("🧠 Sentiment Analysis App")
user_input = st.text_area("Enter your text here:")

if st.button("Analyze"):
    result = predict_sentiment(user_input)
    st.success(f"Predicted Sentiment: **{result}**")
