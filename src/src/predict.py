import joblib
from preprocess import clean_text

def predict_sentiment(text):
    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return prediction[0]

if __name__ == "__main__":
    sample = input("Enter text: ")
    sentiment = predict_sentiment(sample)
    print(f"Predicted Sentiment: {sentiment}")
