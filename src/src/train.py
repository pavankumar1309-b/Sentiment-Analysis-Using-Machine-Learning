import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from preprocess import preprocess_data, vectorize_text

# Load your dataset
df = pd.read_csv("data/sentiment.csv")  # Assumes columns: "text", "label"

# Preprocess
df = preprocess_data(df)
X, vectorizer = vectorize_text(df["text"])
y = df["label"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
