import joblib
import sys
import os

MODEL_PATH = "models/sentiment_model.pkl"

def predict(text):
    if not os.path.exists(MODEL_PATH):
        return

    pipeline = joblib.load(MODEL_PATH)
    prediction = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0]
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = proba[1] if prediction == 1 else proba[0]
    
    print(f"Input: \"{text}\"")
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.2%}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict(" ".join(sys.argv[1:]))
    else:
        while True:
            text = input("\n>> ")
            if text.lower() in ['exit', 'quit']:
                break
            predict(text)
