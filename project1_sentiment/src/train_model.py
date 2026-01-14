import kagglehub
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

MODEL_PATH = "models/sentiment_model.pkl"
SAMPLE_SIZE = 5000

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

def train():
    path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    csv_file = os.path.join(path, "IMDB Dataset.csv")
    if not os.path.exists(csv_file):
        files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if files:
            csv_file = os.path.join(path, files[0])
    
    df = pd.read_csv(csv_file)
    if SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=42)

    df['cleaned_review'] = df['review'].apply(clean_text)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['label'], test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression())
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    cm = confusion_matrix(y_test, y_pred)
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('reports/confusion_matrix.png')

    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(pipeline, MODEL_PATH)

if __name__ == "__main__":
    train()
