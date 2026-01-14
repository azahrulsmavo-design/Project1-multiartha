# Project 1: Sentiment Analysis (IMDB)

Simple text classification model to predict **Positive** or **Negative** sentiment from movie reviews.

## 📂 Structure
- `src/train_model.py`: Trains the model using TF-IDF + Logistic Regression.
- `src/predict_model.py`: Predicts sentiment from user input.
- `models/`: Stores the trained model (`sentiment_model.pkl`).

## 🚀 Usage

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Train
Downloads dataset & trains model (Accuracy ~84%).
```bash
python src/train_model.py
```

### 3. Predict
Interactive prediction:
```bash
python src/predict_model.py
```
Or single line:
```bash
python src/predict_model.py "I loved this movie!"
```
