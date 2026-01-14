# Project 2: Image Deployment (CIFAR-10)

End-to-end image classification pipeline (CNN) with API deployment.

## 📂 Structure
- `src/`: Training (`train.py`) & Evaluation (`evaluate.py`) scripts.
- `app/`: FastAPI application (`main.py`).
- `models/`: Trained model (`cifar10_cnn.pt`).
- `Dockerfile`: Container configuration.
- `docs/`: [Detailed Code Explanation](docs/STEP_BY_STEP_EXPLANATION.md).

## 🚀 Usage

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Train & Evaluate
Trains for 5 epochs and generates confusion matrix.
```bash
python src/train.py
python src/evaluate.py
```

### 3. Run API
Starts the server at `http://127.0.0.1:8000`.
```bash
uvicorn app.main:app --reload
```
Test via Swagger UI: `http://127.0.0.1:8000/docs`.

### 4. Docker (Optional)
```bash
docker build -t cifar-api .
docker run -p 8000:8000 cifar-api
```
