# SentimentFlow AI - Quick Start Guide

## Prerequisites

- Python 3.13+ installed
- pip (Python package manager)

## Step 1: Install Dependencies

Open a terminal in the project root and run:

```bash
pip install -r requirements.txt
```

## Step 2: Generate Dataset (Optional)

The dataset is already provided. To regenerate:

```bash
python generate_dataset.py
```

## Step 3: Train the Model

```bash
python train_model.py
```

This will:
- Load the CSV dataset from `dataset/sentiment_dataset.csv`
- Preprocess and tokenize the text
- Train the RNN model for up to 50 epochs
- Save the trained model and encoders to `model/`

## Step 4: Run the App

```bash
streamlit run app.py
```

Then open your browser to the URL shown (usually `http://localhost:8501`).

## Quick Test (No UI)

```python
from predict import predict_sentiment

sentiment, confidence = predict_sentiment("This product is amazing!")
print(f"{sentiment} ({confidence:.2f}%)")
```

## File Overview

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web UI |
| `predict.py` | Load model & predict sentiment |
| `train_model.py` | Train the RNN model |
| `preprocess.py` | Text cleaning utilities |
| `generate_dataset.py` | Generate synthetic training data |
| `requirements.txt` | Python package dependencies |
| `model/rnn_model.h5` | Trained neural network |
| `model/tokenizer.pkl` | Fitted Keras Tokenizer |
| `model/label_encoder.pkl` | Fitted LabelEncoder |
| `dataset/sentiment_dataset.csv` | Labeled training reviews |
