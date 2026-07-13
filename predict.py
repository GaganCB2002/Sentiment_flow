"""
Prediction module for SentimentFlow AI.
Loads the trained model and provides the predict_sentiment() function.
"""

import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import clean_text

# Global references (populated once by load_artifacts)
model = None
tokenizer = None
label_encoder = None


def load_artifacts():
    """
    Load the trained model, tokenizer, and label encoder from disk.
    Called once on app startup (cached by Streamlit).
    Raises FileNotFoundError if model/rnn_model.h5 is missing.
    """
    global model, tokenizer, label_encoder
    model_path = 'model/rnn_model.h5'
    tokenizer_path = 'model/tokenizer.pkl'
    encoder_path = 'model/label_encoder.pkl'

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model not found. Please run 'python train_model.py' first."
        )

    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)


def predict_sentiment(text):
    """
    Predict sentiment for a single review string.

    Pipeline: clean → tokenize → pad → model predict → decode label
    Returns (sentiment_label: str, confidence_percentage: float).
    """
    if model is None:
        load_artifacts()

    # Step 1: Normalize the raw text
    cleaned = clean_text(text)

    # Step 2: Convert words to integer sequences using the fitted tokenizer
    seq = tokenizer.texts_to_sequences([cleaned])

    # Step 3: Pad/truncate to the fixed length the model expects (100 tokens)
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')

    # Step 4: Run the model — output is a 3-class probability distribution
    pred = model.predict(padded, verbose=0)[0]

    # Step 5: Pick the class with highest probability
    class_idx = int(np.argmax(pred))
    confidence = float(pred[class_idx] * 100)

    # Step 6: Convert integer class back to human-readable label
    sentiment = label_encoder.inverse_transform([class_idx])[0]

    return sentiment, confidence
