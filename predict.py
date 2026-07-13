import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import clean_text

model = None
tokenizer = None
label_encoder = None

def load_artifacts():
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
    if model is None:
        load_artifacts()

    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')

    pred = model.predict(padded, verbose=0)[0]
    class_idx = int(np.argmax(pred))
    confidence = float(pred[class_idx] * 100)
    sentiment = label_encoder.inverse_transform([class_idx])[0]

    return sentiment, confidence
