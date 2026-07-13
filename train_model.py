"""
SentimentFlow AI - Model Training Script

Pipeline:
  1. Load CSV dataset
  2. Clean and tokenize text
  3. Pad sequences to fixed length
  4. Encode labels
  5. Train/test split (80/20)
  6. Build RNN model (Embedding → SimpleRNN → Dense → Softmax)
  7. Train for up to 50 epochs with ReduceLROnPlateau
  8. Save model + tokenizer + label_encoder to model/
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from preprocess import clean_text

# ─────────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv('dataset/sentiment_dataset.csv')
print(f"Dataset shape: {df.shape}")

# Support both old (text/sentiment) and new (Review/Sentiment) column names
col_text = 'Review' if 'Review' in df.columns else 'text'
col_sent = 'Sentiment' if 'Sentiment' in df.columns else 'sentiment'
print(f"Class distribution:\n{df[col_sent].value_counts()}")

# ─────────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ─────────────────────────────────────────────────
print("\nCleaning text...")
df['cleaned_text'] = df[col_text].apply(clean_text)

# ─────────────────────────────────────────────────
# 3. TOKENIZATION
# ─────────────────────────────────────────────────
# Convert each word to a unique integer index
# num_words = 10000: keep only the 10k most frequent words
# oov_token: out-of-vocabulary words get index 1
print("Tokenizing text...")
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['cleaned_text'])
sequences = tokenizer.texts_to_sequences(df['cleaned_text'])

# ─────────────────────────────────────────────────
# 4. PADDING
# ─────────────────────────────────────────────────
# All sequences must have the same length for batch processing
# Shorter sequences get zeros appended (post-padding)
# Longer sequences get truncated from the end
max_len = 100
print(f"Padding sequences to max length: {max_len}")
padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# ─────────────────────────────────────────────────
# 5. LABEL ENCODING
# ─────────────────────────────────────────────────
# Convert string labels ("Positive", "Negative", "Neutral") to integers (0, 1, 2)
print("Encoding labels...")
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df[col_sent])
print(f"Classes: {label_encoder.classes_}")

# ─────────────────────────────────────────────────
# 6. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────
# 80% training, 20% testing with stratification (same class ratio in both splits)
X_train, X_test, y_train, y_test = train_test_split(
    padded, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {vocab_size}")

# ─────────────────────────────────────────────────
# 7. MODEL ARCHITECTURE
# ─────────────────────────────────────────────────
# Embedding: maps each word index to a dense 100-dimensional vector
# SimpleRNN: 128 units, processes sequence and outputs final hidden state
# Dense(64): fully connected hidden layer with ReLU activation
# Dense(3): output layer with Softmax → probability distribution over 3 classes
model = Sequential([
    Embedding(vocab_size, 100),
    SimpleRNN(128, return_sequences=False),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.build((None, max_len))
model.summary()

# ─────────────────────────────────────────────────
# 8. TRAINING
# ─────────────────────────────────────────────────
# ReduceLROnPlateau: halves learning rate when validation loss stops improving
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
]

print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Evaluate on held-out test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# ─────────────────────────────────────────────────
# 9. SAVE ARTIFACTS
# ─────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)

model.save('model/rnn_model.h5')
print("Model saved to model/rnn_model.h5")

with open('model/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("Tokenizer saved to model/tokenizer.pkl")

with open('model/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("Label encoder saved to model/label_encoder.pkl")

# ─────────────────────────────────────────────────
# 10. TRAINING HISTORY PLOT
# ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print("Training history plot saved to training_history.png")

print("\n✅ Training complete!")
