# Run Project - SentimentFlow AI

## Prerequisites

- Python 3.13 or higher installed
- pip package manager

---

## Step 1: Install Dependencies

Open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

---

## Step 2: (Optional) Generate Dataset

The dataset is already provided. To regenerate fresh Twitter-style sentiment data:

```bash
python generate_dataset.py
```

This creates/updates `dataset/sentiment_dataset.csv` with ~3720 labeled reviews.

---

## Step 3: Train the Model

```bash
python train_model.py
```

What happens during training:
- Loads and cleans the CSV dataset
- Tokenizes text and pads sequences to length 100
- Splits data: 80% train, 20% test
- Builds and trains an RNN (Embedding → SimpleRNN → Dense → Softmax)
- Runs for up to 50 epochs with automatic learning rate reduction
- Saves the trained model and encoders to `model/` folder

---

## Step 4: Run the Web App

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## Step 5: Use the App

1. Type a review in the text area (minimum 5 characters)
2. Click **🔮 Predict** to analyze sentiment
3. View the result: sentiment label (😊 / 😐 / 😞) + confidence percentage
4. Click **🗑️ Clear** to reset
5. Or click a sample button to try pre-written reviews instantly

---

## Quick Test Without UI

Create a Python script or run in terminal:

```python
from predict import predict_sentiment

sentiment, confidence = predict_sentiment("I love this phone")
print(f"{sentiment} ({confidence:.2f}%)")
```

Expected output: `Positive (100.00%)`

---

## Project Structure

```
SentimentFlow AI/
├── app.py                  # Streamlit web UI
├── predict.py              # Load model & predict sentiment
├── train_model.py          # Train the RNN model
├── preprocess.py           # Text cleaning
├── generate_dataset.py     # Generate synthetic Twitter dataset
├── requirements.txt        # Python dependencies
├── README.md               # Full documentation
├── run_project.md          # This file
├── dataset/
│   └── sentiment_dataset.csv   # Labeled training reviews
└── model/
    ├── rnn_model.h5        # Trained neural network
    ├── tokenizer.pkl       # Word-to-index mapping
    └── label_encoder.pkl   # Label-to-integer mapping
```
