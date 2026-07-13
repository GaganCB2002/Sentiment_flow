"""
Text preprocessing utilities for SentimentFlow AI.
Cleans raw review text before tokenization and model prediction.
"""

import re


def clean_text(text):
    """
    Normalize a review string for model input.

    Steps:
    1. Lowercase — ensures case-insensitive matching
    2. Strip punctuation/digits — keeps only a-z and whitespace
    3. Collapse whitespace — replaces multiple spaces with one

    Note: Stopwords are NOT removed because short reviews lose too much signal.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
