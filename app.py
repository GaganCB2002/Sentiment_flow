"""
SentimentFlow AI - Streamlit Web Application

Provides a browser UI for real-time sentiment analysis.
Users type or paste a review, click Predict, and see:
  - Sentiment label (Positive / Neutral / Negative) with emoji
  - Confidence percentage with a progress bar
"""

import streamlit as st
from predict import load_artifacts, predict_sentiment

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentFlow AI",
    page_icon="🎭",
    layout="centered"
)

# Minimal custom CSS for spacing and button sizing
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────
# LOAD MODEL (cached — runs once per session)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    load_artifacts()

try:
    load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ─────────────────────────────────────────────
# UI HEADER
# ─────────────────────────────────────────────
st.title("🎭 SentimentFlow AI")
st.markdown(
    "Enter a movie review, product review, or social media comment "
    "to predict its sentiment."
)

# ─────────────────────────────────────────────
# SESSION STATE (preserves text across reruns)
# ─────────────────────────────────────────────
if 'text' not in st.session_state:
    st.session_state.text = ""


def fill_text(text):
    st.session_state.text = text


# ─────────────────────────────────────────────
# TEXT INPUT AREA
# ─────────────────────────────────────────────
st.markdown("### ✍️ Enter your review")
text = st.text_area(
    "Write your review here:",
    value=st.session_state.text,
    height=150,
    placeholder="Type or paste your review here...",
    label_visibility="collapsed"
)

# ─────────────────────────────────────────────
# PREDICT & CLEAR BUTTONS
# ─────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.session_state.text = ""
    st.rerun()

# ─────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────
if predict_btn:
    # Input validation
    if not text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    elif len(text.strip()) < 5:
        st.warning("⚠️ Please enter at least 5 characters.")
    else:
        # Run the model
        with st.spinner("Analyzing sentiment..."):
            sentiment, confidence = predict_sentiment(text)

        # Map sentiment to emoji
        emoji_map = {
            "Positive": "😊",
            "Neutral": "😐",
            "Negative": "😞"
        }
        emoji = emoji_map.get(sentiment, "😐")

        # Display results
        st.markdown("## Results")
        col1, col2 = st.columns(2)

        with col1:
            st.success(f"### Prediction")
            st.markdown(f"# {sentiment} {emoji}")

        with col2:
            st.info(f"### Confidence")
            st.markdown(f"## {confidence:.2f}%")
            st.progress(confidence / 100)

        # ─────────────────────────────────────────
        # AUTO-SUGGESTIONS FOR NEXT STEPS
        # ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 💡 Suggested Next Steps")

        suggestions = {
            "Positive": [
                "✅ Keep delivering the same quality and experience.",
                "📢 Encourage the customer to leave a public review or testimonial.",
                "🎯 Use this positive feedback in marketing materials.",
                "📧 Send a thank-you email or loyalty discount.",
            ],
            "Neutral": [
                "📋 Ask the customer for more specific feedback on what could be improved.",
                "🔍 Follow up to understand if there was anything missing from their experience.",
                "📈 Consider adding a small perk or incentive to turn neutral into positive.",
                "🛠 Review the product/service touchpoints that received average ratings.",
            ],
            "Negative": [
                "📞 Respond immediately — apologize and acknowledge the issue.",
                "🔧 Offer a concrete solution: refund, replacement, or discount.",
                "📝 Document the complaint internally to prevent recurrence.",
                "⭐ Prioritize this feedback in your product/service improvement roadmap.",
            ]
        }

        tips = suggestions.get(sentiment, suggestions["Neutral"])
        for tip in tips:
            st.markdown(tip)
