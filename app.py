import streamlit as st
from predict import load_artifacts, predict_sentiment

st.set_page_config(
    page_title="SentimentFlow AI",
    page_icon="🎭",
    layout="centered"
)

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


@st.cache_resource
def load_model():
    load_artifacts()


try:
    load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.title("🎭 SentimentFlow AI")
st.markdown(
    "Enter a movie review, product review, or social media comment "
    "to predict its sentiment."
)

if 'text' not in st.session_state:
    st.session_state.text = ""


def fill_text(text):
    st.session_state.text = text


st.markdown("### ✍️ Enter your review")
text = st.text_area(
    "Write your review here:",
    value=st.session_state.text,
    height=150,
    placeholder="Type or paste your review here...",
    label_visibility="collapsed"
)

col1, col2 = st.columns(2)
with col1:
    predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.session_state.text = ""
    st.rerun()

st.markdown("---")
st.markdown("### 📝 Try a sample review")

samples = [
    "This product is amazing and I love it.",
    "The product arrived yesterday.",
    "This is the worst product I have ever used."
]
sample_labels = ["😊 Positive", "😐 Neutral", "😞 Negative"]

cols = st.columns(3)
for i, col in enumerate(cols):
    with col:
        if st.button(sample_labels[i], use_container_width=True):
            fill_text(samples[i])
            st.rerun()

st.markdown("---")

if predict_btn:
    if not text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    elif len(text.strip()) < 5:
        st.warning("⚠️ Please enter at least 5 characters.")
    else:
        with st.spinner("Analyzing sentiment..."):
            sentiment, confidence = predict_sentiment(text)

        emoji_map = {
            "Positive": "😊",
            "Neutral": "😐",
            "Negative": "😞"
        }
        emoji = emoji_map.get(sentiment, "😐")

        st.markdown("## Results")
        col1, col2 = st.columns(2)

        with col1:
            st.success(f"### Prediction")
            st.markdown(f"# {sentiment} {emoji}")

        with col2:
            st.info(f"### Confidence")
            if confidence > 99:
                conf_display = f"## {confidence:.2f}%"
            else:
                conf_display = f"## {confidence:.2f}%"
            st.markdown(conf_display)
            st.progress(confidence / 100)
