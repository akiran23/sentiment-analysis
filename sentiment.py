import streamlit as st
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import tempfile
import os

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="MP3 Call Quality Analyzer", layout="centered")

st.title("📞 MP3 → Call Quality & Sentiment Analyzer")
st.write(
    "Upload an MP3 audio file in **English or Indian languages** "
    "to get transcription, sentiment, and **1–10 quality ratings**."
)

# ---------------- MODEL LOADING (CACHED) ----------------
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    speech_model = whisper.load_model("small").to(device)

    sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    sent_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
    sent_model = AutoModelForSequenceClassification.from_pretrained(
        sentiment_model_name
    ).to(device)

    return speech_model, sent_tokenizer, sent_model, device


# ---------------- SCORING LOGIC ----------------
sentiment_to_score = {
    0: 2,   # Very Negative
    1: 4,   # Negative
    2: 6,   # Neutral
    3: 8,   # Positive
    4: 10   # Very Positive
}

PRODUCT_KEYWORDS = [
    "price", "cost", "plan", "package", "features",
    "subscription", "demo", "trial", "details"
]

LEAD_KEYWORDS = [
    "interested", "call me", "contact me", "follow up",
    "sign up", "next steps", "email me"
]

NEGATIVE_WORDS = [
    "problem", "issue", "complaint", "bad",
    "worst", "refund", "angry"
]

def customer_tone_score(sentiment_idx, text):
    base = sentiment_to_score[sentiment_idx]
    penalty = sum(word in text.lower() for word in NEGATIVE_WORDS)
    return max(1, min(10, base - penalty))

def agent_tone_score(sentiment_idx):
    return sentiment_to_score[sentiment_idx]

def product_enquiry_score(text):
    matches = sum(word in text.lower() for word in PRODUCT_KEYWORDS)
    return min(10, matches * 2)

def lead_generation_score(text):
    matches = sum(word in text.lower() for word in LEAD_KEYWORDS)
    return min(10, matches * 3)

def overall_score(scores):
    return round(sum(scores) / len(scores), 1)


# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload MP3 file",
    type=["mp3"],
    help="Supported: English, Hindi, Tamil, Telugu, Malayalam, Kannada, Bengali, Marathi, etc."
)

# ---------------- RUN ONLY AFTER UPLOAD ----------------
if uploaded_file is not None:
    with st.spinner("Loading models..."):
        speech_model, sent_tokenizer, sent_model, device = load_models()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    try:
        # ---------------- TRANSCRIPTION ----------------
        st.subheader("🎧 Transcription")
        with st.spinner("Transcribing audio..."):
            result = speech_model.transcribe(audio_path)
            transcription = result.get("text", "").strip()

        if not transcription:
            st.error("No speech detected in the audio.")
        else:
            st.write(transcription)

            # ---------------- SENTIMENT ANALYSIS ----------------
            st.subheader("🧠 Sentiment Analysis")
            with st.spinner("Analyzing sentiment..."):
                inputs = sent_tokenizer(
                    transcription,
                    return_tensors="pt",
                    truncation=True,
                    padding=True
                ).to(device)

                with torch.no_grad():
                    outputs = sent_model(**inputs)

                probs = torch.nn.functional.softmax(
                    outputs.logits, dim=1
                ).cpu().numpy()[0]

            sentiment_idx = int(np.argmax(probs))

            labels = [
                "Very Negative",
                "Negative",
                "Neutral",
                "Positive",
                "Very Positive"
            ]

            st.success(f"**Sentiment:** {labels[sentiment_idx]}")

            # ---------------- BUSINESS RATINGS ----------------
            customer_tone = customer_tone_score(sentiment_idx, transcription)
            agent_tone = agent_tone_score(sentiment_idx)
            product_score = product_enquiry_score(transcription)
            lead_score = lead_generation_score(transcription)

            overall = overall_score([
                customer_tone,
                agent_tone,
                product_score,
                lead_score
            ])

            st.subheader("📊 Interaction Quality Ratings (1–10)")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Customer Tone", customer_tone)
                st.metric("Product Enquiry Quality", product_score)
            with col2:
                st.metric("Agent Tone", agent_tone)
                st.metric("Lead Generation Potential", lead_score)

            st.metric("⭐ Overall Call Quality", overall)

            with st.expander("📈 Sentiment Probability Breakdown"):
                for label, prob in zip(labels, probs):
                    st.write(f"{label}: {prob:.3f}")

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

else:
    st.info("👆 Please upload an MP3 file to start analysis.")
