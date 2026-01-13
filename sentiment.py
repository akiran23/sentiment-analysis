import streamlit as st
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import tempfile
import os

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="MP3 Sentiment Analyzer", layout="centered")

st.title("🌐 MP3 → Text Sentiment Analyzer")
st.write(
    "Upload an MP3 audio file in **English or any Indian language** "
    "to get transcription and sentiment analysis."
)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload MP3 file",
    type=["mp3"],
    help="Supported: English, Hindi, Tamil, Telugu, Malayalam, Kannada, Bengali, Marathi, etc."
)

# ---------------- RUN ONLY AFTER UPLOAD ----------------
if uploaded_file is not None:
    with st.spinner("Loading models..."):
        # Load models lazily AFTER upload
        speech_model = whisper.load_model("small")

        sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        sent_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        sent_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

    # Save MP3 to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

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
            )
            outputs = sent_model(**inputs)

            scores = outputs.logits.detach().cpu().numpy()[0]
            probs = torch.nn.functional.softmax(torch.tensor(scores), dim=0).numpy()

        labels = ["⭐️ Very Negative", "⭐️⭐️ Negative", "⭐️⭐️⭐️ Neutral", "⭐️⭐️⭐️⭐️ Positive", "⭐️⭐️⭐️⭐️⭐️ Very Positive"]

        sentiment = labels[np.argmax(probs)]
        confidence = float(np.max(probs))

        st.success(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.2f}")

        with st.expander("📊 Sentiment Probability Breakdown"):
            for label, prob in zip(labels, probs):
                st.write(f"{label}: {prob:.3f}")

    # Cleanup
    os.remove(audio_path)

else:
    st.info("👆 Please upload an MP3 file to start analysis.")
