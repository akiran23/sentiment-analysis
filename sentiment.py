import streamlit as st
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# --- Load Models (once) ---
@st.cache(allow_output_mutation=True)
def load_speech_model():
    return whisper.load_model("small")  # multilingual

@st.cache(allow_output_mutation=True)
def load_sentiment_model():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tok = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tok, mod

speech_model = load_speech_model()
sent_tok, sent_model = load_sentiment_model()

# --- Streamlit UI ---
st.title("🌐 MP3 → Text Sentiment Analyzer")
st.write("Upload an MP3 in English or any Indian language and get sentiment.")

mp3_file = st.file_uploader("Upload MP3 Audio File", type=["mp3"])

if mp3_file:
    # Save temporary mp3
    with open("audio.mp3", "wb") as f:
        f.write(mp3_file.getbuffer())
        
    st.info("Transcribing audio…")
    # Whisper transcription
    result = speech_model.transcribe("audio.mp3")
    text = result.get("text", "").strip()
    
    if not text:
        st.error("No speech detected or could not transcribe.")
    else:
        st.subheader("📜 Transcription")
        st.write(text)
        
        # Sentiment analysis
        st.info("Analyzing sentiment…")
        inputs = sent_tok(text, return_tensors="pt", padding=True, truncation=True)
        outputs = sent_model(**inputs)
        
        scores = outputs.logits.detach().cpu().numpy()[0]
        # For nlptown: 1–5 stars
        probs = torch.nn.functional.softmax(torch.tensor(scores), dim=0).numpy()
        
        labels = ["⭐️", "⭐️⭐️", "⭐️⭐️⭐️", "⭐️⭐️⭐️⭐️", "⭐️⭐️⭐️⭐️⭐️"]
        sentiment = labels[np.argmax(probs)]
        confidence = np.max(probs)

        st.subheader("🧠 Sentiment")
        st.write(f"**Sentiment:** {sentiment}  |  **Confidence:** {confidence:.2f}")
        
        # Optional: score-by-class
        with st.expander("Show All Sentiment Scores"):
            for lbl, p in zip(labels, probs):
                st.write(f"{lbl}: {p:.3f}")

