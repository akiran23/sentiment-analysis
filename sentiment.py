import streamlit as st
import torch
from transformers import pipeline, Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
from indicnlp.tokenize import indic_tokenize
import torchaudio

# Load models (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
@st.cache_resource
def load_models():
    # Whisper-like for Indic STT (simplified; use AI4Bharat's IndicWav2Vec for production)
    stt_processor = Wav2Vec2Processor.from_pretrained("ai4bharat/indicwav2vec-v1-hindi")
    stt_model = Wav2Vec2ForCTC.from_pretrained("ai4bharat/indicwav2vec-v1-hindi")
    sentiment = pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis", device=0 if device=="cuda" else -1)
    return stt_processor, stt_model, sentiment

stt_processor, stt_model, sentiment_pipeline = load_models()

st.title("Voice Sentiment Auditor for English & Indian Languages")
uploaded_file = st.file_uploader("Upload voice file (WAV)", type=["wav"])

if uploaded_file:
    # Load audio
    waveform, sample_rate = torchaudio.load(uploaded_file)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Transcribe
    inputs = stt_processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = stt_model(inputs.input_values.to(device)).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = stt_processor.batch_decode(predicted_ids)[0]
    st.text_area("Transcription", transcription, height=100)
    
    # Sentiment analysis (tokenize for Indic if needed)
    tokens = indic_tokenize.trivial_tokenize(transcription)
    text = ' '.join(tokens)
    result = sentiment_pipeline(text)[0]
    score = result['score']
    label = result['label']
    
    # Dashboard visuals
    st.metric("Sentiment", f"{label} ({score:.2%})")
    
    col1, col2 = st.columns(2)
    with col1:
        if label == "POSITIVE":
            st.success("Positive sentiment")
        elif label == "NEGATIVE":
            st.error("Negative sentiment")
        else:
            st.warning("Neutral sentiment")
    with col2:
        st.audio(uploaded_file)
    
    # Batch analysis placeholder (extend for multiple files)
    st.info("Upload more files or integrate folder processing for audits.")
