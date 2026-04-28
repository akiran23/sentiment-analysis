import streamlit as st
import tempfile
import os
import re
import pandas as pd
from faster_whisper import WhisperModel

# ---------------------------
# Load Model (lightweight)
# ---------------------------
@st.cache_resource
def load_model():
    return WhisperModel("base", compute_type="int8")  # fast CPU

model = load_model()

# ---------------------------
# Transcription
# ---------------------------
def transcribe_audio(file_path):
    segments, _ = model.transcribe(file_path)
    text = " ".join([seg.text for seg in segments])
    return text.lower()

# ---------------------------
# Clean text
# ---------------------------
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

# ---------------------------
# Call Type Classification
# ---------------------------
def classify_call(text):
    sales_patterns = r"\b(buy|purchase|offer|loan|interest|scheme|benefit)\b"
    service_patterns = r"\b(issue|problem|complaint|help|support)\b"

    sales_score = len(re.findall(sales_patterns, text))
    service_score = len(re.findall(service_patterns, text))

    return "Sales" if sales_score > service_score else "Service"

# ---------------------------
# QA Scoring
# ---------------------------
def score_call(text, call_type):

    rude_words = ["idiot", "stupid", "shut up"]
    if any(word in text for word in rude_words):
        return {"fatal": True, "total": 0, "data": []}

    results = []

    def add(title, param, score, condition):
        results.append({
            "Title": title,
            "Parameter": param,
            "Score": score if condition else 0
        })

    add("Script", "Greeting", 3, "hello" in text or "good morning" in text)
    add("Script", "Brand intro", 3, "muthoot" in text)
    add("Script", "Closing", 4, "thank you" in text)

    add("Etiquette", "Politeness", 4, "please" in text)
    add("Clarity", "No slang", 4, "bro" not in text)
    add("Professionalism", "Apology", 3, "sorry" in text)

    add("Rapport", "Ownership", 4, "i will help" in text)

    add("Objection", "Convincing", 10, "benefit" in text)

    if call_type == "Sales":
        add("Cross Sell", "Pitch", 10, "offer" in text)
    else:
        add("Cross Sell", "Pitch", 0, False)

    df = pd.DataFrame(results)
    total = df["Score"].sum()

    return {"fatal": False, "total": total, "data": df}

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Fast Call QA", layout="wide")

st.title("⚡ Fast Call QA Analyzer (No Torch Issues)")

uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])

if uploaded_file:
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    with st.spinner("Transcribing..."):
        text = clean_text(transcribe_audio(path))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Transcript")
        st.write(text)

    call_type = classify_call(text)

    with col2:
        st.subheader("📊 Call Type")
        st.write(call_type)

    result = score_call(text, call_type)

    st.subheader("📈 QA Score")

    if result["fatal"]:
        st.error("❌ Zero tolerance triggered → Score = 0")
    else:
        st.success(f"Total Score: {result['total']}")
        st.dataframe(result["data"], use_container_width=True)

        csv = result["data"].to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Report", csv, "report.csv")

    os.remove(path)
