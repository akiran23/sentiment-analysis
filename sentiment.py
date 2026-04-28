import streamlit as st
import pandas as pd
import re


# ---------------------------
# Setup
# ---------------------------


# ---------------------------
# Transcription (API)
# ---------------------------
def transcribe_audio(file):
    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=file
    )
    return transcript.text.lower()

# ---------------------------
# Clean text
# ---------------------------
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

# ---------------------------
# Classification
# ---------------------------
def classify_call(text):
    sales = len(re.findall(r"\b(buy|offer|loan|scheme)\b", text))
    service = len(re.findall(r"\b(issue|problem|complaint)\b", text))
    return "Sales" if sales > service else "Service"

# ---------------------------
# QA scoring
# ---------------------------
def score_call(text, call_type):

    rude_words = ["idiot", "stupid", "shut up"]
    if any(w in text for w in rude_words):
        return {"fatal": True, "total": 0, "data": None}

    results = []

    def add(p, s, cond):
        results.append({"Parameter": p, "Score": s if cond else 0})

    add("Greeting", 3, "hello" in text)
    add("Brand intro", 3, "muthoot" in text)
    add("Closing", 4, "thank you" in text)
    add("Politeness", 4, "please" in text)
    add("Apology", 3, "sorry" in text)
    add("Ownership", 4, "i will help" in text)

    if call_type == "Sales":
        add("Cross sell", 10, "offer" in text)

    df = pd.DataFrame(results)
    return {"fatal": False, "total": df["Score"].sum(), "data": df}

# ---------------------------
# UI
# ---------------------------
st.title("📞 Call QA Analyzer (Audio Enabled)")

audio_file = st.file_uploader("Upload Call Recording", type=["mp3", "wav"])

if audio_file:

    st.audio(audio_file)

    with st.spinner("Transcribing..."):
        text = transcribe_audio(audio_file)

    st.subheader("📝 Transcript")
    st.write(text)

    call_type = classify_call(text)
    st.write("📊 Call Type:", call_type)

    result = score_call(text, call_type)

    if result["fatal"]:
        st.error("❌ Zero tolerance → Score = 0")
    else:
        st.success(f"Total Score: {result['total']}")
        st.dataframe(result["data"])
