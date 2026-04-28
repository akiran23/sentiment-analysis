import streamlit as st
import os
import json
import wave
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment

# -------------------------------
# Load Model
# -------------------------------
MODEL_PATH = "model"
model = Model(MODEL_PATH)

# -------------------------------
# Convert Audio to WAV mono
# -------------------------------
def convert_to_wav(uploaded_file):
    audio = AudioSegment.from_file(uploaded_file)
    audio = audio.set_channels(1).set_frame_rate(16000)
    wav_path = "temp.wav"
    audio.export(wav_path, format="wav")
    return wav_path

# -------------------------------
# Transcribe Audio
# -------------------------------
def transcribe_audio(wav_path):
    wf = wave.open(wav_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())

    text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            text += " " + res.get("text", "")

    final_res = json.loads(rec.FinalResult())
    text += " " + final_res.get("text", "")
    return text.lower()

# -------------------------------
# Classification: Sales vs Service
# -------------------------------
def classify_call(text):
    sales_keywords = ["buy", "purchase", "offer", "loan", "interest rate", "scheme"]
    service_keywords = ["issue", "problem", "complaint", "help", "support"]

    sales_score = sum([1 for k in sales_keywords if k in text])
    service_score = sum([1 for k in service_keywords if k in text])

    return "Sales" if sales_score > service_score else "Service"

# -------------------------------
# Scoring Logic
# -------------------------------
def score_call(text, call_type):

    scores = {}

    # Fatal + Zero tolerance
    rude_words = ["idiot", "stupid", "shut up"]
    if any(word in text for word in rude_words):
        return {"TOTAL": 0, "FATAL": True}

    # Example rules (expandable)
    scores["Opening"] = 3 if "hello" in text or "good morning" in text else 0
    scores["Intro"] = 3 if "muthoot" in text else 0
    scores["Closing"] = 4 if "thank you" in text else 0
    scores["Politeness"] = 3 if "please" in text else 0
    scores["Apology"] = 3 if "sorry" in text else 0
    scores["Clarity"] = 4 if "uh" not in text else 0
    scores["Professional"] = 3 if "sir" in text or "madam" in text else 0
    scores["Ownership"] = 4 if "i will help" in text else 0
    scores["Convincing"] = 10 if "benefit" in text else 0

    # Cross-sell only for sales
    if call_type == "Sales":
        scores["CrossSell"] = 10 if "offer" in text else 0
    else:
        scores["CrossSell"] = 0

    total = sum(scores.values())

    return {
        "TOTAL": total,
        "FATAL": False,
        "DETAILS": scores
    }

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📞 Call QA Analyzer (No API Key)")

uploaded_file = st.file_uploader("Upload Call Recording (MP3/WAV)")

if uploaded_file:
    st.audio(uploaded_file)

    with st.spinner("Processing..."):
        wav_path = convert_to_wav(uploaded_file)
        text = transcribe_audio(wav_path)
        call_type = classify_call(text)
        result = score_call(text, call_type)

    st.subheader("📝 Transcription")
    st.write(text)

    st.subheader("📊 Call Type")
    st.write(call_type)

    st.subheader("📈 Score")

    if result["FATAL"]:
        st.error("❌ Fatal/Zero Tolerance Triggered → Score = 0")
    else:
        st.write("Total Score:", result["TOTAL"])
        st.json(result["DETAILS"])
