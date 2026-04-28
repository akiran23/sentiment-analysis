import streamlit as st
import tempfile
import whisper
import os

# ---------------------------
# Load Whisper Model (cached)
# ---------------------------
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# ---------------------------
# Transcription
# ---------------------------
def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result["text"].lower()

# ---------------------------
# Call Type Classification
# ---------------------------
def classify_call(text):
    sales_keywords = ["buy", "purchase", "offer", "loan", "interest", "scheme"]
    service_keywords = ["issue", "problem", "complaint", "help", "support"]

    sales_score = sum(k in text for k in sales_keywords)
    service_score = sum(k in text for k in service_keywords)

    return "Sales" if sales_score > service_score else "Service"

# ---------------------------
# QA Scoring Engine
# ---------------------------
def score_call(text, call_type):

    # ZERO TOLERANCE / FATAL
    rude_words = ["idiot", "stupid", "shut up", "nonsense"]
    if any(word in text for word in rude_words):
        return {"TOTAL": 0, "FATAL": True, "DETAILS": {}}

    scores = {}

    # Script adherence
    scores["Opening (Greeting)"] = 3 if any(x in text for x in ["hello", "good morning", "good evening"]) else 0
    scores["Branding / Intro"] = 3 if "muthoot" in text else 0
    scores["Closing"] = 4 if "thank you" in text else 0

    # Etiquette
    scores["Politeness"] = 4 if "please" in text else 0
    scores["No Dead Air (proxy)"] = 3 if len(text.split()) > 30 else 0

    # Clarity
    scores["No Slang"] = 4 if not any(x in text for x in ["bro", "dude"]) else 0
    scores["No Fillers"] = 4 if "uh" not in text else 0

    # Professionalism
    scores["Professional Tone"] = 3 if any(x in text for x in ["sir", "madam"]) else 0
    scores["Apology"] = 3 if "sorry" in text else 0

    # Rapport
    scores["Ownership"] = 4 if "i will help" in text or "let me check" in text else 0

    # Objection handling
    scores["Convincing"] = 10 if "benefit" in text or "advantage" in text else 0

    # Cross-sell (only sales)
    if call_type == "Sales":
        scores["Cross Sell"] = 10 if "offer" in text or "scheme" in text else 0
    else:
        scores["Cross Sell"] = 0

    total = sum(scores.values())

    return {"TOTAL": total, "FATAL": False, "DETAILS": scores}

# ---------------------------
# UI
# ---------------------------
st.title("📞 Call QA Analyzer (No API Key)")

uploaded_file = st.file_uploader("Upload Audio (MP3/WAV)", type=["mp3", "wav"])

if uploaded_file:

    st.audio(uploaded_file)

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    with st.spinner("Transcribing..."):
        text = transcribe_audio(temp_path)

    st.subheader("📝 Transcription")
    st.write(text)

    call_type = classify_call(text)

    st.subheader("📊 Call Type")
    st.write(call_type)

    result = score_call(text, call_type)

    st.subheader("📈 QA Score")

    if result["FATAL"]:
        st.error("❌ Zero Tolerance Triggered → Score = 0")
    else:
        st.success(f"Total Score: {result['TOTAL']}")
        st.json(result["DETAILS"])

    os.remove(temp_path)
