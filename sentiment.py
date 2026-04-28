import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import os

# -----------------------------
# Load Model (once)
# -----------------------------
@st.cache_resource
def load_model():
    return WhisperModel("base", compute_type="int8")

model = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("📞 Call QA Analyzer (No API Required)")
st.write("Upload MP3/WAV → Get QA Score, Call Type, Violations")

uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav"])

# -----------------------------
# Transcription (LOCAL)
# -----------------------------
def transcribe_audio(file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    segments, _ = model.transcribe(tmp_path)

    transcript = " ".join([seg.text for seg in segments])

    os.remove(tmp_path)

    return transcript.lower()

# -----------------------------
# Call Type Classification
# -----------------------------
def classify_call(text):
    sales_keywords = ["buy", "offer", "discount", "price", "loan", "interest", "scheme"]
    service_keywords = ["issue", "problem", "complaint", "not working", "help", "support"]

    sales_score = sum(word in text for word in sales_keywords)
    service_score = sum(word in text for word in service_keywords)

    return "Sales" if sales_score > service_score else "Service"

# -----------------------------
# Fatal / Zero Tolerance
# -----------------------------
def check_fatal(text):
    fatal_phrases = ["don't know", "cannot help", "no solution", "not possible"]
    return any(p in text for p in fatal_phrases)

def check_zero_tolerance(text):
    rude_words = ["idiot", "stupid", "shut up", "useless", "angry tone"]
    return any(w in text for w in rude_words)

# -----------------------------
# Scoring Helper
# -----------------------------
def score(condition, points):
    return points if condition else 0

# -----------------------------
# Evaluation Logic
# -----------------------------
def evaluate_call(text, call_type):
    s = {}

    # Script/SOP
    s["Opening <4 sec"] = score("hello" in text[:50], 3)
    s["Greeting/Branding"] = score("welcome" in text or "muthoot" in text, 3)
    s["Paraphrasing"] = score("let me confirm" in text or "you mean" in text, 3)
    s["Proper Script"] = score("sir" in text or "madam" in text, 4)
    s["Closing Script"] = score("thank you" in text, 4)
    s["Confirm Name"] = score("your name" in text, 3)

    # Etiquette
    s["Hold Procedure"] = score("please hold" in text, 4)
    s["Dead Air"] = 3
    s["Mute Usage"] = 3

    # Clarity
    s["No Slang"] = score(not any(x in text for x in ["bro", "dude"]), 4)
    s["No Fillers"] = score(text.count("uh") < 5, 4)
    s["Clear Speaking"] = 4
    s["Listening Skills"] = score("i understand" in text or "got it" in text, 4)
    s["Tone"] = score(not any(x in text for x in ["angry", "frustrated"]), 4)

    # Calmness
    s["No Interruption"] = 5
    s["Confidence"] = 5

    # Professionalism
    s["No Jargon"] = 4
    s["Professional"] = score("sir" in text or "madam" in text, 3)
    s["Apology"] = score("sorry" in text, 3)

    # Rapport
    s["Ownership"] = score("i will help" in text or "i will check" in text, 4)
    s["Acknowledgement"] = score("okay" in text or "understood" in text, 3)
    s["Probing"] = score("can you explain" in text or "may i know" in text, 3)

    # Objection Handling
    s["Convincing"] = score("benefit" in text or "advantage" in text, 10)

    # Cross-sell
    if call_type == "Sales":
        s["Cross Sell"] = score("also" in text or "additional" in text, 10)
    else:
        s["Cross Sell"] = 0

    # Fatal & Zero tolerance
    fatal = check_fatal(text)
    zero_tol = check_zero_tolerance(text)

    if fatal or zero_tol:
        total = 0
    else:
        total = sum(s.values())

    return s, total, fatal, zero_tol

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if uploaded_file:
    st.info("⏳ Transcribing audio...")

    transcript = transcribe_audio(uploaded_file)

    st.subheader("📝 Transcript")
    st.write(transcript)

    call_type = classify_call(transcript)
    st.subheader(f"📊 Call Type: {call_type}")

    scores, total, fatal, zero_tol = evaluate_call(transcript, call_type)

    st.subheader("📋 Score Breakdown")
    st.json(scores)

    if fatal:
        st.error("❌ Fatal Error → Score = 0")

    if zero_tol:
        st.error("❌ Zero Tolerance Violation → Score = 0")

    st.subheader(f"🏁 Final Score: {total}")
