import streamlit as st
import tempfile
import whisper
import os
import re
import pandas as pd

# ---------------------------
# Load Whisper Model (FAST)
# ---------------------------
@st.cache_resource
def load_model():
    return whisper.load_model("small")  # faster than base, good accuracy

model = load_model()

# ---------------------------
# Transcription
# ---------------------------
def transcribe_audio(file_path):
    result = model.transcribe(file_path, fp16=False)
    return result["text"].lower()

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
    service_patterns = r"\b(issue|problem|complaint|help|support|not working)\b"

    sales_score = len(re.findall(sales_patterns, text))
    service_score = len(re.findall(service_patterns, text))

    return "Sales" if sales_score > service_score else "Service"

# ---------------------------
# Sentiment Detection
# ---------------------------
def detect_sentiment(text):
    negative_words = ["angry", "frustrated", "bad", "worst", "not happy"]
    return "Negative" if any(w in text for w in negative_words) else "Neutral/Positive"

# ---------------------------
# QA Scoring Engine
# ---------------------------
def score_call(text, call_type):

    # -----------------------
    # ZERO TOLERANCE
    # -----------------------
    rude_words = ["idiot", "stupid", "shut up", "nonsense"]
    if any(word in text for word in rude_words):
        return {"fatal": True, "total": 0, "data": []}

    results = []

    def add(title, param, score, condition):
        results.append({
            "Title": title,
            "Parameter": param,
            "Score": score if condition else 0
        })

    # -----------------------
    # Script adherence
    # -----------------------
    add("Script", "Opening greeting", 3,
        bool(re.search(r"\b(hello|good morning|good evening)\b", text)))

    add("Script", "Brand intro", 3,
        "muthoot" in text)

    add("Script", "Closing", 4,
        "thank you" in text)

    # -----------------------
    # Etiquette
    # -----------------------
    add("Etiquette", "Politeness", 4,
        "please" in text)

    add("Etiquette", "Dead air avoided (proxy)", 3,
        len(text.split()) > 40)

    # -----------------------
    # Clarity
    # -----------------------
    add("Clarity", "No slang", 4,
        not re.search(r"\b(bro|dude)\b", text))

    add("Clarity", "No fillers", 4,
        "uh" not in text)

    add("Clarity", "Understood customer", 4,
        "i understand" in text or "got it" in text)

    # -----------------------
    # Calmness
    # -----------------------
    add("Calmness", "No interruption (proxy)", 5,
        "wait" not in text)

    # -----------------------
    # Professionalism
    # -----------------------
    add("Professionalism", "Professional tone", 3,
        bool(re.search(r"\b(sir|madam)\b", text)))

    add("Professionalism", "Apology", 3,
        "sorry" in text)

    # -----------------------
    # Rapport
    # -----------------------
    add("Rapport", "Ownership", 4,
        "i will help" in text or "let me check" in text)

    add("Rapport", "Acknowledgement", 3,
        "understand" in text)

    # -----------------------
    # Objection handling
    # -----------------------
    add("Objection", "Convincing skills", 10,
        "benefit" in text or "advantage" in text)

    # -----------------------
    # Cross-sell
    # -----------------------
    if call_type == "Sales":
        add("Cross Sell", "Pitch", 10,
            "offer" in text or "scheme" in text)
    else:
        add("Cross Sell", "Pitch", 0, False)

    df = pd.DataFrame(results)
    total_score = df["Score"].sum()

    return {"fatal": False, "total": total_score, "data": df}


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Call QA Analyzer", layout="wide")

st.title("📞 Call QA Analyzer (Optimized)")

uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])

if uploaded_file:

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    with st.spinner("Transcribing..."):
        raw_text = transcribe_audio(temp_path)

    text = clean_text(raw_text)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Transcription")
        st.write(text)

    call_type = classify_call(text)
    sentiment = detect_sentiment(text)

    with col2:
        st.subheader("📊 Insights")
        st.write("Call Type:", call_type)
        st.write("Sentiment:", sentiment)

    result = score_call(text, call_type)

    st.subheader("📈 QA Scorecard")

    if result["fatal"]:
        st.error("❌ Zero Tolerance Triggered → Score = 0")
    else:
        st.success(f"Total Score: {result['total']}")
        st.dataframe(result["data"], use_container_width=True)

        # Download
        csv = result["data"].to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Report", csv, "qa_report.csv", "text/csv")

    os.remove(temp_path)
