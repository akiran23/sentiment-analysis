import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="Call QA Analyzer", layout="wide")

# ---------------------------
# Clean text
# ---------------------------
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())

# ---------------------------
# Classification
# ---------------------------
def classify_call(text):
    sales_patterns = r"\b(buy|purchase|offer|loan|interest|scheme|benefit)\b"
    service_patterns = r"\b(issue|problem|complaint|help|support)\b"

    sales_score = len(re.findall(sales_patterns, text))
    service_score = len(re.findall(service_patterns, text))

    return "Sales" if sales_score > service_score else "Service"

# ---------------------------
# QA Scoring Engine
# ---------------------------
def score_call(text, call_type):

    rude_words = ["idiot", "stupid", "shut up", "nonsense"]

    # ZERO TOLERANCE
    if any(word in text for word in rude_words):
        return {"fatal": True, "total": 0, "data": None}

    results = []

    def add(title, param, score, condition):
        results.append({
            "Title": title,
            "Parameter": param,
            "Score": score if condition else 0
        })

    # Script
    add("Script", "Opening greeting", 3,
        bool(re.search(r"\b(hello|good morning|good evening)\b", text)))

    add("Script", "Brand intro", 3, "muthoot" in text)

    add("Script", "Closing", 4, "thank you" in text)

    # Etiquette
    add("Etiquette", "Politeness", 4, "please" in text)

    add("Etiquette", "Dead air avoided (proxy)", 3,
        len(text.split()) > 30)

    # Clarity
    add("Clarity", "No slang", 4,
        not re.search(r"\b(bro|dude)\b", text))

    add("Clarity", "No fillers", 4,
        "uh" not in text)

    add("Clarity", "Understood customer", 4,
        "i understand" in text or "got it" in text)

    # Calmness
    add("Calmness", "No interruption (proxy)", 5,
        "wait" not in text)

    # Professionalism
    add("Professionalism", "Professional tone", 3,
        bool(re.search(r"\b(sir|madam)\b", text)))

    add("Professionalism", "Apology", 3,
        "sorry" in text)

    # Rapport
    add("Rapport", "Ownership", 4,
        "i will help" in text or "let me check" in text)

    add("Rapport", "Acknowledgement", 3,
        "understand" in text)

    # Objection
    add("Objection", "Convincing skills", 10,
        "benefit" in text or "advantage" in text)

    # Cross sell
    if call_type == "Sales":
        add("Cross Sell", "Pitch", 10,
            "offer" in text or "scheme" in text)
    else:
        add("Cross Sell", "Pitch", 0, False)

    df = pd.DataFrame(results)
    total = df["Score"].sum()

    return {"fatal": False, "total": total, "data": df}


# ---------------------------
# UI
# ---------------------------
st.title("📞 Call QA Analyzer (Audio or Transcript)")

option = st.radio("Choose Input Type", ["Upload Audio", "Paste Transcript"])

text = ""

# ---------------------------
# AUDIO (no processing)
# ---------------------------
if option == "Upload Audio":
    audio_file = st.file_uploader("Upload MP3/WAV", type=["mp3", "wav"])

    if audio_file:
        st.audio(audio_file)
        st.warning("⚠️ Audio transcription disabled for stability. Please paste transcript below.")

        text = st.text_area("Paste transcript for analysis")

# ---------------------------
# TEXT
# ---------------------------
else:
    text = st.text_area("Paste Call Transcript Here")

# ---------------------------
# PROCESS
# ---------------------------
if st.button("Analyze Call"):

    if not text.strip():
        st.error("Please provide transcript")
    else:
        text = clean_text(text)

        call_type = classify_call(text)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 Cleaned Transcript")
            st.write(text)

        with col2:
            st.subheader("📊 Call Type")
            st.write(call_type)

        result = score_call(text, call_type)

        st.subheader("📈 QA Score")

        if result["fatal"]:
            st.error("❌ Zero Tolerance Triggered → Score = 0")
        else:
            st.success(f"Total Score: {result['total']}")
            st.dataframe(result["data"], use_container_width=True)

            csv = result["data"].to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Report", csv, "qa_report.csv")
