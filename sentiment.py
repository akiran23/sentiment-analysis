import streamlit as st
import pandas as pd
import whisper
import tempfile
import os

st.set_page_config(page_title="Audio Call QA Scoring", layout="wide")
st.title("📞 Audio Call QA (Upload → Transcribe → Score)")

# -----------------------------
# LOAD WHISPER MODEL (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return whisper.load_model("base")  # use "small" or "medium" for better accuracy

model = load_model()

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload Call Audio", type=["mp3", "wav"])

transcript = ""

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    st.audio(uploaded_file)

    st.info("Transcribing audio...")

    result = model.transcribe(temp_path)
    transcript = result["text"]

    st.subheader("📝 Transcription")
    st.write(transcript)

    os.remove(temp_path)

# -----------------------------
# CALL TYPE
# -----------------------------
call_type = st.radio("Select Call Type", ["Service", "Sales"])

# -----------------------------
# SCORECARD
# -----------------------------
scorecard = [
    {"bucket": "Script", "param": "Opening within 4 secs", "score": 3},
    {"bucket": "Script", "param": "Proper greeting & branding", "score": 3},
    {"bucket": "Script", "param": "Paraphrasing when needed", "score": 3},
    {"bucket": "Script", "param": "Used proper scripts", "score": 4},
    {"bucket": "Script", "param": "Proper closing", "score": 4},
    {"bucket": "Script", "param": "Confirmed customer name", "score": 3},

    {"bucket": "Etiquette", "param": "Proper hold usage", "score": 4},
    {"bucket": "Etiquette", "param": "No dead air (<10 sec)", "score": 3},
    {"bucket": "Etiquette", "param": "Proper mute usage", "score": 3},

    {"bucket": "Clarity", "param": "No slang", "score": 4},
    {"bucket": "Clarity", "param": "No fillers", "score": 4},
    {"bucket": "Clarity", "param": "Customer understood agent", "score": 4},
    {"bucket": "Clarity", "param": "Agent understood customer", "score": 4},
    {"bucket": "Clarity", "param": "Tone was appropriate", "score": 4},

    {"bucket": "Calmness", "param": "No interruption", "score": 5},
    {"bucket": "Calmness", "param": "Confidence", "score": 5},

    {"bucket": "Professionalism", "param": "No jargon", "score": 4},
    {"bucket": "Professionalism", "param": "Professional behavior", "score": 3},
    {"bucket": "Professionalism", "param": "Apology where needed", "score": 3},

    {"bucket": "Rapport", "param": "Ownership", "score": 4},
    {"bucket": "Rapport", "param": "Acknowledgement", "score": 3},
    {"bucket": "Rapport", "param": "Probing", "score": 3},

    {"bucket": "Objection", "param": "Convincing skills", "score": 10},
    {"bucket": "Sales", "param": "Cross-sell", "score": 10},
]

fatal_checks = ["Correct solution provided?"]
zero_tolerance = ["Agent rude/sarcastic?", "Abusive language used?"]

# -----------------------------
# SCORING UI
# -----------------------------
st.subheader("📊 Parameter-wise Scoring")

results = []
total_score = 0
max_score = 0

for item in scorecard:
    if item["bucket"] == "Sales" and call_type != "Sales":
        continue

    col1, col2 = st.columns([4, 1])

    with col1:
        st.write(f"**{item['bucket']}** → {item['param']} ({item['score']})")

    with col2:
        choice = st.selectbox("", ["Full Score", "0"], key=item["param"])

    score = item["score"] if choice == "Full Score" else 0

    results.append({
        "Bucket": item["bucket"],
        "Parameter": item["param"],
        "Max Score": item["score"],
        "Score": score
    })

    total_score += score
    max_score += item["score"]

# -----------------------------
# FATAL & ZT
# -----------------------------
st.subheader("🚨 Fatal Check")
fatal = any(st.radio(q, ["Yes", "No"], key=q) == "No" for q in fatal_checks)

st.subheader("⚠️ Zero Tolerance")
zt = any(st.radio(q, ["No", "Yes"], key=q) == "Yes" for q in zero_tolerance)

# -----------------------------
# FINAL SCORE
# -----------------------------
if fatal or zt:
    final_score = 0
    status = "❌ AUTO FAIL"
else:
    final_score = total_score
    status = "✅ PASS"

# -----------------------------
# OUTPUT
# -----------------------------
df = pd.DataFrame(results)

st.subheader("📋 Scorecard")
st.dataframe(df, use_container_width=True)

st.subheader("📈 Final Score")
st.metric("Score", f"{final_score} / {max_score}")
st.write("Status:", status)

# DOWNLOAD
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download CSV",
    csv,
    "qa_scorecard.csv",
    "text/csv"
)
