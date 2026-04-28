import streamlit as st
import pandas as pd

from deep_translator import GoogleTranslator

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Call QA Scoring", layout="wide")
st.title("📞 AI Call QA Scoring System (Multilingual + Parameter-wise)")

# -----------------------------
# LANGUAGE FUNCTIONS
# -----------------------------
def detect_language(text):
    lang, _ = langid.classify(text)
    return lang

def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text

# -----------------------------
# INPUT: CALL TYPE
# -----------------------------
call_type = st.radio("Select Call Type", ["Service", "Sales"])

# -----------------------------
# INPUT: TRANSCRIPT
# -----------------------------
transcript = st.text_area("Paste Call Transcript (Any Language)")

if transcript:
    lang = detect_language(transcript)
    translated = translate_to_english(transcript)

    st.subheader("🌍 Language Detection")
    st.write(f"Detected Language: **{lang.upper()}**")

    st.subheader("🔁 Translated to English")
    st.write(translated)

# -----------------------------
# SCORECARD CONFIG
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

    {"bucket": "Clarity", "param": "No slang / no language switching", "score": 4},
    {"bucket": "Clarity", "param": "No fillers / fumbling", "score": 4},
    {"bucket": "Clarity", "param": "Customer understood agent", "score": 4},
    {"bucket": "Clarity", "param": "Agent understood customer", "score": 4},
    {"bucket": "Clarity", "param": "Good tone / energy", "score": 4},

    {"bucket": "Calmness", "param": "No interruption", "score": 5},
    {"bucket": "Calmness", "param": "Confidence throughout", "score": 5},

    {"bucket": "Professionalism", "param": "No jargon", "score": 4},
    {"bucket": "Professionalism", "param": "Professional behavior", "score": 3},
    {"bucket": "Professionalism", "param": "Apology where needed", "score": 3},

    {"bucket": "Rapport", "param": "Ownership / alternatives", "score": 4},
    {"bucket": "Rapport", "param": "Acknowledgement", "score": 3},
    {"bucket": "Rapport", "param": "Probing skills", "score": 3},

    {"bucket": "Objection", "param": "Convincing skills", "score": 10},
    {"bucket": "Sales", "param": "Cross-sell effectiveness", "score": 10},
]

fatal_checks = ["Correct / complete solution provided?"]
zero_tolerance = ["Agent rude or sarcastic?", "Abusive language used?"]

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
        st.write(f"**{item['bucket']}** → {item['param']} (Max: {item['score']})")

    with col2:
        choice = st.selectbox("", ["Full Score", "0"], key=item["param"])

    score = item["score"] if choice == "Full Score" else 0

    results.append({
        "Bucket": item["bucket"],
        "Parameter": item["param"],
        "Max Score": item["score"],
        "Score Given": score,
        "Status": "✅ Met" if score > 0 else "❌ Not Met"
    })

    total_score += score
    max_score += item["score"]

# -----------------------------
# FATAL CHECK
# -----------------------------
st.subheader("🚨 Fatal Check")

fatal_triggered = False
for q in fatal_checks:
    val = st.radio(q, ["Yes", "No"], key=q)
    if val == "No":
        fatal_triggered = True

# -----------------------------
# ZERO TOLERANCE
# -----------------------------
st.subheader("⚠️ Zero Tolerance")

zt_triggered = False
for q in zero_tolerance:
    val = st.radio(q, ["No", "Yes"], key=q)
    if val == "Yes":
        zt_triggered = True

# -----------------------------
# FINAL SCORE
# -----------------------------
if fatal_triggered or zt_triggered:
    final_score = 0
    status = "❌ AUTO-FAIL"
else:
    final_score = total_score
    status = "✅ PASS"

# -----------------------------
# DISPLAY TABLE
# -----------------------------
st.subheader("📋 Parameter-wise Scorecard")

df = pd.DataFrame(results)
st.dataframe(df, use_container_width=True)

# -----------------------------
# BUCKET SUMMARY
# -----------------------------
st.subheader("📊 Bucket Summary")

bucket_df = df.groupby("Bucket").agg({
    "Score Given": "sum",
    "Max Score": "sum"
}).reset_index()

st.dataframe(bucket_df, use_container_width=True)

# -----------------------------
# FINAL RESULT
# -----------------------------
st.subheader("📈 Final Result")

st.metric("Final Score", f"{final_score} / {max_score}")
st.write("Status:", status)

# -----------------------------
# DOWNLOAD CSV
# -----------------------------
csv = df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="⬇️ Download Scorecard",
    data=csv,
    file_name="call_qa_scorecard.csv",
    mime="text/csv"
)

# -----------------------------
# INSIGHTS
# -----------------------------
st.subheader("🧠 Insights")

if final_score == 0:
    st.error("Critical failure due to Fatal / Zero Tolerance violation.")
else:
    ratio = final_score / max_score

    if ratio > 0.8:
        st.success("Strong call handling.")
    elif ratio > 0.6:
        st.warning("Average performance. Improvement needed.")
    else:
        st.error("Poor call quality. Immediate coaching required.")
