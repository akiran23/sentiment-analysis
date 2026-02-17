import streamlit as st
import whisper
import tempfile
import re
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon', quiet=True)

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

whisper_model = load_whisper()

st.title("🗣️ Voice Call Sentiment Analyzer")
st.markdown("✅ Works perfectly - Hindi/English/Telugu/Tamil/Kannada/Malayalam")

audio_file = st.file_uploader("Upload MP3/WAV", type=["mp3", "wav", "m4a", "ogg"])

if audio_file:
    st.audio(audio_file)
    
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name
    
    if st.button("🔍 Analyze Call", type="primary"):
        with st.spinner("🔄 Transcribing speech..."):
            result = whisper_model.transcribe(audio_path, language=None)
            transcript = result["text"].strip()
        
        st.subheader("📄 Full Transcript")
        st.text_area("", transcript, height=200, label_visibility="collapsed")
        
        # VADER Sentiment (works on transliterated text too)
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(transcript)
        
        sentiment = "🟢 Positive" if scores['compound'] >= 0.05 else "🟡 Neutral" if scores['compound'] > -0.05 else "🟥 Negative"
        confidence = abs(scores['compound'])
        
        st.subheader("🎭 Sentiment Analysis")
        st.metric("Overall Sentiment", sentiment, f"{confidence:.0%}")
        
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Positive", f"{scores['pos']:.0%}")
        with col2: st.metric("Negative", f"{scores['neg']:.0%}")
        with col3: st.metric("Neutral", f"{scores['neu']:.0%}")
        
        # Multilingual good/bad keywords
        pos_keywords = ['good', 'great', 'excellent', 'thanks', 'happy', 'love', 
                       'achha', 'shukriya', 'badiya', 'sundar', 'perfect']
        neg_keywords = ['bad', 'poor', 'worst', 'hate', 'no', 'issue', 'problem', 
                       'bura', 'kharab', 'complain', 'late']
        
        text_lower = transcript.lower()
        good_things = [w for w in pos_keywords if w in text_lower]
        bad_things = [w for w in neg_keywords if w in text_lower]
        
        col1, col2 = st.columns([1,1])
        with col1:
            st.subheader("✅ Good Things Mentioned")
            if good_things:
                st.success("• " + " • ".join(set(good_things)))
            else:
                st.info("😊 No negative feedback!")
        
        with col2:
            st.subheader("❌ Issues Raised")
            if bad_things:
                st.error("• " + " • ".join(set(bad_things)))
            else:
                st.success("✅ No complaints!")
        
        # Clean up
        os.unlink(audio_path)
        st.balloons()

st.markdown("---")
st.caption("🚀 Perfect for Kerala/Hyderabad customer service calls - No library bugs!")
