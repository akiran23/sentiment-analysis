{rtf1\ansi\ansicpg1252\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Complete Multilingual Call Analysis Dashboard with MP3 Support\
# Dashboard + Backend Analyzer | pip install streamlit torch transformers pydub speechrecognition plotly pandas numpy scikit-learn\
\
import streamlit as st\
import pandas as pd\
import plotly.express as px\
import plotly.graph_objects as go\
from plotly.subplots import make_subplots\
import torch\
import numpy as np\
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration\
import io\
from pathlib import Path\
import tempfile\
import warnings\
warnings.filterwarnings('ignore')\
\
# Reuse the analyzer class from previous response\
class MultilingualCallAnalyzer:\
    def __init__(self):\
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")\
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")\
        self.sentiment_pipe = pipeline("sentiment-analysis", \
                                     model="cardiffnlp/twitter-xlm-roberta-base-sentiment")\
        self.emotion_pipe = pipeline("text-classification", model="vashuag/HindiEmotion")\
        \
        # Call center specific keywords (tailored for sales/customer service)\
        self.sales_keywords = ['buy', 'purchase', 'price', 'cost', 'deal', 'offer', 'subscription', 'upgrade', 'plan']\
        self.service_keywords = ['complaint', 'issue', 'problem', 'not working', 'refund', 'cancel', 'help', 'support']\
        self.products = ['phone', 'mobile', 'bike', 'ev', 'camera', 'laptop']  # User interests\
        self.missed_ops_keywords = ['interested but', 'maybe later', 'think about', 'too expensive', 'compare']\
    \
    @st.cache_data\
    def process_mp3(self, audio_bytes):\
        """Process MP3 file to analysis results"""\
        # Save uploaded MP3 to temp file\
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:\
            tmp.write(audio_bytes)\
            tmp_path = tmp.name\
        \
        try:\
            # Convert MP3 to WAV for Whisper\
            audio = AudioSegment.from_mp3(tmp_path)\
            audio = audio.set_frame_rate(16000).set_channels(1)\
            wav_path = tmp_path.replace('.mp3', '.wav')\
            audio.export(wav_path, format='wav')\
            \
            # Transcribe\
            with open(wav_path, 'rb') as f:\
                audio_input, _ = self.whisper_processor(f.read(), return_tensors='pt', sampling_rate=16000)\
            \
            generated_ids = self.whisper_model.generate(audio_input.input_features)\
            text = self.whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].lower()\
            \
            # Analyze\
            category = self.categorize_call(text)\
            sentiment = self.analyze_sentiment(text)\
            entities = self.extract_insights(text)\
            scores = self.calculate_scores(text, category)\
            \
            return \{\
                'transcription': text[:500] + '...' if len(text) > 500 else text,\
                'category': category,\
                'quality_score': scores['quality'],\
                'monetization_score': scores['monetization'],\
                'products': entities['products'],\
                'missed_opportunities': entities['missed_ops'],\
                'irate_customer': sentiment['irate'],\
                'full_text': text\
            \}\
        finally:\
            # Cleanup\
            for path in [tmp_path, wav_path]:\
                if Path(path).exists():\
                    Path(path).unlink()\
    \
    def categorize_call(self, text):\
        sales_score = sum(1 for kw in self.sales_keywords if kw in text)\
        service_score = sum(1 for kw in self.service_keywords if kw in text)\
        return 'Sales' if sales_score > service_score else 'Service'\
    \
    def analyze_sentiment(self, text):\
        try:\
            result = self.sentiment_pipe(text[:512])[0]\
            emotion = self.emotion_pipe(text[:512])[0]\
            irate = emotion['label'] in ['anger', 'disgust'] and emotion['score'] > 0.7\
            return \{'label': result['label'], 'irate': irate\}\
        except:\
            return \{'label': 'NEUTRAL', 'irate': False\}\
    \
    def extract_insights(self, text):\
        products = [p for p in self.products if p in text]\
        missed_ops = any(kw in text for kw in self.missed_ops_keywords)\
        return \{'products': products, 'missed_ops': missed_ops\}\
    \
    def calculate_scores(self, text, category):\
        quality = 5\
        monetization = 3\
        \
        if category == 'Sales':\
            if any(kw in text for kw in ['deal', 'sold', 'confirmed']):\
                monetization = 9\
            quality += 2 if len(self.extract_insights(text)['products']) > 0 else 0\
        else:  # Service\
            if any(kw in text for kw in ['resolved', 'fixed', 'satisfied']):\
                monetization = 8\
        \
        sentiment_bonus = 3 if 'positive' in self.analyze_sentiment(text)['label'].lower() else 0\
        quality = min(10, quality + sentiment_bonus)\
        monetization = min(10, monetization)\
        \
        return \{'quality': quality, 'monetization': monetization\}\
\
# === DASHBOARD ===\
def main():\
    st.set_page_config(page_title="Call Analysis Dashboard", layout="wide")\
    st.title("\uc0\u55356 \u57252  Multilingual Call Center Analytics Dashboard")\
    st.markdown("Analyze MP3 voice calls in English, Hindi, Telugu, Tamil, Malayalam, Kannada")\
    \
    analyzer = MultilingualCallAnalyzer()\
    \
    # Sidebar for file uploads\
    st.sidebar.header("Upload MP3 Files")\
    uploaded_files = st.sidebar.file_uploader("Choose MP3 files", \
                                            type=['mp3'], accept_multiple_files=True)\
    \
    if uploaded_files:\
        results_df = pd.DataFrame()\
        progress_bar = st.sidebar.progress(0)\
        \
        for i, file in enumerate(uploaded_files):\
            with st.spinner(f'Analyzing \{file.name\}...'):\
                result = analyzer.process_mp3(file.read())\
                result['filename'] = file.name\
                results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)\
            \
            progress_bar.progress((i + 1) / len(uploaded_files))\
        \
        # === METRICS ===\
        col1, col2, col3, col4 = st.columns(4)\
        with col1:\
            st.metric("Total Calls", len(results_df))\
        with col2:\
            sales_pct = len(results_df[results_df['category']=='Sales'])/len(results_df)*100\
            st.metric("Sales Calls %", f"\{sales_pct:.1f\}%")\
        with col3:\
            avg_quality = results_df['quality_score'].mean()\
            st.metric("Avg Quality Score", f"\{avg_quality:.1f\}/10")\
        with col4:\
            irate_calls = results_df['irate_customer'].sum()\
            st.metric("Irate Customers", irate_calls)\
        \
        # === OVERVIEW TABLE ===\
        st.subheader("\uc0\u55357 \u56522  Analysis Results")\
        display_df = results_df[['filename', 'category', 'quality_score', \
                               'monetization_score', 'products', 'missed_opportunities', \
                               'irate_customer']].copy()\
        display_df['products'] = display_df['products'].apply(lambda x: ', '.join(x) if x else 'None')\
        display_df['irate_customer'] = display_df['irate_customer'].apply(lambda x: '\uc0\u55357 \u57000 ' if x else '\u9989 ')\
        st.dataframe(display_df, use_container_width=True)\
        \
        # === CHARTS ===\
        col1, col2 = st.columns(2)\
        \
        with col1:\
            st.subheader("\uc0\u55357 \u56520  Call Categories")\
            cat_counts = results_df['category'].value_counts()\
            fig_pie = px.pie(values=cat_counts.values, names=cat_counts.index, \
                           color_discrete_sequence=['#FF6B6B', '#4ECDC4'])\
            fig_pie.update_layout(height=400)\
            st.plotly_chart(fig_pie, use_container_width=True)\
        \
        with col2:\
            st.subheader("\uc0\u55357 \u56522  Score Distributions")\
            fig_scores = make_subplots(specs=[[\{"secondary_y": True\}]])\
            fig_scores.add_trace(go.Histogram(x=results_df['quality_score'], \
                                            name="Quality Score", nbinsx=10), secondary_y=False)\
            fig_scores.add_trace(go.Histogram(x=results_df['monetization_score'], \
                                            name="Monetization", nbinsx=10), secondary_y=True)\
            fig_scores.update_layout(height=400, title="Score Analysis")\
            st.plotly_chart(fig_scores, use_container_width=True)\
        \
        # === INSIGHTS ===\
        st.subheader("\uc0\u55357 \u56481  Key Insights")\
        missed_ops = results_df['missed_opportunities'].sum()\
        top_products = results_df.explode('products')['products'].value_counts().head(3)\
        \
        col1, col2, col3 = st.columns(3)\
        with col1:\
            st.info(f"**\{missed_ops\}** missed sales opportunities detected")\
        with col2:\
            if not top_products.empty:\
                st.success(f"Top product: **\{top_products.index[0]\}** (\{top_products.iloc[0]\} mentions)")\
        with col3:\
            low_quality = results_df[results_df['quality_score'] < 5].shape[0]\
            st.warning(f"**\{low_quality\}** calls need quality improvement")\
        \
        # === RAW DATA ===\
        with st.expander("View Full Transcriptions"):\
            for _, row in results_df.iterrows():\
                st.markdown(f"**\{row['filename']\}** (\{row['category']\}) - Quality: \{row['quality_score']:.1f\}")\
                st.text_area("", row['full_text'], height=100, key=row['filename'])\
\
if __name__ == "__main__":\
    main()\
}
