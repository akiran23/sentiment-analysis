if audio_file:
    st.audio(audio_file)
    audio_bytes = audio_file.read()
    audio_file.seek(0)  # Reset for st.audio
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name
    
    if st.button("🔍 Analyze Call", type="primary"):
        try:
            with st.spinner("🔄 Transcribing speech..."):
                whisper_model = load_whisper()  # Cached, fast
                result = whisper_model.transcribe(audio_path, language=None)
                transcript = result["text"].strip()
            # ... rest unchanged
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
