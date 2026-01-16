import streamlit as st
import os
import logging
from datetime import datetime
from Utilities import *
import tempfile
import wave

# Configure logging
logger = logging.getLogger(__name__)

# Audio processing dependencies
try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None


def convert_audio_to_wav(audio_file, audio_format):
    """
    Convert audio file to WAV format for speech recognition.
    
    Args:
        audio_file: uploaded audio file object
        audio_format: original format (mp3, m4a, ogg, etc.)
    
    Returns:
        path to temporary WAV file
    """
    if AudioSegment is None:
        raise ImportError("pydub is required for audio format conversion. Install with: pip install pydub")
    
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_format}") as tmp_input:
        tmp_input.write(audio_file.read())
        input_path = tmp_input.name
    
    try:
        # Load audio file
        if audio_format == "mp3":
            audio = AudioSegment.from_mp3(input_path)
        elif audio_format == "m4a":
            audio = AudioSegment.from_file(input_path, format="m4a")
        elif audio_format == "ogg":
            audio = AudioSegment.from_ogg(input_path)
        elif audio_format == "flac":
            audio = AudioSegment.from_file(input_path, format="flac")
        else:
            audio = AudioSegment.from_file(input_path)
        
        # Convert to WAV
        output_path = tempfile.mktemp(suffix=".wav")
        audio.export(output_path, format="wav")
        
        # Clean up input file
        os.unlink(input_path)
        
        return output_path
    except Exception as e:
        # Clean up on error
        if os.path.exists(input_path):
            os.unlink(input_path)
        raise e


def transcribe_audio(audio_file_path, language="en-US"):
    """
    Transcribe audio file to text using speech recognition.
    
    Args:
        audio_file_path: path to WAV audio file
        language: language code for recognition (default: en-US)
    
    Returns:
        transcribed text string
    """
    if sr is None:
        raise ImportError("speech_recognition is required. Install with: pip install SpeechRecognition")
    
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_file_path) as source:
        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        # Record the audio
        audio_data = recognizer.record(source)
        
        try:
            # Use Google Speech Recognition (free)
            text = recognizer.recognize_google(audio_data, language=language)
            return text
        except sr.UnknownValueError:
            raise ValueError("Speech recognition could not understand the audio")
        except sr.RequestError as e:
            raise ConnectionError(f"Could not request results from speech recognition service: {e}")


def split_audio_by_duration(audio_file_path, max_duration_seconds=60):
    """
    Split long audio file into smaller chunks for better transcription.
    
    Args:
        audio_file_path: path to audio file
        max_duration_seconds: maximum duration for each chunk in seconds
    
    Returns:
        list of paths to audio chunk files
    """
    if AudioSegment is None:
        raise ImportError("pydub is required for audio splitting. Install with: pip install pydub")
    
    audio = AudioSegment.from_wav(audio_file_path)
    duration_ms = len(audio)
    max_duration_ms = max_duration_seconds * 1000
    
    if duration_ms <= max_duration_ms:
        return [audio_file_path]
    
    chunks = []
    for i in range(0, duration_ms, max_duration_ms):
        chunk = audio[i:i + max_duration_ms]
        chunk_path = tempfile.mktemp(suffix=f"_chunk_{i // max_duration_ms}.wav")
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    
    return chunks


def audio_processing_page():
    """Main page for audio processing and storage in ChromaDB."""
    st.title("🎤 Audio Processing")
    
    # Check dependencies
    if sr is None:
        st.error("❌ SpeechRecognition library not installed. Please install it to use audio processing.")
        st.code("pip install SpeechRecognition", language="bash")
        return
    
    if AudioSegment is None:
        st.warning("⚠️ PyDub not installed. Only WAV files will be supported. For MP3/M4A/OGG support, install pydub.")
        st.code("pip install pydub", language="bash")
    
    # File uploader
    supported_formats = ["wav"]
    if AudioSegment is not None:
        supported_formats.extend(["mp3", "m4a", "ogg", "flac"])
    
    uploaded_audio = st.file_uploader(
        "Upload Audio File",
        type=supported_formats,
        key='uploaded_audio'
    )
    
    # Transcription options
    st.markdown("**Transcription Settings:**")
    col1, col2 = st.columns(2)
    
    with col1:
        language_options = {
            "English (US)": "en-US",
            "English (UK)": "en-GB",
            "Spanish": "es-ES",
            "French": "fr-FR",
            "German": "de-DE",
            "Italian": "it-IT",
            "Portuguese": "pt-BR",
            "Chinese (Mandarin)": "zh-CN",
            "Japanese": "ja-JP",
            "Korean": "ko-KR"
        }
        selected_language = st.selectbox("Language", list(language_options.keys()))
        language_code = language_options[selected_language]
    
    with col2:
        chunk_audio = st.checkbox("Split long audio files", value=True, help="Split audio files longer than 60 seconds for better transcription")
    
    # Metadata inputs for audio (applied to all chunks)
    st.markdown("**Audio metadata (applies to all extracted text chunks):**")
    audio_source = st.text_input("Audio Source", value="", help="e.g., Meeting recording, Podcast, Interview")
    audio_category = st.text_input("Audio Category", value="", help="e.g., Business Meeting, Education, News")
    audio_speaker = st.text_input("Speaker/Host", value="", help="Name of speaker or host")
    
    # Collection selection
    st.markdown("**Target ChromaDB Collection:**")
    collection_names = []
    if chromadb is not None:
        try:
            client = chromadb.PersistentClient(
                path=CHROMA_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            cols = client.list_collections()
            for c in cols:
                try:
                    name = c.name if hasattr(c, 'name') else str(c)
                except Exception:
                    name = str(c)
                collection_names.append(name)
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
    
    new_collection_label = "-- Create new collection --"
    options = collection_names.copy()
    options.insert(0, new_collection_label)
    
    # Collection selection with refresh button
    sel_col, ref_col = st.columns([8, 1])
    with sel_col:
        selected_collection = st.selectbox(
            "Choose collection (existing or create new):",
            options,
            index=0
        )
    with ref_col:
        if st.button("🔄", help="Refresh collections list", key="refresh_collections_audio_page"):
            try:
                st.rerun()
            except AttributeError:
                try:
                    st.experimental_rerun()
                except Exception:
                    raise RuntimeError('Streamlit rerun API not available. Please upgrade Streamlit.')
    
    # Handle new collection creation
    new_collection_name = None
    chosen_collection_name = None
    is_new = selected_collection == new_collection_label
    
    if is_new:
        new_collection_name = st.text_input("New collection name:", value="")
        if new_collection_name:
            chosen_collection_name = new_collection_name.strip()
        else:
            chosen_collection_name = None
    else:
        chosen_collection_name = selected_collection
    
    # Validate collection name
    def _validate_collection_name(name: str, existing: list) -> tuple:
        if not name or not name.strip():
            return False, "Collection name cannot be empty"
        if len(name) < 1 or len(name) > 64:
            return False, "Collection name must be between 1 and 64 characters"
        import re
        if not re.match(r'^[A-Za-z0-9_\-]+$', name):
            return False, "Use only letters, numbers, underscores and hyphens"
        if name in existing:
            return False, "A collection with that name already exists"
        return True, ""
    
    collection_name_valid = True
    collection_name_msg = ""
    if is_new:
        ok, msg = _validate_collection_name(chosen_collection_name or "", collection_names)
        collection_name_valid = ok
        collection_name_msg = msg
        if not ok and chosen_collection_name:
            st.error(f"Invalid collection name: {msg}")
        elif not chosen_collection_name:
            st.info("Enter a name for the new collection before processing")
    
    # Process audio button
    if uploaded_audio is not None:
        # Show audio file info
        file_size_mb = uploaded_audio.size / (1024 * 1024)
        st.info(f"📁 File: {uploaded_audio.name} ({file_size_mb:.2f} MB)")
        
        # Play audio preview if browser supports it
        st.audio(uploaded_audio)
        
        # Disable process button if new collection and name invalid
        disable_process = False
        if is_new and (not chosen_collection_name or not collection_name_valid):
            disable_process = True
        
        if st.button("Transcribe and Process Audio", disabled=disable_process):
            temp_files = []  # Track temporary files for cleanup
            
            try:
                with st.spinner('Processing audio file...'):
                    # Reset file pointer
                    uploaded_audio.seek(0)
                    
                    logger.info(f"Starting audio processing for file: {uploaded_audio.name}")
                    logger.debug(f"File size: {uploaded_audio.size} bytes")
                    logger.debug(f"Collection name: {chosen_collection_name}")
                    logger.debug(f"Language: {language_code}")
                    
                    # Get audio format
                    audio_format = uploaded_audio.name.split('.')[-1].lower()
                    
                    # Convert to WAV if necessary
                    if audio_format != "wav":
                        st.info(f"Converting {audio_format.upper()} to WAV format...")
                        wav_path = convert_audio_to_wav(uploaded_audio, audio_format)
                        temp_files.append(wav_path)
                    else:
                        # Save WAV file to temporary location
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                            tmp_wav.write(uploaded_audio.read())
                            wav_path = tmp_wav.name
                        temp_files.append(wav_path)
                    
                    logger.info(f"Audio file saved to: {wav_path}")
                    
                    # Split audio if enabled
                    audio_chunks = []
                    if chunk_audio:
                        st.info("Splitting audio into manageable chunks...")
                        audio_chunks = split_audio_by_duration(wav_path, max_duration_seconds=60)
                        temp_files.extend([c for c in audio_chunks if c != wav_path])
                        logger.info(f"Audio split into {len(audio_chunks)} chunks")
                    else:
                        audio_chunks = [wav_path]
                    
                    # Transcribe each audio chunk
                    st.info(f"Transcribing audio ({len(audio_chunks)} chunk{'s' if len(audio_chunks) > 1 else ''})...")
                    transcriptions = []
                    
                    progress_bar = st.progress(0)
                    for i, chunk_path in enumerate(audio_chunks):
                        try:
                            logger.debug(f"Transcribing chunk {i+1}/{len(audio_chunks)}")
                            text = transcribe_audio(chunk_path, language=language_code)
                            transcriptions.append(text)
                            logger.debug(f"Chunk {i+1} transcribed: {len(text)} characters")
                        except Exception as e:
                            logger.error(f"Failed to transcribe chunk {i+1}: {e}")
                            st.warning(f"⚠️ Failed to transcribe chunk {i+1}: {e}")
                        
                        progress_bar.progress((i + 1) / len(audio_chunks))
                    
                    if not transcriptions:
                        st.error("❌ No audio could be transcribed. Please check the audio quality and try again.")
                        return
                    
                    # Combine all transcriptions
                    full_transcription = " ".join(transcriptions)
                    logger.info(f"Transcription complete. Total length: {len(full_transcription)} characters")
                    
                    # Display transcription
                    with st.expander("📝 Transcription Preview", expanded=True):
                        st.text_area("Transcribed Text", full_transcription, height=200)
                    
                    # Chunk the transcription text for ChromaDB
                    st.info("Processing transcription into text chunks...")
                    text_chunks = fixed_size_chunking(full_transcription, 1500)
                    logger.info(f"Created {len(text_chunks)} text chunks for storage")
                    
                    # Prepare metadata
                    audio_meta = {}
                    if audio_source:
                        audio_meta['source'] = audio_source
                    if audio_category:
                        audio_meta['category'] = audio_category
                    if audio_speaker:
                        audio_meta['speaker'] = audio_speaker
                    audio_meta['content_type'] = 'audio_transcription'
                    audio_meta['original_filename'] = uploaded_audio.name
                    
                    logger.debug(f"Audio metadata: {audio_meta}")
                    
                    # Store in ChromaDB
                    st.info(f"Storing {len(text_chunks)} chunks in ChromaDB...")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(current: int, total: int):
                        progress = float(current) / float(total) if total > 0 else 0
                        progress_bar.progress(progress)
                        status_text.text(f"Processing chunks: {current}/{total}")
                    
                    collection = initialize_chroma_db(
                        documents=text_chunks,
                        batch_size=10,
                        progress_callback=update_progress,
                        document_metadata=audio_meta if audio_meta else None,
                        collection_name=chosen_collection_name
                    )
                    
                    logger.info(f"ChromaDB processing complete. Collection: {chosen_collection_name}")
                    
                    progress_bar.progress(1.0)
                    status_text.text(f"Successfully processed all {len(text_chunks)} chunks!")
                    
                    # Store collection in session state
                    st.session_state['chroma_collection'] = collection
                    st.session_state['audio_processed'] = True
                    
                    st.success("✅ Audio transcription and storage complete!")
                    
                    # Preview chunks
                    with st.expander("Text chunks preview (first 5)", expanded=False):
                        for i, chunk in enumerate(text_chunks[:5]):
                            st.text(f"Chunk {i}:")
                            st.text(chunk[:1500] + "..." if len(chunk) > 1500 else chunk)
                    
                    # Navigation buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Go to Chat"):
                            try:
                                st.experimental_rerun()
                            except Exception:
                                try:
                                    st.rerun()
                                except Exception:
                                    raise RuntimeError('Streamlit rerun API not available. Please upgrade Streamlit.')
                    with col2:
                        if st.button("Clear & Upload Another"):
                            st.session_state['uploaded_audio'] = None
                            st.session_state['audio_processed'] = False
                            try:
                                st.experimental_rerun()
                            except Exception:
                                try:
                                    st.rerun()
                                except Exception:
                                    raise RuntimeError('Streamlit rerun API not available. Please upgrade Streamlit.')
                
            except Exception as e:
                logger.error(f"Failed to process audio: {str(e)}", exc_info=True)
                st.error(f"❌ Failed to process audio: {e}")
                st.info("Check the log file for detailed error information.")
            
            finally:
                # Clean up temporary files
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                            logger.debug(f"Cleaned up temp file: {temp_file}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up temp file {temp_file}: {e}")


if __name__ == "__main__":
    audio_processing_page()
