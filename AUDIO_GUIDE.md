# Audio Processing Quick Start Guide

## Overview
The Audio Processing feature allows you to transcribe audio files and add them to your ChromaDB knowledge base. Once transcribed, you can chat with the audio content just like documents!

## Supported Use Cases
- **Meeting Recordings**: Transcribe team meetings, client calls, or interviews
- **Podcasts & Lectures**: Convert educational content into searchable text
- **Voice Notes**: Process voice memos and notes
- **Interviews**: Transcribe interview recordings for analysis

## Prerequisites

### Required
1. **SpeechRecognition** - For audio transcription
   ```bash
   pip install SpeechRecognition
   ```

2. **pydub** - For audio format conversion
   ```bash
   pip install pydub
   ```

### Optional (Recommended)
3. **FFmpeg** - Required for non-WAV audio formats (MP3, M4A, OGG, FLAC)
   - **Windows**: Download from https://ffmpeg.org/download.html or use `choco install ffmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt-get install ffmpeg`

## Quick Installation
Run the automated setup script:
```powershell
.\install_audio_dependencies.ps1
```

Or verify your setup:
```powershell
python scripts/test_audio_setup.py
```

## Supported Audio Formats
- **WAV** - Always supported (no FFmpeg needed)
- **MP3** - Requires FFmpeg
- **M4A** - Requires FFmpeg
- **OGG** - Requires FFmpeg
- **FLAC** - Requires FFmpeg

## How to Use

### Step 1: Start the App
```powershell
streamlit run localragdemo.py
```

### Step 2: Navigate to Audio Processing
Click **"🎤 Audio Processing"** in the sidebar

### Step 3: Configure Settings
1. **Upload Audio File** - Select your audio file
2. **Language** - Choose the language spoken in the audio (default: English US)
3. **Split long audio** - Enable for files > 60 seconds (recommended)

### Step 4: Add Metadata (Optional)
- **Audio Source**: e.g., "Team Standup 2025-01-15"
- **Audio Category**: e.g., "Business Meeting", "Education"
- **Speaker/Host**: Name of the speaker

### Step 5: Choose Collection
- Select an existing collection to add the audio
- Or create a new collection for organizing different types of audio

### Step 6: Transcribe
Click **"Transcribe and Process Audio"** and wait for:
1. Audio conversion (if needed)
2. Audio splitting (if enabled)
3. Transcription (may take time for long files)
4. Text chunking
5. Storage in ChromaDB

### Step 7: Chat with Your Audio
Navigate to **"💬 Chat"** and ask questions about the transcribed content!

## Tips for Best Results

### Audio Quality
- Use clear audio with minimal background noise
- Ensure good microphone quality for recordings
- Reduce echo and reverb if possible

### File Length
- For files > 5 minutes, enable "Split long audio"
- Shorter segments generally transcribe more accurately
- Consider pre-splitting very long recordings

### Language Selection
- Select the correct language for accurate transcription
- Supported languages: English, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, Korean, and more

### Metadata Best Practices
- Use consistent naming conventions
- Add speaker names for multi-speaker content
- Include date and context in the source field
- Use categories to organize different types of audio

## Organizing Audio Collections

### Same Collection Strategy
**When to use**: Mix documents and audio in one collection
- Unified knowledge base (e.g., "company_knowledge")
- Related content (meeting notes + meeting audio)
- Cross-search across all content types

### Separate Collection Strategy
**When to use**: Create dedicated audio collections
- High volume of audio content
- Different time periods (e.g., "meetings_2025_q1")
- Different topics (e.g., "sales_calls", "training_sessions")
- Client-specific content

### Example Collection Names
- `team_meetings_2025`
- `customer_interviews`
- `training_sessions`
- `podcast_episodes`
- `legal_depositions`

## Transcription Details

### Service Used
The app uses **Google Speech Recognition** (free tier):
- Requires internet connection
- Audio is sent to Google for processing
- Generally accurate for clear audio
- Rate limits may apply for heavy usage

### Privacy Considerations
- Audio is processed by Google's cloud service
- Do not use for highly sensitive content without review
- Consider implementing local transcription (e.g., Whisper) for sensitive data

### Accuracy Factors
- Audio quality (clear vs. noisy)
- Speaker accent and clarity
- Background noise level
- Multiple speakers (may reduce accuracy)
- Technical terminology (may need custom models)

## Troubleshooting

### "SpeechRecognition not installed"
```bash
pip install SpeechRecognition
```

### "pydub not installed"
```bash
pip install pydub
```

### "FFmpeg not found"
- Install FFmpeg and add to system PATH
- Or use WAV files only

### "Could not understand audio"
- Check audio quality
- Reduce background noise
- Ensure correct language is selected
- Try a shorter audio clip first

### "Request error from speech recognition service"
- Check internet connection
- Verify Google Speech API is accessible
- Try again later (may be rate limited)

### Transcription is slow
- Normal for long audio files
- Enable audio splitting for better progress tracking
- Consider processing shorter segments

### Transcription is inaccurate
- Improve audio quality
- Reduce background noise
- Speak clearly and at moderate pace
- Use a better microphone
- Select correct language

## Advanced Usage

### Batch Processing Multiple Audio Files
Currently not supported in the UI, but you can:
1. Process files one at a time to the same collection
2. Or write a custom script using the `audio_page.py` functions

### Custom Chunking Strategies
The transcribed text uses fixed-size chunking (200 characters). To change:
- Modify the `fixed_size_chunking` call in `audio_page.py`
- Or use different chunking functions from `Utilities.py`

### Mixing Documents and Audio
You can add both documents and audio to the same collection:
1. Process documents via **"📄 Document Processing"**
2. Process audio via **"🎤 Audio Processing"**
3. Select the same collection name for both
4. Chat with all content together!

## Comparison with Document Processing

| Feature | Document Processing | Audio Processing |
|---------|-------------------|------------------|
| Input Format | PDF files | WAV, MP3, M4A, OGG, FLAC |
| Processing | Text extraction | Speech-to-text transcription |
| Internet Required | No | Yes (for transcription) |
| Speed | Fast | Slower (depends on length) |
| Accuracy | High (if PDF is text-based) | Variable (depends on audio quality) |
| Metadata | Source, Category | Source, Category, Speaker |
| Use Cases | Documents, reports, manuals | Meetings, podcasts, interviews |

## Next Steps
1. Process your first audio file
2. Review the transcription accuracy
3. Add to a collection
4. Navigate to Chat and ask questions
5. Experiment with different audio types
6. Organize collections by topic or time period

## Getting Help
- Check logs in the `logs/` directory
- Use the Debug Panel (🔧) in the sidebar
- Run `python scripts/test_audio_setup.py` to verify setup
- Review error messages for specific issues

## Future Enhancements (Possible)
- Local transcription with Whisper
- Speaker diarization (identifying different speakers)
- Batch audio processing
- Real-time transcription
- Audio quality pre-processing
- Custom vocabulary support
- Timestamp preservation
