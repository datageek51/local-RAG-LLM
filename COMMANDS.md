# Audio Processing - Quick Command Reference

## Installation Commands

### Install Python Dependencies
```powershell
# All audio dependencies
pip install SpeechRecognition pydub

# Or use the requirements file
pip install -r requirements.txt
```

### Install FFmpeg (Choose One Method)

#### Windows - Chocolatey
```powershell
choco install ffmpeg
```

#### Windows - Scoop
```powershell
scoop install ffmpeg
```

#### Windows - Manual
1. Download from: https://ffmpeg.org/download.html
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to PATH
4. Restart terminal

#### macOS
```bash
brew install ffmpeg
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

### Verify Installation
```powershell
# Run automated test
python scripts/test_audio_setup.py

# Or use the installation script
.\install_audio_dependencies.ps1
```

## Run Commands

### Start Application
```powershell
# Activate virtual environment (if not already active)
.\tempvenv\Scripts\Activate.ps1

# Run Streamlit app
streamlit run localragdemo.py
```

### Access Audio Processing
1. Open browser (usually auto-opens)
2. Click **"🎤 Audio Processing"** in sidebar
3. Upload audio file
4. Configure and transcribe

## Testing Commands

### Test Audio Setup
```powershell
python scripts/test_audio_setup.py
```

### Check ChromaDB
```powershell
python scripts/check_chromadb.py
```

### Verify Python Environment
```powershell
# Check Python version
python --version

# Check installed packages
pip list | Select-String "speech|pydub|chromadb|ollama"

# Check FFmpeg
ffmpeg -version
```

## Troubleshooting Commands

### Reinstall Audio Dependencies
```powershell
pip uninstall SpeechRecognition pydub -y
pip install SpeechRecognition pydub
```

### Clear ChromaDB (Careful!)
```powershell
# Backup first!
Copy-Item -Recurse chroma_db chroma_db_backup

# Remove collections (use UI instead, safer)
# Or delete entire directory and restart
Remove-Item -Recurse -Force chroma_db
```

### Check Logs
```powershell
# View latest log file
Get-Content (Get-ChildItem logs | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName -Tail 50

# View all recent logs
Get-ChildItem logs -Filter "*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

### View Streamlit Cache
```powershell
# Clear Streamlit cache
streamlit cache clear
```

## Common Workflows

### First Time Setup
```powershell
# 1. Clone and navigate to repo
git clone <repo-url> local-RAG-LLM
cd local-RAG-LLM

# 2. Create virtual environment
python -m venv .\tempvenv

# 3. Activate virtual environment
.\tempvenv\Scripts\Activate.ps1

# 4. Install base dependencies
pip install -r requirements.txt

# 5. Install audio dependencies
pip install SpeechRecognition pydub

# 6. Install FFmpeg (choose your method)
choco install ffmpeg
# OR
scoop install ffmpeg

# 7. Verify setup
python scripts/test_audio_setup.py

# 8. Run application
streamlit run localragdemo.py
```

### Daily Usage
```powershell
# 1. Navigate to project
cd C:\Users\navinp\repos\local-RAG-LLM

# 2. Activate virtual environment
.\tempvenv\Scripts\Activate.ps1

# 3. Run application
streamlit run localragdemo.py

# 4. Process audio files via UI

# 5. When done, deactivate
deactivate
```

### Update Application
```powershell
# 1. Pull latest changes
git pull

# 2. Update dependencies
pip install -r requirements.txt --upgrade

# 3. Verify setup
python scripts/test_audio_setup.py

# 4. Run application
streamlit run localragdemo.py
```

## Environment Variables

### Set ChromaDB Path
```powershell
# Temporary (current session)
$env:CHROMA_PATH = "D:\MyData\vector_db"

# Permanent (user level)
[System.Environment]::SetEnvironmentVariable('CHROMA_PATH', 'D:\MyData\vector_db', 'User')
```

### Set Default Collection
```powershell
$env:COLLECTION_NAME = "my_collection"
```

### Set Models
```powershell
$env:EMBEDDING_MODEL = "phi"
$env:LLM_MODEL = "gemma:2b"
```

## Quick Checks

### Is Python Working?
```powershell
python --version
# Expected: Python 3.10+ 
```

### Is Virtual Environment Active?
```powershell
$env:VIRTUAL_ENV
# Should show path to tempvenv
```

### Are Audio Libraries Installed?
```powershell
python -c "import speech_recognition; import pydub; print('OK')"
# Expected: OK
```

### Is FFmpeg Available?
```powershell
ffmpeg -version
# Should show version info
```

### Is ChromaDB Working?
```powershell
python -c "import chromadb; print('ChromaDB version:', chromadb.__version__)"
# Should show version
```

### Is Ollama Running?
```powershell
python -c "import ollama; print('Ollama OK')"
# Expected: Ollama OK
```

## File Locations

### Configuration Files
- `requirements.txt` - Python dependencies
- `Utilities.py` - Configuration and utilities
- `.gitignore` - Git ignore rules (includes chroma_db)

### Application Files
- `localragdemo.py` - Main application
- `audio_page.py` - Audio processing page
- `chat_page.py` - Chat interface
- `Utilities.py` - Shared utilities

### Data Directories
- `chroma_db/` - Vector database storage
- `logs/` - Application logs
- `split_pdfs/` - Temporary PDF splits
- `tempvenv/` - Python virtual environment

### Documentation Files
- `README.md` - Main documentation
- `AUDIO_GUIDE.md` - Audio processing guide
- `AUDIO_EXAMPLES.md` - Example use cases
- `AUDIO_FEATURE_SUMMARY.md` - Feature summary

### Scripts
- `install_audio_dependencies.ps1` - Automated setup
- `scripts/test_audio_setup.py` - Verify setup
- `scripts/check_chromadb.py` - Check ChromaDB
- `scripts/test_chroma_ollama.py` - Test integration

## Port and Access

### Default URLs
- Streamlit app: http://localhost:8501
- Network access: http://<your-ip>:8501

### Change Port
```powershell
streamlit run localragdemo.py --server.port 8080
```

### Open in Browser
```powershell
streamlit run localragdemo.py --server.headless false
```

## Backup Commands

### Backup ChromaDB
```powershell
# Create backup
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item -Recurse chroma_db "chroma_db_backup_$timestamp"
```

### Restore ChromaDB
```powershell
# List backups
Get-ChildItem -Filter "chroma_db_backup_*"

# Restore from backup
Remove-Item -Recurse -Force chroma_db
Copy-Item -Recurse chroma_db_backup_20250115_120000 chroma_db
```

## Performance Tips

### Monitor Resource Usage
```powershell
# Open Task Manager
taskmgr

# Or use PowerShell
Get-Process python | Select-Object CPU, WorkingSet, ProcessName
```

### Reduce Memory Usage
```powershell
# Use smaller models
$env:LLM_MODEL = "tinyllama"
$env:EMBEDDING_MODEL = "phi"

# Restart application
streamlit run localragdemo.py
```

## Getting Help

### View Documentation
```powershell
# Open guides in default editor
notepad AUDIO_GUIDE.md
notepad AUDIO_EXAMPLES.md
notepad README.md
```

### Check Logs
```powershell
# View latest log
Get-ChildItem logs | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content
```

### Run Diagnostics
```powershell
python scripts/test_audio_setup.py
```
