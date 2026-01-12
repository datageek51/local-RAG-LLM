# PowerShell script to install audio processing dependencies
# Run this script to set up audio transcription capabilities

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Audio Processing Dependencies Setup" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "WARNING: No virtual environment detected!" -ForegroundColor Yellow
    Write-Host "It's recommended to activate your virtual environment first:" -ForegroundColor Yellow
    Write-Host "  .\tempvenv\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        Write-Host "Installation cancelled." -ForegroundColor Red
        exit
    }
}

Write-Host "Step 1: Installing Python packages..." -ForegroundColor Green
Write-Host "  - SpeechRecognition (for audio transcription)" -ForegroundColor White
Write-Host "  - pydub (for audio format conversion)" -ForegroundColor White
Write-Host ""

pip install SpeechRecognition pydub

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Python packages installed successfully!" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to install Python packages." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 2: Checking for FFmpeg..." -ForegroundColor Green

# Check if FFmpeg is installed
$ffmpegInstalled = $false
try {
    $ffmpegVersion = ffmpeg -version 2>$null
    if ($LASTEXITCODE -eq 0) {
        $ffmpegInstalled = $true
        Write-Host "✓ FFmpeg is already installed!" -ForegroundColor Green
    }
} catch {
    $ffmpegInstalled = $false
}

if (-not $ffmpegInstalled) {
    Write-Host "✗ FFmpeg is not installed or not in PATH." -ForegroundColor Red
    Write-Host ""
    Write-Host "FFmpeg is required for audio format conversion (MP3, M4A, OGG, etc.)" -ForegroundColor Yellow
    Write-Host "Without FFmpeg, only WAV files will be supported." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To install FFmpeg:" -ForegroundColor Cyan
    Write-Host "  1. Download from: https://ffmpeg.org/download.html" -ForegroundColor White
    Write-Host "  2. Extract the archive" -ForegroundColor White
    Write-Host "  3. Add the 'bin' folder to your system PATH" -ForegroundColor White
    Write-Host ""
    Write-Host "Alternative (using Chocolatey):" -ForegroundColor Cyan
    Write-Host "  choco install ffmpeg" -ForegroundColor White
    Write-Host ""
    Write-Host "Alternative (using Scoop):" -ForegroundColor Cyan
    Write-Host "  scoop install ffmpeg" -ForegroundColor White
    Write-Host ""
}

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Installation Summary" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "✓ SpeechRecognition installed" -ForegroundColor Green
Write-Host "✓ pydub installed" -ForegroundColor Green

if ($ffmpegInstalled) {
    Write-Host "✓ FFmpeg detected" -ForegroundColor Green
    Write-Host ""
    Write-Host "You're all set! Audio processing will support all formats." -ForegroundColor Green
} else {
    Write-Host "⚠ FFmpeg not detected (optional)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "You can use audio processing with WAV files now." -ForegroundColor Yellow
    Write-Host "Install FFmpeg for MP3, M4A, OGG, and FLAC support." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run: streamlit run localragdemo.py" -ForegroundColor White
Write-Host "  2. Navigate to: 🎤 Audio Processing" -ForegroundColor White
Write-Host "  3. Upload an audio file and start transcribing!" -ForegroundColor White
Write-Host ""
