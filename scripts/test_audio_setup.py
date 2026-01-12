"""
Test script to verify audio processing dependencies are correctly installed.
Run this script before using the audio processing features.
"""

import sys


def test_speech_recognition():
    """Test if SpeechRecognition is installed."""
    print("Testing SpeechRecognition...")
    try:
        import speech_recognition as sr
        print(f"  ✓ SpeechRecognition version: {sr.__version__}")
        
        # Test creating a recognizer
        recognizer = sr.Recognizer()
        print("  ✓ Recognizer initialized successfully")
        return True
    except ImportError:
        print("  ✗ SpeechRecognition not installed")
        print("    Install with: pip install SpeechRecognition")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_pydub():
    """Test if pydub is installed."""
    print("\nTesting pydub...")
    try:
        from pydub import AudioSegment
        print("  ✓ pydub imported successfully")
        return True
    except ImportError:
        print("  ✗ pydub not installed")
        print("    Install with: pip install pydub")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_ffmpeg():
    """Test if FFmpeg is available."""
    print("\nTesting FFmpeg...")
    try:
        import subprocess
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Extract version from first line
            version_line = result.stdout.split('\n')[0]
            print(f"  ✓ FFmpeg found: {version_line}")
            return True
        else:
            print("  ✗ FFmpeg command failed")
            return False
    except FileNotFoundError:
        print("  ✗ FFmpeg not found in PATH")
        print("    Download from: https://ffmpeg.org/download.html")
        print("    Or install via package manager:")
        print("      - Windows (Chocolatey): choco install ffmpeg")
        print("      - Windows (Scoop): scoop install ffmpeg")
        print("      - macOS: brew install ffmpeg")
        print("      - Linux: sudo apt-get install ffmpeg")
        return False
    except Exception as e:
        print(f"  ✗ Error checking FFmpeg: {e}")
        return False


def test_chromadb():
    """Test if ChromaDB is available."""
    print("\nTesting ChromaDB...")
    try:
        import chromadb
        version = getattr(chromadb, '__version__', 'unknown')
        print(f"  ✓ ChromaDB version: {version}")
        return True
    except ImportError:
        print("  ✗ ChromaDB not installed")
        print("    Install with: pip install chromadb")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_ollama():
    """Test if Ollama is available."""
    print("\nTesting Ollama...")
    try:
        import ollama
        print("  ✓ Ollama Python package installed")
        
        # Try to connect to Ollama service
        try:
            models = ollama.list()
            print(f"  ✓ Ollama service is running")
            if hasattr(models, 'models') and models.models:
                print(f"  ✓ {len(models.models)} model(s) available")
            return True
        except Exception as e:
            print(f"  ⚠ Ollama package installed but service not responding")
            print(f"    Make sure Ollama is running: {e}")
            return True  # Package is installed, which is what we're testing
    except ImportError:
        print("  ✗ Ollama not installed")
        print("    Install with: pip install ollama")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all tests and provide summary."""
    print("=" * 60)
    print("Audio Processing Dependencies Test")
    print("=" * 60)
    print()
    
    results = {
        "SpeechRecognition": test_speech_recognition(),
        "pydub": test_pydub(),
        "FFmpeg": test_ffmpeg(),
        "ChromaDB": test_chromadb(),
        "Ollama": test_ollama()
    }
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    required_passed = results["SpeechRecognition"] and results["pydub"]
    optional_passed = results["FFmpeg"] and results["ChromaDB"] and results["Ollama"]
    
    print("\nRequired for Audio Processing:")
    for name in ["SpeechRecognition", "pydub"]:
        status = "✓ PASS" if results[name] else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print("\nOptional (Enhanced Functionality):")
    print(f"  {'✓ PASS' if results['FFmpeg'] else '⚠ WARN'}: FFmpeg (for MP3/M4A/OGG support)")
    print(f"  {'✓ PASS' if results['ChromaDB'] else '⚠ WARN'}: ChromaDB (for persistent storage)")
    print(f"  {'✓ PASS' if results['Ollama'] else '⚠ WARN'}: Ollama (for local LLM)")
    
    print("\n" + "=" * 60)
    
    if required_passed:
        print("✓ Ready for audio processing!")
        if not results["FFmpeg"]:
            print("  Note: Only WAV files supported without FFmpeg")
        print("\nTo get started:")
        print("  1. Run: streamlit run localragdemo.py")
        print("  2. Navigate to: 🎤 Audio Processing")
        print("  3. Upload an audio file and start transcribing!")
        return 0
    else:
        print("✗ Audio processing setup incomplete")
        print("\nInstall missing required dependencies:")
        if not results["SpeechRecognition"]:
            print("  pip install SpeechRecognition")
        if not results["pydub"]:
            print("  pip install pydub")
        return 1


if __name__ == "__main__":
    sys.exit(main())
