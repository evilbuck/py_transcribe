#!/usr/bin/env python3
"""
Quick test with the shorter audio file
"""
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, '.')

from transcribe import AudioTranscriber, validate_input_file

def main():
    """Quick test with short audio file"""
    print("=== Quick Test with Short Audio ===")
    
    # Use the shorter audio file
    test_audio = Path("assets/testing speech audio file.m4a")
    
    if test_audio.exists():
        try:
            # Validate input
            audio_path = validate_input_file(test_audio)
            print(f"Using audio file: {audio_path} (duration: ~11 seconds)")
            
            # Create transcriber
            transcriber = AudioTranscriber(
                model_size="tiny",
                device="cpu", 
                compute_type="float32",
                verbose=True
            )
            
            # Test transcription
            print("Starting transcription...")
            text = transcriber.transcribe_text_only(audio_path)
            print(f"✓ Transcribed text: '{text.strip()}'")
            
            # Clean up
            transcriber.close()
            print("✓ Test completed successfully!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Short audio file not found")

if __name__ == "__main__":
    main()