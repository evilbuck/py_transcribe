#!/usr/bin/env python3
"""
Debug test to identify the hanging issue
"""
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, '.')

def test_minimal_import():
    """Test minimal import to see where it hangs"""
    print("Testing minimal import...")
    
    try:
        print("1. Importing transcribe module...")
        from transcribe import AudioTranscriber
        print("✓ Import successful")
        
        print("2. Creating AudioTranscriber instance...")
        transcriber = AudioTranscriber(model_size="tiny", device="cpu", compute_type="float32")
        print("✓ Instance created")
        
        print("3. Getting model info...")
        info = transcriber.get_model_info()
        print(f"✓ Model info: {info}")
        
        print("4. Checking if model loads...")
        # This is where it might hang
        transcriber._initialize_model()
        print("✓ Model initialized")
        
        print("5. Cleaning up...")
        transcriber.close()
        print("✓ Cleanup complete")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal_import()