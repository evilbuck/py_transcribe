#!/usr/bin/env python3
"""
Example usage of the AudioTranscriber without CLI
"""
from pathlib import Path
from transcribe import AudioTranscriber, validate_input_file

def main():
    """Example usage of the transcriber module"""
    
    # Example 1: Simple text-only transcription
    print("=== Example 1: Simple Text Transcription ===")
    
    # Check if test audio file exists (use the short one for demo)
    test_audio = Path("assets/testing speech audio file.m4a")
    if test_audio.exists():
        try:
            # Validate input
            audio_path = validate_input_file(test_audio)
            
            # Create transcriber with tiny model for fast testing
            transcriber = AudioTranscriber(
                model_size="tiny",
                device="cpu", 
                compute_type="float32",
                verbose=True
            )
            
            # Get just the text
            text = transcriber.transcribe_text_only(audio_path)
            print(f"Transcribed text (first 200 chars): {text[:200]}...")
            
            # Clean up
            transcriber.close()
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Test audio file not found, skipping example 1")
    
    print("\n" + "="*50)
    
    # Example 2: Full transcription with metadata and output file
    print("=== Example 2: Full Transcription with Output ===")
    
    if test_audio.exists():
        try:
            transcriber = AudioTranscriber(
                model_size="tiny",
                device="cpu",
                debug=True  # Enable debug logging
            )
            
            output_file = Path("example_output.txt")
            
            # Get full results
            results = transcriber.transcribe_file(
                audio_path, 
                output_file,
                transcription_params={
                    'beam_size': 3,  # Faster for demo
                    'vad_filter': True,
                    'temperature': 0.0
                }
            )
            
            print(f"Language detected: {results['language']}")
            print(f"Confidence: {results['language_probability']:.2f}")
            print(f"Segments: {results['segment_count']}")
            print(f"Processing speed: {results['speed_ratio']:.1f}x real-time")
            print(f"Output saved to: {results['output_file']}")
            
            # Clean up
            transcriber.close()
            if output_file.exists():
                output_file.unlink()  # Remove example file
                
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Test audio file not found, skipping example 2")

if __name__ == "__main__":
    main()