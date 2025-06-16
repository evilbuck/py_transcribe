#!/usr/bin/env python3
"""
Offline Audio Transcriber - Command-line tool for transcribing audio files using faster-whisper
"""
import argparse
import sys
import os
from pathlib import Path
import time
import subprocess


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files to text using faster-whisper (offline)",
        prog="transcribe"
    )

    parser.add_argument(
        "input_file",
        help="Path to the audio file to transcribe"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path for the output text file"
    )

    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size to use (default: base)"
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Force specific device for inference (default: auto-detect)"
    )

    parser.add_argument(
        "--compute-type",
        choices=["auto", "float32", "float16", "int8", "int8_float16"],
        default="auto",
        help="Force specific compute type (default: auto-detect based on device)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed system information and optimization hints"
    )

    return parser.parse_args()


def validate_input_file(file_path):
    """Validate input audio file"""
    supported_formats = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma'}

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    if path.suffix.lower() not in supported_formats:
        raise ValueError(f"Unsupported audio format: {path.suffix}. Supported: {', '.join(supported_formats)}")

    return path


def validate_output_path(output_path):
    """Validate output file path and directory"""
    path = Path(output_path)

    # Check if parent directory exists and is writable
    parent_dir = path.parent
    if not parent_dir.exists():
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise PermissionError(f"Cannot create output directory: {parent_dir}. {e}")

    if not os.access(parent_dir, os.W_OK):
        raise PermissionError(f"Output directory is not writable: {parent_dir}")

    return path


def get_audio_duration(file_path):
    """Get audio file duration in seconds using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(file_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        return duration
    except (subprocess.CalledProcessError, ValueError) as e:
        raise RuntimeError(f"Failed to get audio duration: {e}")


def get_optimal_device(override=None):
    """Determine optimal device for faster-whisper"""
    if override and override != "auto":
        return override
    
    # Simple device detection - prefer CPU for reliability
    return "cpu"


def get_optimal_compute_type(override=None):
    """Determine optimal compute type to avoid warnings"""
    if override and override != "auto":
        return override
    
    # Use float32 for CPU reliability
    return "float32"


def format_time(seconds):
    """Format seconds into human-readable time string"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins:.0f}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:.0f}h {mins:.0f}m {secs:.0f}s"


def transcribe_audio(model, input_path, output_path):
    """Transcribe audio file using Whisper model with streaming segments"""
    print(f"Transcribing audio: {input_path.name}")
    start_time = time.time()

    # Use simple, reliable parameters
    segments, info = model.transcribe(
        str(input_path),
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=1000),
        temperature=0.0
    )

    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    audio_duration = info.duration
    print(f"Audio duration: {format_time(audio_duration)}")
    print("Processing segments...")

    # Write transcription to file with progress tracking
    with open(output_path, 'w', encoding='utf-8') as f:
        segment_count = 0
        last_segment_end = 0
        last_progress_update = time.time()

        for segment in segments:
            # Write segment text
            f.write(segment.text.strip())
            f.write('\n')

            segment_count += 1
            last_segment_end = segment.end

            # Update progress every 5 seconds to avoid overwhelming output
            current_time = time.time()
            if current_time - last_progress_update >= 5.0:
                progress_percentage = min((last_segment_end / audio_duration) * 100, 100)
                elapsed_time = current_time - start_time

                print(f"Progress: {progress_percentage:.1f}% - {segment_count} segments processed - Elapsed: {format_time(elapsed_time)}")
                last_progress_update = current_time

    total_time = time.time() - start_time
    speed_ratio = audio_duration / total_time if total_time > 0 else 0

    print(f"✓ Transcription completed!")
    print(f"  Audio duration: {format_time(audio_duration)}")
    print(f"  Processing time: {format_time(total_time)}")
    print(f"  Speed ratio: {speed_ratio:.1f}x faster than real-time")
    print(f"  Total segments: {segment_count}")
    print(f"  Output saved to: {output_path}")


def main():
    """Main entry point"""
    try:
        args = parse_arguments()

        # Validate input file
        input_path = validate_input_file(args.input_file)
        print(f"✓ Input file validated: {input_path}")

        # Validate output path
        output_path = validate_output_path(args.output)
        print(f"✓ Output path validated: {output_path}")

        # Check audio duration
        duration = get_audio_duration(input_path)
        print(f"✓ Audio duration: {format_time(duration)}")

        # Use simple, reliable device settings
        device = get_optimal_device(args.device)
        compute_type = get_optimal_compute_type(args.compute_type)

        if args.verbose:
            print(f"🖥️  Using device: {device}")
            print(f"⚙️  Using compute type: {compute_type}")

        # Initialize Whisper model
        print(f"Loading Whisper model '{args.model}' (device: {device}, compute: {compute_type})...")
        start_time = time.time()

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("faster-whisper is not installed. Run: pip install faster-whisper")

        try:
            model = WhisperModel(args.model, device=device, compute_type=compute_type)
        except Exception as e:
            print(f"⚠️  Falling back to CPU due to: {e}")
            model = WhisperModel(args.model, device="cpu", compute_type="float32")

        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.1f}s")

        # Transcribe audio file
        transcribe_audio(model, input_path, output_path)

        return 0

    except (FileNotFoundError, ValueError, PermissionError, ImportError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nTranscription interrupted by user", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())