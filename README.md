# Offline Audio Transcriber

A command-line tool for transcribing audio files to text using OpenAI's Whisper technology via the faster-whisper library. Designed for completely offline operation after initial setup.

## Features

- **Offline Operation**: Works without internet connection after initial model download
- **High Performance**: 4x faster than standard Whisper using faster-whisper
- **Parallel Processing**: Automatic chunking and parallel transcription for long files (>30 minutes)
- **GPU Acceleration**: Automatic Metal Performance Shaders (MPS) on macOS Sequoia+
- **Memory Efficient**: Handles large audio files using streaming segments
- **Multiple Formats**: Supports mp3, wav, m4a, ogg, flac, aac, wma
- **Simple Interface**: Single command execution with clear progress feedback
- **Configurable Threading**: Auto-detect CPU cores or manual thread specification

## Installation

1. **Install system dependencies:**
   ```bash
   # macOS (using Homebrew)
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   
   # Windows (using Chocolatey)
   choco install ffmpeg
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Make the script executable:**
   ```bash
   chmod +x transcribe.py
   ```

4. **Pre-download models (optional but recommended):**
   ```python
   python -c "from faster_whisper import WhisperModel; WhisperModel('base')"
   ```

## Usage

### Basic Usage
```bash
./transcribe.py input.mp3 -o output.txt
```

### With Custom Model Size
```bash
./transcribe.py input.wav -o transcript.txt --model small
```

### Available Options

- `input_file`: Path to the audio file to transcribe (required)
- `-o, --output`: Path for the output text file (required)
- `--model`: Whisper model size - tiny, base, small, medium, large (default: base)
- `--threads N`: Number of threads for parallel processing (default: auto-detect)
- `--chunk-size N`: Chunk size in minutes for parallel processing (default: 10)
- `--no-parallel`: Disable parallel processing even for long files

## Model Sizes

| Model  | Size  | Speed | Accuracy |
|--------|-------|-------|----------|
| tiny   | 39MB  | Fastest | Lowest |  
| base   | 74MB  | Fast    | Good   |
| small  | 244MB | Medium  | Better |
| medium | 769MB | Slow    | High   |
| large  | 1550MB| Slowest | Highest|

## Examples

```bash
# Basic transcription
./transcribe.py meeting.mp3 -o meeting_transcript.txt

# High accuracy transcription
./transcribe.py interview.wav -o interview.txt --model large

# Quick transcription
./transcribe.py note.m4a -o note.txt --model tiny

# Parallel processing with custom settings
./transcribe.py long_lecture.mp3 -o lecture.txt --threads 4 --chunk-size 15

# Disable parallel processing for long files
./transcribe.py webinar.wav -o webinar.txt --no-parallel
```

## Output

The tool provides progress feedback during transcription:

```
✓ Input file validated: /path/to/audio.mp3
✓ Output path validated: /path/to/output.txt
Loading Whisper model 'base'...
✓ Model loaded in 2.3s
Transcribing audio: audio.mp3
Detected language: en (probability: 0.99)
Processed 10 segments (15.2s elapsed)
✓ Transcription completed!
  Audio duration: 180.5s
  Processing time: 45.2s
  Speed ratio: 4.0x faster than real-time
  Output saved to: /path/to/output.txt
```

## System Requirements

- **Platform**: macOS Sequoia or higher (for Metal GPU acceleration)
- **Python**: 3.8 or higher
- **Memory**: Varies by model size and audio file length
- **Storage**: Model cache requires 39MB - 1.5GB depending on model

## Error Handling

The tool provides clear error messages for common issues:

- File not found or unreadable
- Unsupported audio formats
- Output directory permissions
- Missing dependencies
- Model loading failures

## Technical Details

- Uses faster-whisper's streaming segments for memory efficiency
- Automatic device selection (Metal on macOS, CUDA on supported systems, CPU fallback)
- Voice Activity Detection (VAD) to skip silence
- **Parallel Processing**: Automatic chunking for files over 30 minutes
  - Uses ffmpeg for reliable audio splitting
  - Thread pool for concurrent chunk processing
  - Chronological transcript reassembly
  - Auto-detects optimal thread count (75% of CPU cores, max 8)
- Models cached in `~/.cache/huggingface/hub/`

## Troubleshooting

**Model fails to load:**
- Ensure you have internet connection for initial model download
- Check available disk space for model cache

**Audio file not recognized:**
- Verify the file format is supported
- Check file permissions and path

**Slow performance:**
- Try a smaller model size (tiny, base, small)
- Ensure Metal GPU acceleration is available on macOS

## Testing

Run the test suite to verify functionality:

```bash
python -m pytest tests/ -v
```# py_transcribe
