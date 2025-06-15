# Offline Audio Transcriber - Product Requirements Document

## Overview

Build a command-line audio transcription tool that processes audio files completely offline using OpenAI's Whisper technology via the faster-whisper library.

## Core Requirements

### Functional Requirements

- **Primary Command**: `./transcribe <input_file> -o <output_file>`
- **Offline Operation**: Must work without internet connection after initial setup
- **Large File Support**: Handle audio files of any reasonable size efficiently
- **Audio Format Support**: Accept common audio formats (mp3, wav, m4a, etc.)
- **Text Output**: Generate clean, readable transcription text files

### Technical Requirements

- **Language**: Python 3.8+
- **Core Library**: faster-whisper (4x faster than standard whisper)
- **Target Platform**: macOS Sequoia or higher
- **GPU Acceleration**: Automatic Metal Performance Shaders (MPS) utilization
- **Dependencies**: Minimal - avoid ffmpeg unless absolutely necessary

### Performance Requirements

- **Speed**: Prioritize transcription speed over complex optimizations
- **Memory**: Efficient memory usage for large files
- **Simplicity**: Clean, maintainable codebase with minimal complexity

## User Interface

### Command Line Interface

```bash
./transcribe input.mp3 -o output.txt [options]
```

### Required Arguments

- `input_file`: Path to audio file to transcribe
- `-o, --output`: Path for output text file

### Optional Arguments

- `--model`: Whisper model size (tiny/base/small/medium/large) [default: base]

### Expected Output

- Progress indication during transcription
- Success confirmation with duration and output path
- Error messages for invalid inputs or failures

## Implementation Specifications

### Project Structure

```
transcribe
├── transcribe.py          # Main CLI script
├── requirements.txt       # Dependencies
└── README.md             # Setup and usage instructions
```

### Core Dependencies

```
faster-whisper>=0.9.0
```

### Model Management

- Models automatically downloaded on first use
- Local caching in `~/.cache/huggingface/hub/`
- No manual model management required

### Error Handling

- Validate input file exists and is readable
- Check output directory is writable
- Graceful handling of unsupported audio formats
- Clear error messages for all failure cases

## Setup Process

### Installation

1. Install Python dependencies: `pip install faster-whisper`
2. Pre-download models while online (optional but recommended)
3. Make script executable: `chmod +x transcribe.py`

### Model Pre-download (Offline Preparation)

```python
# Run once while online to cache models
from faster_whisper import WhisperModel
WhisperModel('base')  # Downloads and caches base model
```

## Acceptance Criteria

### Must Have

- [x] Transcribe audio files to text completely offline
- [x] Support command line interface as specified
- [x] Handle large audio files without memory issues
- [x] Automatic GPU acceleration on macOS Sequoia
- [x] Simple single-file implementation
- [x] Clear error messages and progress feedback

### Success Metrics

- **Performance**: Process audio at least 4x faster than standard Whisper
- **Reliability**: Handle 99%+ of common audio formats successfully
- **Usability**: Single command execution with intuitive arguments
- **Offline**: Zero network requirements after initial setup

## Technical Constraints

- **Platform**: macOS Sequoia+ (Metal GPU support)
- **Offline**: No internet dependency post-installation
- **Simplicity**: Avoid complex threading, chunking, or preprocessing
- **Dependencies**: Minimize external tool requirements

## Non-Requirements

- Real-time transcription
- Multi-file batch processing
- Audio preprocessing/enhancement
- Custom model training
- Web interface or GUI
- Configuration files or complex settings

## Implementation Notes

- Use faster-whisper's streaming segments for memory efficiency
- Leverage automatic device selection (Metal on macOS)
- Single-pass processing without chunking complexity
- Direct file I/O without intermediate processing steps
