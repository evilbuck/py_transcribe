# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an offline audio transcription CLI tool using faster-whisper. The project implements a modular Python application that transcribes audio files completely offline after initial setup.

## Core Architecture

- **Modular design**: Separated CLI interface from core transcription logic
- **CLI Framework**: Click-based command line interface with Rich formatting
- **Core Library**: `AudioTranscriber` class for transcription functionality
- **Target platform**: Cross-platform with optimized CPU processing
- **Offline-first**: No internet required after initial model download

## File Structure

- `transcribe.py`: Main entry point (delegates to CLI)
- `cli.py`: Click-based command line interface with Rich formatting
- `transcriber.py`: Core `AudioTranscriber` class and transcription logic
- `utils.py`: Utility functions for validation and formatting
- `test_*.py`: Comprehensive unit tests for all modules

## Command Interface

```bash
python transcribe.py <input_file> -o <output_file> [OPTIONS]
```

Required arguments:

- `input_file`: Path to audio file
- `-o, --output`: Output text file path

Optional arguments:

- `--model`: Whisper model size (tiny/base/small/medium/large), defaults to 'base'
- `--device`: Force specific device (auto/cpu/cuda/mps), defaults to 'auto'
- `--compute-type`: Force specific compute type (auto/float32/float16/int8/int8_float16), defaults to 'auto'
- `--verbose, -v`: Show detailed system information and optimization hints
- `--quiet, -q`: Suppress progress display and non-essential output
- `--version`: Show version information
- `--help`: Show help message

## Key Implementation Requirements

- **Modular Design**: Clean separation of CLI interface and core transcription logic
- **Rich UI**: Beautiful progress bars and colored output using Rich library
- **Cross-platform**: Optimized CPU processing for maximum compatibility
- **Memory Efficiency**: Use faster-whisper's streaming segments for large files
- **Model Caching**: Models auto-download to `~/.cache/huggingface/hub/` on first use
- **Comprehensive Testing**: Full unit test coverage for all modules
- **Error Handling**: Validate input files, output directories, and provide clear error messages

## Dependencies

Core dependencies:

- `faster-whisper>=1.0.0`: Core transcription engine
- `click>=8.0.0`: Command line interface framework
- `rich>=13.0.0`: Beautiful terminal output and progress bars

Install with: `pip install faster-whisper click rich`

## Testing Audio Files

The tool should handle common audio formats (mp3, wav, m4a) and process files of any reasonable size efficiently without memory issues.

## Architecture Documentation

- Document all architecture decisions and changes in `docs/dev/*.md`
- Refer to docs/dev/ documentation for an architecture overview and summary

## Testing

Run comprehensive tests:

```bash
python test_all.py
```

Individual test modules:

```bash
python test_utils.py      # Utility function tests
python test_transcriber.py # Core transcription tests
python test_cli.py        # CLI interface tests
```

## Development Workflow Memories

- add user docs after adding a feature in docs/\*.md
- take it step by step. Generate tests for planned tasks. Use the tests to iterate during each step. Code like a human that iteratively builds up functionality and tests piece by piece.
- create a plan and present it and/or tasks to get approval before writing or modifying files.
- use the assets/testing-file for testing with real files
- Always test every feature with unit tests using unittest. Use the tests to validate the work. Iterate until tests pass. Verify all tests at completion of a task. Always be green unless there is a known, purposeful breakage that can't be avoided until the following step
- After completion of a task, generate a descriptive commit message and commit the changes.

## Code Structure

The modular architecture provides:

- **Separation of concerns**: CLI logic separate from transcription logic
- **Testability**: Each module can be tested independently
- **Extensibility**: Easy to add new CLI features or alternative interfaces
- **Maintainability**: Clean, focused modules with clear responsibilities

## Debugging and Logging

- use a configuration flag to debug
- utilize logger library and log levels to control debug
- do not add debug logging that needs to be removed later. use the logger and logging levels

