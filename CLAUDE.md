# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an offline audio transcription CLI tool using faster-whisper. The project implements a single-file Python script that transcribes audio files completely offline after initial setup.

## Core Architecture

- **Single-file implementation**: `transcribe.py` contains the entire CLI application
- **Library**: Uses faster-whisper (4x faster than standard whisper) 
- **Target platform**: macOS Sequoia+ with Metal GPU acceleration
- **Offline-first**: No internet required after initial model download

## Command Interface

```bash
./transcribe <input_file> -o <output_file> [--model model_size]
```

Required arguments:
- `input_file`: Path to audio file
- `-o, --output`: Output text file path

Optional arguments:
- `--model`: Whisper model size (tiny/base/small/medium/large), defaults to 'base'

## Key Implementation Requirements

- **GPU Acceleration**: Automatic Metal Performance Shaders (MPS) utilization on macOS
- **Memory Efficiency**: Use faster-whisper's streaming segments for large files
- **Single-pass Processing**: Avoid chunking complexity
- **Model Caching**: Models auto-download to `~/.cache/huggingface/hub/` on first use
- **Error Handling**: Validate input files, output directories, and provide clear error messages

## Dependencies

Core dependency: `faster-whisper>=0.9.0`

Install with: `pip install faster-whisper`

## Testing Audio Files

The tool should handle common audio formats (mp3, wav, m4a) and process files of any reasonable size efficiently without memory issues.

## Architecture Documentation

- Document all architecture decisions and changes in `docs/dev/*.md`
- Refer to docs/dev/ documentation for an architecture overview and summary

## Development Workflow Memories

- add user docs after adding a feature in docs/*.md
- take it step by step. Generate tests for planned tasks. Use the tests to iterate during each step. Code like a human that iteratively builds up functionality and tests piece by piece.
- create a plan and present it and/or tasks to get approval before writing or modifying files.
- use the assets/testing-file for testing with real files