#!/usr/bin/env python3
"""
Offline Audio Transcriber - Command-line tool for transcribing audio files using faster-whisper

This is the main entry point that delegates to the Click-based CLI interface.
"""
import sys

if __name__ == "__main__":
    from cli import main
    sys.exit(main())