"""
Utility functions for audio transcription
"""
import os
from pathlib import Path
import subprocess


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


def create_progress_bar(percentage, width=40):
    """Create a visual progress bar string"""
    filled = int(width * percentage / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"