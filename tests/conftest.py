"""
Pytest configuration and fixtures for transcribe tests
"""
import pytest
import tempfile
from pathlib import Path
import subprocess
import os


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file for testing"""
    text_file = temp_dir / "sample.txt"
    text_file.write_text("This is a test file with some content.")
    return text_file


@pytest.fixture
def test_audio_file():
    """Path to the test audio file (if it exists)"""
    audio_path = Path("assets/testing speech audio file.m4a")
    if audio_path.exists():
        return audio_path
    return None


@pytest.fixture
def mock_audio_file(temp_dir):
    """Create a mock audio file for testing (empty file with audio extension)"""
    audio_file = temp_dir / "mock_audio.mp3"
    audio_file.write_bytes(b"fake audio content")
    return audio_file


@pytest.fixture(scope="session")
def has_faster_whisper():
    """Check if faster-whisper is available"""
    try:
        import faster_whisper
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session") 
def has_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


@pytest.fixture
def audio_duration_mock(monkeypatch):
    """Mock get_audio_duration to return a fixed value"""
    def mock_duration(file_path):
        return 11.0  # 11 seconds
    
    monkeypatch.setattr("transcribe.utils.get_audio_duration", mock_duration)
    return mock_duration