"""
Unit tests for utils module
"""
import pytest
import tempfile
from pathlib import Path
import subprocess
from unittest.mock import patch, MagicMock

from transcribe.utils import (
    validate_input_file,
    validate_output_path,
    get_audio_duration,
    format_time,
    create_progress_bar
)


class TestValidateInputFile:
    """Test input file validation"""
    
    def test_valid_audio_file(self, mock_audio_file):
        """Test validation of valid audio file"""
        # Rename to have proper audio extension
        audio_file = mock_audio_file.parent / "test.mp3"
        mock_audio_file.rename(audio_file)
        
        result = validate_input_file(audio_file)
        assert result == audio_file
    
    def test_nonexistent_file(self):
        """Test validation of non-existent file"""
        with pytest.raises(FileNotFoundError):
            validate_input_file("nonexistent.mp3")
    
    def test_unsupported_format(self, sample_text_file):
        """Test validation of unsupported file format"""
        with pytest.raises(ValueError, match="Unsupported audio format"):
            validate_input_file(sample_text_file)
    
    def test_directory_instead_of_file(self, temp_dir):
        """Test validation when path is directory"""
        # Create directory with audio extension
        audio_dir = temp_dir / "fake.mp3"
        audio_dir.mkdir()
        
        with pytest.raises(ValueError, match="Path is not a file"):
            validate_input_file(audio_dir)
    
    @pytest.mark.parametrize("extension", [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"])
    def test_supported_formats(self, temp_dir, extension):
        """Test all supported audio formats"""
        audio_file = temp_dir / f"test{extension}"
        audio_file.write_bytes(b"fake audio")
        
        result = validate_input_file(audio_file)
        assert result == audio_file


class TestValidateOutputPath:
    """Test output path validation"""
    
    def test_valid_output_path(self, temp_dir):
        """Test validation of valid output path"""
        output_path = temp_dir / "output.txt"
        result = validate_output_path(output_path)
        assert result == output_path
    
    def test_create_missing_directory(self, temp_dir):
        """Test creation of missing parent directory"""
        output_path = temp_dir / "subdir" / "output.txt"
        result = validate_output_path(output_path)
        
        assert result == output_path
        assert output_path.parent.exists()
    
    def test_nested_directory_creation(self, temp_dir):
        """Test creation of nested directories"""
        output_path = temp_dir / "a" / "b" / "c" / "output.txt"
        result = validate_output_path(output_path)
        
        assert result == output_path
        assert output_path.parent.exists()


class TestGetAudioDuration:
    """Test audio duration extraction"""
    
    @patch('subprocess.run')
    def test_successful_duration_extraction(self, mock_run):
        """Test successful duration extraction"""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="123.456\n"
        )
        
        duration = get_audio_duration("test.mp3")
        assert duration == 123.456
        
        # Verify ffprobe was called correctly
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "ffprobe" in args
        assert "test.mp3" in args
    
    @patch('subprocess.run')
    def test_ffprobe_failure(self, mock_run):
        """Test handling of ffprobe failure"""
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffprobe")
        
        with pytest.raises(RuntimeError, match="Failed to get audio duration"):
            get_audio_duration("test.mp3")
    
    @patch('subprocess.run')
    def test_invalid_duration_output(self, mock_run):
        """Test handling of invalid duration output"""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="invalid\n"
        )
        
        with pytest.raises(RuntimeError, match="Failed to get audio duration"):
            get_audio_duration("test.mp3")


class TestFormatTime:
    """Test time formatting"""
    
    def test_seconds_only(self):
        """Test formatting seconds only"""
        assert format_time(30) == "30s"
        assert format_time(45.7) == "46s"
    
    def test_minutes_and_seconds(self):
        """Test formatting minutes and seconds"""
        assert format_time(90) == "1m 30s"
        assert format_time(125) == "2m 5s"
    
    def test_hours_minutes_seconds(self):
        """Test formatting hours, minutes, and seconds"""
        assert format_time(3665) == "1h 1m 5s"
        assert format_time(7200) == "2h 0m 0s"
        assert format_time(3723) == "1h 2m 3s"
    
    def test_zero_time(self):
        """Test formatting zero time"""
        assert format_time(0) == "0s"


class TestCreateProgressBar:
    """Test progress bar creation"""
    
    def test_zero_percent(self):
        """Test 0% progress bar"""
        bar = create_progress_bar(0, 10)
        assert bar == "[░░░░░░░░░░]"
    
    def test_fifty_percent(self):
        """Test 50% progress bar"""
        bar = create_progress_bar(50, 10)
        assert bar == "[█████░░░░░]"
    
    def test_hundred_percent(self):
        """Test 100% progress bar"""
        bar = create_progress_bar(100, 10)
        assert bar == "[██████████]"
    
    def test_custom_width(self):
        """Test progress bar with custom width"""
        bar = create_progress_bar(25, 4)
        assert bar == "[█░░░]"
    
    def test_partial_progress(self):
        """Test progress bar with partial progress"""
        bar = create_progress_bar(33, 6)
        # 33% of 6 = 1.98, rounded down to 1
        assert bar == "[█░░░░░]"