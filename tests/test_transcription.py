#!/usr/bin/env python3
"""
Tests for transcription functionality
"""
import unittest
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from transcribe import transcribe_audio


class TestTranscription(unittest.TestCase):
    """Test transcription functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.input_file = Path(self.temp_dir) / "test_audio.mp3"
        self.output_file = Path(self.temp_dir) / "output.txt"
        
        # Create a dummy input file
        self.input_file.touch()
    
    def tearDown(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_transcribe_audio_success(self):
        """Test successful audio transcription"""
        # Mock the model and its transcribe method
        mock_model = MagicMock()
        
        # Mock segment objects
        mock_segment1 = MagicMock()
        mock_segment1.text = "Hello world"
        mock_segment1.end = 5.0
        mock_segment2 = MagicMock()
        mock_segment2.text = "This is a test"
        mock_segment2.end = 10.0
        
        # Mock transcription info
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_info.duration = 10.0
        
        # Configure the mock to return segments and info
        mock_model.transcribe.return_value = ([mock_segment1, mock_segment2], mock_info)
        
        # Run transcription
        transcribe_audio(mock_model, self.input_file, self.output_file)
        
        # Verify the model was called correctly
        mock_model.transcribe.assert_called_once_with(
            str(self.input_file),
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=1000)
        )
        
        # Verify output file was created with correct content
        self.assertTrue(self.output_file.exists())
        with open(self.output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("Hello world", content)
            self.assertIn("This is a test", content)
    
    def test_transcribe_audio_empty_segments(self):
        """Test transcription with no segments"""
        mock_model = MagicMock()
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_info.duration = 5.0
        
        # Empty segments
        mock_model.transcribe.return_value = ([], mock_info)
        
        # Run transcription
        transcribe_audio(mock_model, self.input_file, self.output_file)
        
        # Verify output file was created but is empty
        self.assertTrue(self.output_file.exists())
        with open(self.output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertEqual(content, "")


if __name__ == "__main__":
    unittest.main()