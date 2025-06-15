#!/usr/bin/env python3
"""
Tests for Whisper model initialization
"""
import unittest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from transcribe import initialize_whisper_model


class TestModelInitialization(unittest.TestCase):
    """Test Whisper model initialization"""
    
    @patch('transcribe.time')
    @patch('faster_whisper.WhisperModel')
    def test_initialize_whisper_model_success(self, mock_whisper_model, mock_time):
        """Test successful model initialization"""
        # Mock time.time() to return consistent values
        mock_time.time.side_effect = [0.0, 1.5]  # start and end times
        
        # Mock the WhisperModel class
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model
        
        # Test model initialization
        result = initialize_whisper_model("base")
        
        # Verify model was created with correct parameters
        mock_whisper_model.assert_called_once_with("base", device="auto", compute_type="auto")
        self.assertEqual(result, mock_model)
    
    @patch('builtins.__import__', side_effect=ImportError("No module named 'faster_whisper'"))
    def test_initialize_whisper_model_import_error(self, mock_import):
        """Test handling of missing faster-whisper dependency"""
        with self.assertRaises(ImportError) as context:
            initialize_whisper_model("base")
        
        self.assertIn("faster-whisper is not installed", str(context.exception))


if __name__ == "__main__":
    unittest.main()