#!/usr/bin/env python3
"""
Tests for input validation functions
"""
import unittest
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transcribe import validate_input_file, validate_output_path


class TestValidation(unittest.TestCase):
    """Test input validation functions"""
    
    def setUp(self):
        """Set up test files"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test audio files
        self.valid_audio_file = Path(self.temp_dir) / "test.mp3"
        self.valid_audio_file.touch()
        
        self.unsupported_file = Path(self.temp_dir) / "test.txt"
        self.unsupported_file.touch()
    
    def tearDown(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_validate_input_file_exists(self):
        """Test validation of existing audio file"""
        result = validate_input_file(str(self.valid_audio_file))
        self.assertEqual(result, self.valid_audio_file)
    
    def test_validate_input_file_not_found(self):
        """Test validation of non-existent file"""
        with self.assertRaises(FileNotFoundError):
            validate_input_file("nonexistent.mp3")
    
    def test_validate_input_file_unsupported_format(self):
        """Test validation of unsupported file format"""
        with self.assertRaises(ValueError):
            validate_input_file(str(self.unsupported_file))
    
    def test_validate_input_file_directory(self):
        """Test validation when path is a directory"""
        with self.assertRaises(ValueError):
            validate_input_file(self.temp_dir)
    
    def test_validate_output_path_valid(self):
        """Test validation of valid output path"""
        output_path = Path(self.temp_dir) / "output.txt"
        result = validate_output_path(str(output_path))
        self.assertEqual(result, output_path)
    
    def test_validate_output_path_creates_directory(self):
        """Test that output validation creates parent directories"""
        output_path = Path(self.temp_dir) / "subdir" / "output.txt"
        result = validate_output_path(str(output_path))
        self.assertEqual(result, output_path)
        self.assertTrue(output_path.parent.exists())


if __name__ == "__main__":
    unittest.main()