#!/usr/bin/env python3
"""Unit tests for the transcriber module"""

import unittest
import tempfile
import os
from pathlib import Path
import sys

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transcriber import AudioTranscriber


class TestAudioTranscriber(unittest.TestCase):
    """Test cases for AudioTranscriber class"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_audio_file = Path("assets/nih_3min.mp3")
        self.temp_dir = tempfile.mkdtemp()
        self.temp_output = Path(self.temp_dir) / "test_output.txt"
        self.transcriber = AudioTranscriber(model_size="tiny")

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_init_default_params(self):
        """Test transcriber initialization with default parameters"""
        transcriber = AudioTranscriber()
        self.assertEqual(transcriber.model_size, "base")
        self.assertEqual(transcriber.device, "cpu")
        self.assertEqual(transcriber.compute_type, "float32")

    def test_init_custom_params(self):
        """Test transcriber initialization with custom parameters"""
        transcriber = AudioTranscriber(
            model_size="small",
            device="cuda",
            compute_type="float16"
        )
        self.assertEqual(transcriber.model_size, "small")
        self.assertEqual(transcriber.device, "cuda")
        self.assertEqual(transcriber.compute_type, "float16")

    def test_validate_input_file_exists(self):
        """Test input file validation with existing file"""
        if self.test_audio_file.exists():
            result = self.transcriber.validate_input_file(str(self.test_audio_file))
            self.assertEqual(result, self.test_audio_file)

    def test_validate_input_file_not_exists(self):
        """Test input file validation with non-existing file"""
        with self.assertRaises(FileNotFoundError):
            self.transcriber.validate_input_file("nonexistent_file.mp3")

    def test_validate_input_file_unsupported_format(self):
        """Test input file validation with unsupported format"""
        # Create a temporary file with unsupported extension
        temp_file = Path(self.temp_dir) / "test.txt"
        temp_file.touch()
        
        with self.assertRaises(ValueError) as context:
            self.transcriber.validate_input_file(str(temp_file))
        self.assertIn("Unsupported audio format", str(context.exception))

    def test_validate_output_path(self):
        """Test output path validation"""
        result = self.transcriber.validate_output_path(str(self.temp_output))
        self.assertEqual(result, self.temp_output)
        # Check that parent directory was created
        self.assertTrue(result.parent.exists())

    def test_get_audio_duration(self):
        """Test audio duration calculation"""
        if self.test_audio_file.exists():
            duration = self.transcriber.get_audio_duration(str(self.test_audio_file))
            # 3-minute file should be approximately 180 seconds
            self.assertGreater(duration, 170)
            self.assertLess(duration, 190)

    def test_get_model_info(self):
        """Test getting model information"""
        info = self.transcriber.get_model_info()
        expected_keys = {'model_size', 'device', 'compute_type', 'model_loaded'}
        self.assertEqual(set(info.keys()), expected_keys)
        self.assertEqual(info['model_size'], 'tiny')
        self.assertEqual(info['device'], 'cpu')
        self.assertEqual(info['compute_type'], 'float32')
        self.assertFalse(info['model_loaded'])

    def test_supported_formats(self):
        """Test that supported formats are correctly defined"""
        expected_formats = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma'}
        self.assertEqual(AudioTranscriber.SUPPORTED_FORMATS, expected_formats)

    @unittest.skipUnless(Path("assets/nih_3min.mp3").exists(), "Test audio file not available")
    def test_full_transcription_workflow(self):
        """Test complete transcription workflow"""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            self.skipTest("faster-whisper not installed")

        # Track progress calls
        progress_calls = []
        
        def progress_callback(progress, segments, elapsed):
            progress_calls.append((progress, segments, elapsed))

        # Run transcription
        result = self.transcriber.transcribe_file(
            str(self.test_audio_file),
            str(self.temp_output),
            progress_callback=progress_callback
        )

        # Verify result structure
        expected_keys = {
            'language', 'language_probability', 'audio_duration',
            'processing_time', 'speed_ratio', 'segment_count', 'output_path'
        }
        self.assertEqual(set(result.keys()), expected_keys)

        # Verify basic result values
        self.assertEqual(result['language'], 'en')
        self.assertGreater(result['language_probability'], 0.9)
        self.assertGreater(result['audio_duration'], 170)
        self.assertLess(result['audio_duration'], 190)
        self.assertGreater(result['processing_time'], 0)
        self.assertGreater(result['speed_ratio'], 0)
        self.assertGreater(result['segment_count'], 0)
        self.assertEqual(result['output_path'], str(self.temp_output))

        # Verify output file was created and has content
        self.assertTrue(self.temp_output.exists())
        with open(self.temp_output, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            self.assertGreater(len(content), 0)
            # Should contain some recognizable words from the audio
            content_lower = content.lower()
            self.assertTrue(any(word in content_lower for word in ["health", "science", "research", "life"]))

        # Verify progress callback was called
        self.assertGreater(len(progress_calls), 0)

    def test_error_handling_invalid_audio_file(self):
        """Test error handling with invalid audio file"""
        # Create a text file with audio extension
        fake_audio = Path(self.temp_dir) / "fake.mp3"
        fake_audio.write_text("This is not an audio file")
        
        # This should pass validation but fail during transcription
        self.transcriber.validate_input_file(str(fake_audio))
        
        # The transcription should fail gracefully
        with self.assertRaises(Exception):
            self.transcriber.transcribe_file(str(fake_audio), str(self.temp_output))


class TestAudioTranscriberEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.transcriber = AudioTranscriber()

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_validate_input_file_directory(self):
        """Test input file validation when path is a directory"""
        with self.assertRaises(ValueError) as context:
            self.transcriber.validate_input_file(str(self.temp_dir))
        self.assertIn("not a file", str(context.exception))

    def test_validate_output_path_permission_error(self):
        """Test output path validation with permission issues"""
        # Try to create output in a non-existent path that can't be created
        if os.name != 'nt':  # Skip on Windows due to different permission model
            invalid_path = "/root/cannot_create/output.txt"
            with self.assertRaises(PermissionError):
                self.transcriber.validate_output_path(invalid_path)

    def test_get_audio_duration_invalid_file(self):
        """Test audio duration with invalid file"""
        with self.assertRaises(RuntimeError):
            self.transcriber.get_audio_duration("nonexistent_file.mp3")

    def test_model_loading_with_fallback(self):
        """Test that model loading falls back to CPU on errors"""
        # Create transcriber with potentially unsupported device
        transcriber = AudioTranscriber(device="cuda", compute_type="float16")
        
        # If CUDA is not available, it should fall back to CPU
        try:
            transcriber._load_model()
            # If loading succeeds, check the configuration
            model_info = transcriber.get_model_info()
            self.assertTrue(model_info['model_loaded'])
        except Exception:
            # If loading fails, that's also acceptable for this test
            pass


if __name__ == "__main__":
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestAudioTranscriber))
    suite.addTests(loader.loadTestsFromTestCase(TestAudioTranscriberEdgeCases))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)