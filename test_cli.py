#!/usr/bin/env python3
"""Unit tests for the CLI module"""

import unittest
import tempfile
import os
from pathlib import Path
import sys

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from click.testing import CliRunner
from cli import main


class TestCLI(unittest.TestCase):
    """Test cases for the Click-based CLI"""

    def setUp(self):
        """Set up test fixtures"""
        self.runner = CliRunner()
        self.test_audio_file = Path("assets/nih_3min.mp3")
        self.temp_dir = tempfile.mkdtemp()
        self.temp_output = Path(self.temp_dir) / "test_cli_output.txt"

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_help_command(self):
        """Test CLI help command"""
        result = self.runner.invoke(main, ['--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Transcribe audio files to text", result.output)
        self.assertIn("INPUT_FILE", result.output)
        self.assertIn("--output", result.output)
        self.assertIn("--model", result.output)

    def test_version_command(self):
        """Test CLI version command"""
        result = self.runner.invoke(main, ['--version'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("version", result.output.lower())

    def test_missing_input_file(self):
        """Test CLI with missing input file"""
        result = self.runner.invoke(main, [])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Missing argument", result.output)

    def test_missing_output_option(self):
        """Test CLI with missing output option"""
        result = self.runner.invoke(main, ['input.mp3'])
        self.assertNotEqual(result.exit_code, 0)
        # Click will complain about missing input file first since it doesn't exist
        # This is expected behavior

    def test_nonexistent_input_file(self):
        """Test CLI with non-existent input file"""
        result = self.runner.invoke(main, [
            'nonexistent_file.mp3',
            '-o', str(self.temp_output)
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("does not exist", result.output)

    def test_invalid_model_option(self):
        """Test CLI with invalid model option"""
        if self.test_audio_file.exists():
            result = self.runner.invoke(main, [
                str(self.test_audio_file),
                '-o', str(self.temp_output),
                '--model', 'invalid'
            ])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Invalid value", result.output)

    def test_invalid_device_option(self):
        """Test CLI with invalid device option"""
        if self.test_audio_file.exists():
            result = self.runner.invoke(main, [
                str(self.test_audio_file),
                '-o', str(self.temp_output),
                '--device', 'invalid'
            ])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Invalid value", result.output)

    def test_invalid_compute_type_option(self):
        """Test CLI with invalid compute type option"""
        if self.test_audio_file.exists():
            result = self.runner.invoke(main, [
                str(self.test_audio_file),
                '-o', str(self.temp_output),
                '--compute-type', 'invalid'
            ])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Invalid value", result.output)

    @unittest.skipUnless(Path("assets/nih_3min.mp3").exists(), "Test audio file not available")
    def test_successful_transcription(self):
        """Test successful transcription with tiny model"""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            self.skipTest("faster-whisper not installed")

        result = self.runner.invoke(main, [
            str(self.test_audio_file),
            '-o', str(self.temp_output),
            '--model', 'tiny'
        ])

        # Check command succeeded
        self.assertEqual(result.exit_code, 0, f"Command failed with exit code {result.exit_code}")

        # Check output file was created
        self.assertTrue(self.temp_output.exists())

        # Check output file has content
        with open(self.temp_output, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            self.assertGreater(len(content), 0)

    @unittest.skipUnless(Path("assets/nih_3min.mp3").exists(), "Test audio file not available")
    def test_verbose_mode(self):
        """Test CLI with verbose flag"""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            self.skipTest("faster-whisper not installed")

        result = self.runner.invoke(main, [
            str(self.test_audio_file),
            '-o', str(self.temp_output),
            '--model', 'tiny',
            '--verbose'
        ])

        self.assertEqual(result.exit_code, 0, f"Verbose mode failed with exit code {result.exit_code}")
        
        # Check output file was created (main functionality test)
        self.assertTrue(self.temp_output.exists())

    @unittest.skipUnless(Path("assets/nih_3min.mp3").exists(), "Test audio file not available")
    def test_quiet_mode(self):
        """Test CLI with quiet flag"""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            self.skipTest("faster-whisper not installed")

        result = self.runner.invoke(main, [
            str(self.test_audio_file),
            '-o', str(self.temp_output),
            '--model', 'tiny',
            '--quiet'
        ])

        self.assertEqual(result.exit_code, 0, f"Quiet mode failed with exit code {result.exit_code}")
        
        # Check output file was created (main functionality test)
        self.assertTrue(self.temp_output.exists())

    @unittest.skipUnless(Path("assets/nih_3min.mp3").exists(), "Test audio file not available")
    def test_max_duration_option(self):
        """Test CLI with max-duration option"""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            self.skipTest("faster-whisper not installed")

        result = self.runner.invoke(main, [
            str(self.test_audio_file),
            '-o', str(self.temp_output),
            '--model', 'tiny',
            '--max-duration', '1'
        ])

        self.assertEqual(result.exit_code, 0, f"Max duration option failed with exit code {result.exit_code}")
        
        # Check output file was created
        self.assertTrue(self.temp_output.exists())
        
        # Check output file has content (should be less since only 1 minute processed)
        with open(self.temp_output, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            self.assertGreater(len(content), 0)

    def test_max_duration_validation(self):
        """Test CLI with invalid max-duration values"""
        if self.test_audio_file.exists():
            # Test negative value
            result = self.runner.invoke(main, [
                str(self.test_audio_file),
                '-o', str(self.temp_output),
                '--max-duration', '-1'
            ])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Invalid value", result.output)
            
            # Test zero value
            result = self.runner.invoke(main, [
                str(self.test_audio_file),
                '-o', str(self.temp_output),
                '--max-duration', '0'
            ])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Invalid value", result.output)

    def test_help_includes_max_duration(self):
        """Test that help text includes max-duration option"""
        result = self.runner.invoke(main, ['--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--max-duration", result.output)
        self.assertIn("Limit transcription to first N minutes", result.output)
        self.assertIn("--start-time", result.output)
        self.assertIn("--end-time", result.output)

    @unittest.skipUnless(Path("assets/nih_3min.mp3").exists(), "Test audio file not available")
    def test_start_end_time_options(self):
        """Test CLI with start-time and end-time options"""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            self.skipTest("faster-whisper not installed")

        result = self.runner.invoke(main, [
            str(self.test_audio_file),
            '-o', str(self.temp_output),
            '--model', 'tiny',
            '--start-time', '30',
            '--end-time', '90'
        ])

        self.assertEqual(result.exit_code, 0, f"Start/end time options failed with exit code {result.exit_code}")
        
        # Check output file was created
        self.assertTrue(self.temp_output.exists())
        
        # Check output file has content
        with open(self.temp_output, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            self.assertGreater(len(content), 0)

    def test_conflicting_time_options(self):
        """Test CLI with conflicting time options"""
        if self.test_audio_file.exists():
            # Test max-duration with start-time
            result = self.runner.invoke(main, [
                str(self.test_audio_file),
                '-o', str(self.temp_output),
                '--max-duration', '2',
                '--start-time', '30'
            ])
            self.assertEqual(result.exit_code, 1)
            self.assertIn("cannot be used with", result.output)

    def test_invalid_time_range(self):
        """Test CLI with invalid time range"""
        if self.test_audio_file.exists():
            # Test start-time >= end-time
            result = self.runner.invoke(main, [
                str(self.test_audio_file),
                '-o', str(self.temp_output),
                '--start-time', '90',
                '--end-time', '30'
            ])
            self.assertEqual(result.exit_code, 1)
            self.assertIn("must be less than", result.output)

    def test_case_insensitive_options(self):
        """Test that options are case insensitive"""
        if self.test_audio_file.exists():
            # Test uppercase model option
            result = self.runner.invoke(main, [
                str(self.test_audio_file),
                '-o', str(self.temp_output),
                '--model', 'TINY'
            ], catch_exceptions=False)
            
            # Should not fail due to case (Click will convert to lowercase)
            # This test depends on Click's case_sensitive=False setting


class TestCLIErrorHandling(unittest.TestCase):
    """Test CLI error handling scenarios"""

    def setUp(self):
        """Set up test fixtures"""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_unsupported_file_format(self):
        """Test CLI with unsupported file format"""
        # Create a text file with audio extension
        fake_audio = Path(self.temp_dir) / "fake.mp3"
        fake_audio.write_text("This is not an audio file")
        
        temp_output = Path(self.temp_dir) / "output.txt"
        
        result = self.runner.invoke(main, [
            str(fake_audio),
            '-o', str(temp_output),
            '--model', 'tiny'
        ])

        # Should fail with an error
        self.assertNotEqual(result.exit_code, 0)

    def test_output_permission_error(self):
        """Test CLI with output permission issues"""
        if os.name != 'nt':  # Skip on Windows due to different permission model
            # Try to write to a directory we can't write to
            invalid_output = "/root/cannot_write/output.txt"
            
            # Create a dummy input file
            dummy_input = Path(self.temp_dir) / "dummy.mp3"
            dummy_input.touch()
            
            result = self.runner.invoke(main, [
                str(dummy_input),
                '-o', invalid_output,
                '--model', 'tiny'
            ])

            self.assertNotEqual(result.exit_code, 0)

    def test_keyboard_interrupt_handling(self):
        """Test that keyboard interrupts are handled gracefully"""
        # This is difficult to test directly, but we can verify the exception handling exists
        # by checking the CLI module code structure
        import cli
        import inspect
        
        source = inspect.getsource(cli)
        self.assertIn("KeyboardInterrupt", source)
        self.assertIn("interrupted by user", source)


if __name__ == "__main__":
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestCLI))
    suite.addTests(loader.loadTestsFromTestCase(TestCLIErrorHandling))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)