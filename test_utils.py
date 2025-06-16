#!/usr/bin/env python3
"""Unit tests for the utils module"""

import unittest
import tempfile
import os
from pathlib import Path
import sys

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    format_time,
    validate_model_size,
    validate_device,
    validate_compute_type,
    get_file_size_mb,
    estimate_processing_time
)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions"""

    def test_format_time_seconds(self):
        """Test time formatting for seconds"""
        self.assertEqual(format_time(45), "45s")
        self.assertEqual(format_time(0), "0s")
        self.assertEqual(format_time(59.9), "60s")

    def test_format_time_minutes(self):
        """Test time formatting for minutes"""
        self.assertEqual(format_time(60), "1m 0s")
        self.assertEqual(format_time(125), "2m 5s")
        self.assertEqual(format_time(3599), "59m 59s")

    def test_format_time_hours(self):
        """Test time formatting for hours"""
        self.assertEqual(format_time(3600), "1h 0m 0s")
        self.assertEqual(format_time(3725), "1h 2m 5s")
        self.assertEqual(format_time(7325), "2h 2m 5s")

    def test_format_time_float_input(self):
        """Test time formatting with float input"""
        self.assertEqual(format_time(45.7), "46s")
        self.assertEqual(format_time(125.3), "2m 5s")
        self.assertEqual(format_time(3725.8), "1h 2m 6s")

    def test_validate_model_size_valid(self):
        """Test model size validation with valid inputs"""
        valid_models = ["tiny", "base", "small", "medium", "large"]
        for model in valid_models:
            self.assertEqual(validate_model_size(model), model)

    def test_validate_model_size_invalid(self):
        """Test model size validation with invalid inputs"""
        invalid_models = ["invalid", "huge", "mini", ""]
        for model in invalid_models:
            with self.assertRaises(ValueError) as context:
                validate_model_size(model)
            self.assertIn("Invalid model size", str(context.exception))

    def test_validate_device_valid(self):
        """Test device validation with valid inputs"""
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        for device in valid_devices:
            self.assertEqual(validate_device(device), device)

    def test_validate_device_invalid(self):
        """Test device validation with invalid inputs"""
        invalid_devices = ["gpu", "tpu", "invalid", ""]
        for device in invalid_devices:
            with self.assertRaises(ValueError) as context:
                validate_device(device)
            self.assertIn("Invalid device", str(context.exception))

    def test_validate_compute_type_valid(self):
        """Test compute type validation with valid inputs"""
        valid_types = ["auto", "float32", "float16", "int8", "int8_float16"]
        for compute_type in valid_types:
            self.assertEqual(validate_compute_type(compute_type), compute_type)

    def test_validate_compute_type_invalid(self):
        """Test compute type validation with invalid inputs"""
        invalid_types = ["float64", "int16", "invalid", ""]
        for compute_type in invalid_types:
            with self.assertRaises(ValueError) as context:
                validate_compute_type(compute_type)
            self.assertIn("Invalid compute type", str(context.exception))

    def test_get_file_size_mb_existing_file(self):
        """Test file size calculation with existing file"""
        # Create a temporary file with known content
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Write 1024 bytes (1KB)
            temp_file.write(b'x' * 1024)
            temp_file.flush()
            
            # Get file size in MB
            size_mb = get_file_size_mb(temp_file.name)
            
            # Should be approximately 0.001 MB (1KB)
            self.assertAlmostEqual(size_mb, 0.001, places=3)
            
            # Clean up
            os.unlink(temp_file.name)

    def test_get_file_size_mb_nonexistent_file(self):
        """Test file size calculation with non-existent file"""
        size_mb = get_file_size_mb("nonexistent_file.txt")
        self.assertEqual(size_mb, 0.0)

    def test_estimate_processing_time_all_models(self):
        """Test processing time estimation for all model sizes"""
        duration = 180  # 3 minutes
        
        estimates = {}
        for model in ["tiny", "base", "small", "medium", "large"]:
            estimate = estimate_processing_time(duration, model)
            estimates[model] = estimate
            
            # All estimates should be positive
            self.assertGreater(estimate, 0)
            
            # Estimate should be less than the audio duration
            self.assertLess(estimate, duration)
        
        # Larger models should take longer (have smaller speed ratios)
        self.assertLess(estimates["tiny"], estimates["base"])
        self.assertLess(estimates["base"], estimates["small"])
        self.assertLess(estimates["small"], estimates["medium"])
        self.assertLess(estimates["medium"], estimates["large"])

    def test_estimate_processing_time_unknown_model(self):
        """Test processing time estimation with unknown model"""
        duration = 180
        estimate = estimate_processing_time(duration, "unknown")
        
        # Should use default ratio and return reasonable estimate
        self.assertGreater(estimate, 0)
        self.assertLess(estimate, duration)

    def test_estimate_processing_time_edge_cases(self):
        """Test processing time estimation with edge cases"""
        # Zero duration
        self.assertEqual(estimate_processing_time(0, "base"), 0)
        
        # Very short duration
        short_estimate = estimate_processing_time(1, "base")
        self.assertGreater(short_estimate, 0)
        self.assertLess(short_estimate, 1)
        
        # Very long duration
        long_estimate = estimate_processing_time(7200, "base")  # 2 hours
        self.assertGreater(long_estimate, 0)
        self.assertLess(long_estimate, 7200)


class TestUtilsIntegration(unittest.TestCase):
    """Integration tests for utility functions"""

    def test_file_operations_with_real_file(self):
        """Test utility functions with a real test file if available"""
        test_file = Path("assets/nih_3min.mp3")
        
        if test_file.exists():
            # Test file size calculation
            size_mb = get_file_size_mb(str(test_file))
            self.assertGreater(size_mb, 0)
            
            # File should be reasonably sized (between 1MB and 50MB for a 3-min audio)
            self.assertLess(size_mb, 50)
            self.assertGreater(size_mb, 0.5)

    def test_validation_functions_case_sensitivity(self):
        """Test that validation functions handle case properly"""
        # These functions should be case-sensitive
        with self.assertRaises(ValueError):
            validate_model_size("TINY")
        
        with self.assertRaises(ValueError):
            validate_device("CPU")
        
        with self.assertRaises(ValueError):
            validate_compute_type("FLOAT32")


if __name__ == "__main__":
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilsIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)