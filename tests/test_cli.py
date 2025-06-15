#!/usr/bin/env python3
"""
Tests for CLI argument parsing
"""
import unittest
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transcribe import parse_arguments


class TestCLI(unittest.TestCase):
    """Test CLI argument parsing"""
    
    def test_basic_arguments(self):
        """Test basic required arguments"""
        sys.argv = ["transcribe", "input.mp3", "-o", "output.txt"]
        args = parse_arguments()
        
        self.assertEqual(args.input_file, "input.mp3")
        self.assertEqual(args.output, "output.txt")
        self.assertEqual(args.model, "base")  # default
    
    def test_model_selection(self):
        """Test model size selection"""
        sys.argv = ["transcribe", "input.wav", "-o", "output.txt", "--model", "small"]
        args = parse_arguments()
        
        self.assertEqual(args.model, "small")
    
    def test_long_form_output(self):
        """Test long form output argument"""
        sys.argv = ["transcribe", "input.m4a", "--output", "transcript.txt"]
        args = parse_arguments()
        
        self.assertEqual(args.output, "transcript.txt")


if __name__ == "__main__":
    unittest.main()