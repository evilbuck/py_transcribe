#!/usr/bin/env python3
"""
Test script for utils module
"""
import sys
import tempfile
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, '.')

from transcribe.utils import validate_input_file, validate_output_path, format_time, create_progress_bar


def test_format_time():
    """Test time formatting function"""
    print("Testing format_time...")
    
    # Test seconds
    assert format_time(30) == "30s"
    
    # Test minutes
    assert format_time(90) == "1m 30s"
    
    # Test hours
    assert format_time(3665) == "1h 1m 5s"
    
    print("✓ format_time tests passed")


def test_create_progress_bar():
    """Test progress bar creation"""
    print("Testing create_progress_bar...")
    
    # Test 0%
    bar = create_progress_bar(0, 10)
    assert bar == "[░░░░░░░░░░]"
    
    # Test 50%
    bar = create_progress_bar(50, 10)
    assert bar == "[█████░░░░░]"
    
    # Test 100%
    bar = create_progress_bar(100, 10)
    assert bar == "[██████████]"
    
    print("✓ create_progress_bar tests passed")


def test_validate_output_path():
    """Test output path validation"""
    print("Testing validate_output_path...")
    
    # Test with temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_output.txt"
        validated = validate_output_path(output_path)
        assert validated == output_path
        
        # Test creating nested directory
        nested_output = Path(tmpdir) / "nested" / "test_output.txt"
        validated = validate_output_path(nested_output)
        assert validated == nested_output
        assert nested_output.parent.exists()
    
    print("✓ validate_output_path tests passed")


def test_validate_input_file():
    """Test input file validation"""
    print("Testing validate_input_file...")
    
    # Test with non-existent file
    try:
        validate_input_file("nonexistent.mp3")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass
    
    # Test with unsupported format
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmpfile:
        try:
            validate_input_file(tmpfile.name)
            assert False, "Should have raised ValueError for unsupported format"
        except ValueError as e:
            assert "Unsupported audio format" in str(e)
    
    # Test with supported format
    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmpfile:
        validated = validate_input_file(tmpfile.name)
        assert validated == Path(tmpfile.name)
    
    print("✓ validate_input_file tests passed")


def main():
    """Run all tests"""
    print("Running utils module tests...\n")
    
    test_format_time()
    test_create_progress_bar()
    test_validate_output_path()
    test_validate_input_file()
    
    print("\n✅ All utils tests passed!")


if __name__ == "__main__":
    main()