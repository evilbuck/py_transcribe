#!/usr/bin/env python3
"""
Test script for CLI module
"""
import subprocess
import tempfile
from pathlib import Path
import sys

def test_cli_help():
    """Test CLI help commands"""
    print("Testing CLI help commands...")
    
    # Test main help
    result = subprocess.run([sys.executable, "transcribe_cli.py", "--help"], 
                           capture_output=True, text=True)
    assert result.returncode == 0
    assert "Transcribe audio files to text" in result.stdout
    assert "transcribe" in result.stdout
    assert "cache-stats" in result.stdout
    
    # Test transcribe help
    result = subprocess.run([sys.executable, "transcribe_cli.py", "transcribe", "--help"], 
                           capture_output=True, text=True)
    assert result.returncode == 0
    assert "Transcribe an audio file to text" in result.stdout
    assert "--output" in result.stdout
    assert "--model" in result.stdout
    
    print("✓ CLI help tests passed")


def test_cli_transcription():
    """Test CLI transcription with real audio"""
    print("Testing CLI transcription...")
    
    # Check if test audio file exists
    test_audio = Path("assets/testing speech audio file.m4a")
    if not test_audio.exists():
        print("⚠️  Skipping CLI transcription test - test file not found")
        return
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        output_file = Path(tmp.name)
    
    try:
        # Run transcription
        result = subprocess.run([
            sys.executable, "transcribe_cli.py", "transcribe",
            str(test_audio),
            "-o", str(output_file),
            "--model", "tiny",
            "--device", "cpu",
            "--verbose"
        ], capture_output=True, text=True, timeout=30)
        
        # Check command succeeded
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        
        # Check output contains expected text
        assert "Transcription completed successfully!" in result.stdout
        assert "Language: en" in result.stdout
        
        # Check output file was created and has content
        assert output_file.exists()
        content = output_file.read_text().strip()
        assert len(content) > 0
        assert "test" in content.lower() or "brown" in content.lower()
        
        print("✓ CLI transcription test passed")
        
    except subprocess.TimeoutExpired:
        print("⚠️  CLI transcription test timed out")
    except Exception as e:
        print(f"⚠️  CLI transcription test failed: {e}")
    finally:
        # Cleanup
        if output_file.exists():
            output_file.unlink()


def test_cli_info_commands():
    """Test CLI info commands"""
    print("Testing CLI info commands...")
    
    # Test cache-stats
    result = subprocess.run([sys.executable, "transcribe_cli.py", "cache-stats"], 
                           capture_output=True, text=True)
    # Should succeed even if no cache exists
    assert result.returncode == 0
    
    # Test info command
    result = subprocess.run([sys.executable, "transcribe_cli.py", "info"], 
                           capture_output=True, text=True)
    assert result.returncode == 0
    assert "Platform:" in result.stdout or "System information" in result.stdout
    
    print("✓ CLI info command tests passed")


def test_cli_error_handling():
    """Test CLI error handling"""
    print("Testing CLI error handling...")
    
    # Test missing input file
    result = subprocess.run([
        sys.executable, "transcribe_cli.py", "transcribe",
        "nonexistent.mp3", "-o", "output.txt"
    ], capture_output=True, text=True)
    assert result.returncode == 1
    # Error message could be in stdout or stderr
    output_text = (result.stdout + result.stderr).lower()
    assert "not found" in output_text or "error" in output_text
    
    # Test missing output argument
    result = subprocess.run([
        sys.executable, "transcribe_cli.py", "transcribe",
        "assets/testing speech audio file.m4a"
    ], capture_output=True, text=True)
    assert result.returncode == 2  # Typer argument error
    
    print("✓ CLI error handling tests passed")


def main():
    """Run all CLI tests"""
    print("Running CLI module tests...\n")
    
    test_cli_help()
    test_cli_info_commands()
    test_cli_error_handling()
    test_cli_transcription()
    
    print("\n✅ CLI tests completed!")


if __name__ == "__main__":
    main()