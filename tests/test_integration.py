"""
Integration tests to verify the refactored modules work together
"""
import pytest
import tempfile
from pathlib import Path
import subprocess
import sys

# Test that we can import all modules without errors
def test_import_all_modules():
    """Test that all modules can be imported successfully"""
    
    # Core modules
    from transcribe import AudioTranscriber, validate_input_file, validate_output_path, format_time
    from transcribe.logger import get_logger, set_debug_mode
    from transcribe.utils import create_progress_bar
    
    # Advanced modules
    from transcribe.device_detection import detect_device_capabilities, get_optimal_device
    from transcribe.parallel import should_use_parallel_processing, get_optimal_thread_count
    from transcribe.models import get_model_cache_stats
    
    # CLI module (requires typer)
    try:
        from transcribe.cli import main
        assert callable(main)
    except ImportError:
        pytest.skip("CLI module requires typer")
    
    # Verify all imports succeeded
    assert AudioTranscriber is not None
    assert validate_input_file is not None
    assert get_logger is not None


def test_library_usage_workflow():
    """Test the complete library usage workflow"""
    
    # Test the workflow described in documentation
    from transcribe import AudioTranscriber, validate_input_file, get_logger
    
    # Set up logging
    logger = get_logger(verbose=True)
    logger.info("Testing library workflow")
    
    # Create transcriber
    transcriber = AudioTranscriber(model_size="tiny", device="cpu", verbose=True)
    
    # Check model info
    info = transcriber.get_model_info()
    assert info['model_size'] == "tiny"
    assert info['device'] == "cpu"
    assert info['is_loaded'] == False
    
    # Clean up
    transcriber.close()
    
    logger.success("Library workflow test completed")


def test_device_detection_integration():
    """Test device detection integration"""
    
    from transcribe.device_detection import detect_device_capabilities, report_device_capabilities
    from transcribe.logger import get_logger
    
    logger = get_logger(verbose=False)  # Reduce noise
    
    # Detect capabilities
    capabilities = detect_device_capabilities()
    
    # Verify structure
    assert isinstance(capabilities, dict)
    assert 'platform' in capabilities
    assert 'recommended_device' in capabilities
    assert 'recommended_compute_type' in capabilities
    
    # Test reporting (should not crash)
    report_device_capabilities(capabilities, verbose=False)
    
    logger.success("Device detection integration test completed")


def test_parallel_processing_logic():
    """Test parallel processing decision logic"""
    
    from transcribe.parallel import should_use_parallel_processing, get_optimal_thread_count
    from transcribe.logger import get_logger
    
    logger = get_logger(verbose=False)
    
    # Test decision logic
    assert not should_use_parallel_processing(60)  # 1 minute - no parallel
    assert should_use_parallel_processing(2400)    # 40 minutes - use parallel
    
    # Test thread count
    threads = get_optimal_thread_count()
    assert isinstance(threads, int)
    assert 2 <= threads <= 8
    
    # Test user override
    user_threads = get_optimal_thread_count(4)
    assert user_threads == 4
    
    logger.success("Parallel processing logic test completed")


def test_model_cache_functionality():
    """Test model cache functionality"""
    
    from transcribe.models import get_model_cache_stats
    from transcribe.logger import get_logger
    
    logger = get_logger(verbose=False)
    
    # Get cache stats (should not crash)
    stats = get_model_cache_stats()
    
    assert isinstance(stats, dict)
    assert 'cached_models' in stats
    assert 'total_accesses' in stats
    assert 'models' in stats
    
    logger.success("Model cache functionality test completed")


@pytest.mark.skipif(
    subprocess.run([sys.executable, '-c', 'import typer'], capture_output=True).returncode != 0,
    reason="Typer not available"
)
def test_cli_help_commands():
    """Test CLI help commands work"""
    
    # Test main help
    result = subprocess.run(
        [sys.executable, 'transcribe_cli.py', '--help'],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "Transcribe audio files to text" in result.stdout
    
    # Test transcribe command help
    result = subprocess.run(
        [sys.executable, 'transcribe_cli.py', 'transcribe', '--help'],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "Transcribe an audio file to text" in result.stdout


def test_end_to_end_workflow_mock():
    """Test end-to-end workflow with mocked transcription"""
    
    from transcribe import AudioTranscriber, validate_input_file, validate_output_path
    from transcribe.logger import get_logger
    
    logger = get_logger(debug=True)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock input file
        input_file = temp_path / "test.mp3"
        input_file.write_bytes(b"fake audio content")
        
        # Validate input
        validated_input = validate_input_file(input_file)
        assert validated_input == input_file
        
        # Create output path
        output_file = temp_path / "output.txt"
        validated_output = validate_output_path(output_file)
        assert validated_output == output_file
        
        # Create transcriber
        transcriber = AudioTranscriber(model_size="tiny", device="cpu", debug=True)
        
        # Test that we can create transcriber and get info
        info = transcriber.get_model_info()
        assert info['model_size'] == "tiny"
        assert info['device'] == "cpu"
        
        # Clean up
        transcriber.close()
        
        logger.success("End-to-end workflow test completed")


def test_error_handling():
    """Test error handling across modules"""
    
    from transcribe import validate_input_file, validate_output_path
    from transcribe.logger import get_logger
    
    logger = get_logger(verbose=False)
    
    # Test input validation errors
    with pytest.raises(FileNotFoundError):
        validate_input_file("nonexistent.mp3")
    
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        with pytest.raises(ValueError, match="Unsupported audio format"):
            validate_input_file(tmp.name)
    
    # Test that output validation creates directories
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "new_dir" / "output.txt"
        validated = validate_output_path(output_path)
        assert validated.parent.exists()
    
    logger.success("Error handling test completed")


if __name__ == "__main__":
    # Run tests manually if needed
    pytest.main([__file__, "-v"])