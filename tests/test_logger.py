"""
Unit tests for logger module
"""
import pytest
import logging
from io import StringIO
from unittest.mock import patch

from transcribe.logger import TranscriptionLogger, get_logger, set_debug_mode, set_verbose_mode


class TestTranscriptionLogger:
    """Test TranscriptionLogger functionality"""
    
    def test_logger_initialization(self):
        """Test logger initialization with different modes"""
        # Test default mode
        logger = TranscriptionLogger()
        assert logger.logger.level == logging.WARNING
        assert not logger.debug_mode
        assert not logger.verbose_mode
        
        # Test debug mode
        debug_logger = TranscriptionLogger(debug=True)
        assert debug_logger.logger.level == logging.DEBUG
        assert debug_logger.debug_mode
        
        # Test verbose mode
        verbose_logger = TranscriptionLogger(verbose=True)
        assert verbose_logger.logger.level == logging.INFO
        assert verbose_logger.verbose_mode
    
    def test_log_methods(self):
        """Test different logging methods"""
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            logger = TranscriptionLogger(verbose=True)
            
            # Test different log methods
            logger.info("Test info message")
            logger.success("Test success message")
            logger.warning("Test warning message")
            logger.error("Test error message")
            logger.system_info("Test system info")
            logger.gpu_info("Test GPU info")
            logger.model_info("Test model info")
            logger.optimization_info("Test optimization info")
            
            output = mock_stdout.getvalue()
            assert "Test info message" in output
            assert "✓ Test success message" in output
            assert "⚠️  Test warning message" in output
            assert "❌ Error: Test error message" in output
            assert "🖥️  Test system info" in output
            assert "⚡ Test GPU info" in output
            assert "📦 Test model info" in output
            assert "🎯 Test optimization info" in output
    
    def test_debug_mode_formatting(self):
        """Test debug mode uses detailed formatting"""
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            logger = TranscriptionLogger(debug=True)
            logger.debug("Test debug message")
            
            output = mock_stdout.getvalue()
            assert "DEBUG: Test debug message" in output
            # Debug mode should include line numbers and module info
            assert "[DEBUG]" in output


class TestGlobalLogger:
    """Test global logger functions"""
    
    def test_get_logger(self):
        """Test get_logger function"""
        logger1 = get_logger()
        logger2 = get_logger()
        # Should return the same instance
        assert logger1 is logger2
        
        # Test with different modes
        debug_logger = get_logger(debug=True)
        assert debug_logger.debug_mode
        
        verbose_logger = get_logger(verbose=True)
        assert verbose_logger.verbose_mode
    
    def test_set_debug_mode(self):
        """Test set_debug_mode function"""
        set_debug_mode(True)
        logger = get_logger()
        assert logger.debug_mode
        
        set_debug_mode(False)
        logger = get_logger()
        assert not logger.debug_mode
    
    def test_set_verbose_mode(self):
        """Test set_verbose_mode function"""
        set_verbose_mode(True)
        logger = get_logger()
        assert logger.verbose_mode
        
        set_verbose_mode(False)
        logger = get_logger()
        assert not logger.verbose_mode
    
    def test_logger_recreation_on_mode_change(self):
        """Test that logger is recreated when mode changes"""
        logger1 = get_logger(debug=False)
        logger2 = get_logger(debug=True)
        # Should be different instances due to mode change
        assert logger1 is not logger2
        assert not logger1.debug_mode
        assert logger2.debug_mode