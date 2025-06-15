"""
Logging utilities for transcription system
"""
import logging
import sys
from typing import Optional


class TranscriptionLogger:
    """Logger for transcription operations with debug mode support"""
    
    def __init__(self, name: str = "transcribe", debug: bool = False, verbose: bool = False):
        self.logger = logging.getLogger(name)
        self.debug_mode = debug
        self.verbose_mode = verbose
        
        # Remove existing handlers to avoid duplication
        self.logger.handlers.clear()
        
        # Set level based on mode
        if debug:
            self.logger.setLevel(logging.DEBUG)
        elif verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        
        # Create formatter
        if debug:
            formatter = logging.Formatter('[%(levelname)s] %(name)s:%(lineno)d - %(message)s')
        else:
            formatter = logging.Formatter('%(message)s')
        
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(f"DEBUG: {message}")
    
    def info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(f"⚠️  {message}")
    
    def error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(f"❌ Error: {message}")
    
    def success(self, message: str) -> None:
        """Log success message"""
        self.logger.info(f"✓ {message}")
    
    def progress(self, message: str) -> None:
        """Log progress message"""
        self.logger.info(message)
    
    def system_info(self, message: str) -> None:
        """Log system information"""
        self.logger.info(f"🖥️  {message}")
    
    def gpu_info(self, message: str) -> None:
        """Log GPU information"""
        self.logger.info(f"⚡ {message}")
    
    def model_info(self, message: str) -> None:
        """Log model information"""
        self.logger.info(f"📦 {message}")
    
    def optimization_info(self, message: str) -> None:
        """Log optimization information"""
        self.logger.info(f"🎯 {message}")


# Global logger instance
_logger: Optional[TranscriptionLogger] = None


def get_logger(debug: bool = False, verbose: bool = False) -> TranscriptionLogger:
    """Get or create the global logger instance"""
    global _logger
    # If no parameters provided and logger exists, return existing logger
    if debug == False and verbose == False and _logger is not None:
        return _logger
    # Otherwise create new logger with specified parameters
    if _logger is None or _logger.debug_mode != debug or _logger.verbose_mode != verbose:
        _logger = TranscriptionLogger(debug=debug, verbose=verbose)
    return _logger


def set_debug_mode(debug: bool = True) -> None:
    """Enable or disable debug mode"""
    global _logger
    verbose = _logger.verbose_mode if _logger else False
    _logger = TranscriptionLogger(debug=debug, verbose=verbose)


def set_verbose_mode(verbose: bool = True) -> None:
    """Enable or disable verbose mode"""
    global _logger
    debug = _logger.debug_mode if _logger else False
    _logger = TranscriptionLogger(debug=debug, verbose=verbose)