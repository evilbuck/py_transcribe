"""
Offline Audio Transcriber - A modular transcription library using faster-whisper
"""

from .transcriber import AudioTranscriber
from .utils import validate_input_file, validate_output_path, format_time
from .logger import get_logger, set_debug_mode, set_verbose_mode

# Other modules available for advanced usage:
# - transcribe.device_detection: Hardware detection and optimization
# - transcribe.parallel: Parallel processing and chunk management  
# - transcribe.models: Model caching and pool management
# - transcribe.cli: Command-line interface (requires typer)

__version__ = "1.0.0"
__all__ = ["AudioTranscriber", "validate_input_file", "validate_output_path", "format_time", "get_logger", "set_debug_mode", "set_verbose_mode"]