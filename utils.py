"""
Utility functions for the audio transcription tool
"""
from typing import Union


def format_time(seconds: Union[int, float]) -> str:
    """
    Format seconds into human-readable time string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "45s", "2m 5s", "1h 2m 5s")
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins:.0f}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:.0f}h {mins:.0f}m {secs:.0f}s"


def validate_model_size(model_size: str) -> str:
    """
    Validate Whisper model size
    
    Args:
        model_size: Model size string
        
    Returns:
        Validated model size
        
    Raises:
        ValueError: If model size is not supported
    """
    valid_models = {"tiny", "base", "small", "medium", "large"}
    if model_size not in valid_models:
        raise ValueError(f"Invalid model size: {model_size}. Valid options: {', '.join(valid_models)}")
    return model_size


def validate_device(device: str) -> str:
    """
    Validate device option
    
    Args:
        device: Device string
        
    Returns:
        Validated device
        
    Raises:
        ValueError: If device is not supported
    """
    valid_devices = {"auto", "cpu", "cuda", "mps"}
    if device not in valid_devices:
        raise ValueError(f"Invalid device: {device}. Valid options: {', '.join(valid_devices)}")
    return device


def validate_compute_type(compute_type: str) -> str:
    """
    Validate compute type option
    
    Args:
        compute_type: Compute type string
        
    Returns:
        Validated compute type
        
    Raises:
        ValueError: If compute type is not supported
    """
    valid_types = {"auto", "float32", "float16", "int8", "int8_float16"}
    if compute_type not in valid_types:
        raise ValueError(f"Invalid compute type: {compute_type}. Valid options: {', '.join(valid_types)}")
    return compute_type


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    from pathlib import Path
    path = Path(file_path)
    if path.exists():
        return path.stat().st_size / (1024 * 1024)
    return 0.0


def estimate_processing_time(duration_seconds: float, model_size: str) -> float:
    """
    Estimate processing time based on audio duration and model size
    
    Args:
        duration_seconds: Audio duration in seconds
        model_size: Whisper model size
        
    Returns:
        Estimated processing time in seconds
    """
    # Rough estimates based on typical CPU performance
    speed_ratios = {
        "tiny": 20.0,    # 20x faster than real-time
        "base": 15.0,    # 15x faster than real-time
        "small": 10.0,   # 10x faster than real-time
        "medium": 6.0,   # 6x faster than real-time
        "large": 4.0     # 4x faster than real-time
    }
    
    ratio = speed_ratios.get(model_size, 10.0)
    return duration_seconds / ratio