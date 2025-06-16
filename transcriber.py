"""
Core audio transcription functionality using faster-whisper
"""
import os
import time
import subprocess
from pathlib import Path
from typing import Optional, Callable, Dict, Any

from device_detection import OptimalConfiguration, setup_environment


class AudioTranscriber:
    """Core audio transcription class using faster-whisper"""
    
    SUPPORTED_FORMATS = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma'}
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None, compute_type: Optional[str] = None, 
                 batch_size: Optional[int] = None, num_workers: Optional[int] = None):
        """
        Initialize the audio transcriber
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            compute_type: Compute type ('auto', 'float32', 'float16', 'int8', 'int8_float16')
            batch_size: Batch size for processing (auto-detected if None)
            num_workers: Number of worker threads (auto-detected if None)
        """
        self.model_size = model_size
        self.config = OptimalConfiguration()
        
        # Determine optimal configuration
        self.device = self.config.get_optimal_device(device)
        self.compute_type = self.config.get_optimal_compute_type(self.device, compute_type)
        self.batch_size = batch_size or self.config.get_optimal_batch_size(self.device, model_size)
        self.num_workers = num_workers or self.config.get_optimal_threads()
        
        # Set up environment for optimal performance
        setup_environment(self.num_workers)
        
        self.model = None
        
    
    def _load_model(self) -> None:
        """Load the Whisper model if not already loaded"""
        if self.model is not None:
            return
            
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("faster-whisper is not installed. Run: pip install faster-whisper")
        
        try:
            self.model = WhisperModel(
                self.model_size, 
                device=self.device, 
                compute_type=self.compute_type,
                num_workers=self.num_workers if self.device == "cpu" else 1
            )
        except Exception as e:
            # Fallback to CPU with safe compute type
            self.device = "cpu"
            self.compute_type = "float32"
            self.batch_size = 1
            self.model = WhisperModel(
                self.model_size, 
                device=self.device, 
                compute_type=self.compute_type,
                num_workers=self.num_workers
            )
    
    def validate_input_file(self, file_path: str) -> Path:
        """
        Validate input audio file
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Path object of validated file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid audio file
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {path.suffix}. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        return path
    
    def validate_output_path(self, output_path: str) -> Path:
        """
        Validate output file path and directory
        
        Args:
            output_path: Path for the output text file
            
        Returns:
            Path object of validated output path
            
        Raises:
            PermissionError: If directory cannot be created or is not writable
        """
        path = Path(output_path)
        
        # Check if parent directory exists and is writable
        parent_dir = path.parent
        if not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise PermissionError(f"Cannot create output directory: {parent_dir}. {e}")
        
        if not os.access(parent_dir, os.W_OK):
            raise PermissionError(f"Output directory is not writable: {parent_dir}")
        
        return path
    
    def get_audio_duration(self, file_path: str) -> float:
        """
        Get audio file duration in seconds using ffprobe
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Duration in seconds
            
        Raises:
            RuntimeError: If ffprobe fails to get duration
        """
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', str(file_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            return duration
        except (subprocess.CalledProcessError, ValueError) as e:
            raise RuntimeError(f"Failed to get audio duration: {e}")
    
    def transcribe_file(
        self, 
        input_path: str, 
        output_path: str, 
        progress_callback: Optional[Callable[[float, int, float], None]] = None,
        max_duration_minutes: Optional[int] = None,
        start_time_seconds: Optional[float] = None,
        end_time_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text
        
        Args:
            input_path: Path to input audio file
            output_path: Path to output text file
            progress_callback: Optional callback function for progress updates
                              Called with (progress_percentage, segment_count, elapsed_time)
            max_duration_minutes: Optional limit transcription to first N minutes
            start_time_seconds: Optional start time in seconds
            end_time_seconds: Optional end time in seconds
        
        Returns:
            Dictionary with transcription results:
            {
                'language': str,
                'language_probability': float,
                'audio_duration': float,
                'processing_time': float,
                'speed_ratio': float,
                'segment_count': int,
                'output_path': str
            }
        """
        # Validate inputs
        input_file = self.validate_input_file(input_path)
        output_file = self.validate_output_path(output_path)
        
        # Get audio duration
        audio_duration = self.get_audio_duration(input_file)
        
        # Load model if needed
        self._load_model()
        
        # Start transcription
        start_time = time.time()
        
        # Prepare transcription parameters
        transcribe_kwargs = {
            'beam_size': 5,
            'vad_filter': True,
            'vad_parameters': dict(min_silence_duration_ms=1000),
            'temperature': 0.0
        }
        
        # Add time limiting if specified
        if max_duration_minutes is not None:
            clip_end_seconds = max_duration_minutes * 60
            transcribe_kwargs['clip_timestamps'] = f"0,{clip_end_seconds}"
        elif start_time_seconds is not None or end_time_seconds is not None:
            start_time = start_time_seconds or 0
            end_time = end_time_seconds or audio_duration
            transcribe_kwargs['clip_timestamps'] = f"{start_time},{end_time}"
        
        # Use simple, reliable parameters
        segments, info = self.model.transcribe(
            str(input_file),
            **transcribe_kwargs
        )
        
        # Write transcription to file with progress tracking
        with open(output_file, 'w', encoding='utf-8') as f:
            segment_count = 0
            last_segment_end = 0
            last_progress_update = time.time()
            
            for segment in segments:
                # Write segment text
                f.write(segment.text.strip())
                f.write('\n')
                
                segment_count += 1
                last_segment_end = segment.end
                
                # Call progress callback if provided
                if progress_callback and time.time() - last_progress_update >= 1.0:
                    progress_percentage = min((last_segment_end / audio_duration) * 100, 100)
                    elapsed_time = time.time() - start_time
                    progress_callback(progress_percentage, segment_count, elapsed_time)
                    last_progress_update = time.time()
        
        # Calculate final metrics
        total_time = time.time() - start_time
        speed_ratio = audio_duration / total_time if total_time > 0 else 0
        
        return {
            'language': info.language,
            'language_probability': info.language_probability,
            'audio_duration': audio_duration,
            'processing_time': total_time,
            'speed_ratio': speed_ratio,
            'segment_count': segment_count,
            'output_path': str(output_file)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration"""
        info = {
            'model_size': self.model_size,
            'device': self.device,
            'compute_type': self.compute_type,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'model_loaded': self.model is not None
        }
        
        # Add device capabilities summary
        config_summary = self.config.get_configuration_summary(
            self.device, self.compute_type, self.num_workers, self.batch_size
        )
        info['capabilities'] = config_summary['capabilities']
        
        return info