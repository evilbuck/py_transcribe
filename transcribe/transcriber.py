"""
Core audio transcription functionality
"""
import time
from pathlib import Path
from typing import Optional, Dict, Any, Generator, Tuple

from .utils import get_audio_duration, format_time
from .logger import get_logger


class AudioTranscriber:
    """Core audio transcriber using faster-whisper"""
    
    def __init__(self, model_size: str = "base", device: str = "auto", 
                 compute_type: str = "auto", debug: bool = False, verbose: bool = False):
        """
        Initialize the audio transcriber
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (auto, cpu, cuda, mps)
            compute_type: Compute type (auto, float32, float16, int8, int8_float16)
            debug: Enable debug logging
            verbose: Enable verbose logging
        """
        self.model_size = model_size
        self.model = None
        self.logger = get_logger(debug=debug, verbose=verbose)
        
        # Use device detection for optimal settings
        try:
            from .device_detection import get_optimal_device, get_optimal_compute_type
            self.device = get_optimal_device(device)
            self.compute_type = get_optimal_compute_type(compute_type)
        except ImportError:
            # Fallback if device detection module not available
            self.device = device if device != "auto" else "cpu"
            self.compute_type = compute_type if compute_type != "auto" else "float32"
        
    def _initialize_model(self) -> None:
        """Initialize the Whisper model if not already loaded"""
        if self.model is not None:
            return
            
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("faster-whisper is not installed. Run: pip install faster-whisper")
        
        self.logger.debug(f"Initializing model: {self.model_size}, device: {self.device}, compute: {self.compute_type}")
        start_time = time.time()
        
        try:
            # Try optimal configuration first
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            self.logger.model_info(f"Model '{self.model_size}' loaded successfully")
        except Exception as e:
            self.logger.warning(f"Falling back to CPU due to: {e}")
            # Fallback to CPU with safe compute type
            self.model = WhisperModel(self.model_size, device="cpu", compute_type="float32")
            self.device = "cpu"
            self.compute_type = "float32"
        
        load_time = time.time() - start_time
        self.logger.success(f"Model loaded in {load_time:.1f}s")
    
    def transcribe_file(self, input_path: Path, output_path: Optional[Path] = None,
                       transcription_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Transcribe an audio file
        
        Args:
            input_path: Path to input audio file
            output_path: Optional path for output text file
            transcription_params: Optional transcription parameters
            
        Returns:
            Dictionary with transcription results and metadata
        """
        self._initialize_model()
        
        # Default transcription parameters
        if transcription_params is None:
            transcription_params = {
                'beam_size': 5,
                'vad_filter': True,
                'vad_parameters': dict(min_silence_duration_ms=1000),
                'temperature': 0.0
            }
        
        self.logger.info(f"Transcribing audio: {input_path.name}")
        self.logger.debug(f"Parameters: {transcription_params}")
        
        start_time = time.time()
        
        # Get audio duration for progress tracking
        try:
            audio_duration = get_audio_duration(input_path)
            self.logger.info(f"Audio duration: {format_time(audio_duration)}")
        except Exception as e:
            self.logger.warning(f"Could not determine audio duration: {e}")
            audio_duration = 0
        
        # Transcribe with streaming segments for memory efficiency
        try:
            segments, info = self.model.transcribe(str(input_path), **transcription_params)
            
            self.logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
            
            # Collect segments
            all_segments = []
            segment_count = 0
            last_segment_end = 0
            last_progress_update = time.time()
            
            for segment in segments:
                segment_data = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip()
                }
                all_segments.append(segment_data)
                
                segment_count += 1
                last_segment_end = segment.end
                
                # Update progress every 5 seconds to avoid overwhelming output
                current_time = time.time()
                if current_time - last_progress_update >= 5.0:
                    if audio_duration > 0:
                        progress_percentage = min((last_segment_end / audio_duration) * 100, 100)
                        elapsed_time = current_time - start_time
                        self.logger.progress(f"Progress: {progress_percentage:.1f}% - {segment_count} segments processed - Elapsed: {format_time(elapsed_time)}")
                    else:
                        self.logger.progress(f"Processed {segment_count} segments")
                    last_progress_update = current_time
            
            total_time = time.time() - start_time
            speed_ratio = audio_duration / total_time if total_time > 0 and audio_duration > 0 else 0
            
            # Prepare results
            results = {
                'segments': all_segments,
                'language': info.language,
                'language_probability': info.language_probability,
                'audio_duration': audio_duration,
                'processing_time': total_time,
                'speed_ratio': speed_ratio,
                'segment_count': segment_count
            }
            
            # Write to output file if specified
            if output_path:
                self._write_transcript(output_path, all_segments)
                results['output_file'] = str(output_path)
            
            # Log completion
            self.logger.success("Transcription completed!")
            if audio_duration > 0:
                self.logger.info(f"  Audio duration: {format_time(audio_duration)}")
            self.logger.info(f"  Processing time: {format_time(total_time)}")
            if speed_ratio > 0:
                self.logger.info(f"  Speed ratio: {speed_ratio:.1f}x faster than real-time")
            self.logger.info(f"  Total segments: {segment_count}")
            if output_path:
                self.logger.info(f"  Output saved to: {output_path}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {e}")
    
    def transcribe_text_only(self, input_path: Path,
                           transcription_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Transcribe an audio file and return only the text
        
        Args:
            input_path: Path to input audio file
            transcription_params: Optional transcription parameters
            
        Returns:
            Transcribed text as a single string
        """
        results = self.transcribe_file(input_path, transcription_params=transcription_params)
        return '\n'.join(segment['text'] for segment in results['segments'])
    
    def _write_transcript(self, output_path: Path, segments: list) -> None:
        """Write transcript segments to file"""
        self.logger.debug(f"Writing transcript to: {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for segment in segments:
                    f.write(segment['text'])
                    f.write('\n')
        except Exception as e:
            self.logger.error(f"Failed to write transcript: {e}")
            raise RuntimeError(f"Failed to write transcript: {e}")
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model"""
        return {
            'model_size': self.model_size,
            'device': self.device,
            'compute_type': self.compute_type,
            'is_loaded': self.model is not None
        }
    
    def close(self) -> None:
        """Clean up resources"""
        if self.model is not None:
            del self.model
            self.model = None
            self.logger.debug("Model resources cleaned up")