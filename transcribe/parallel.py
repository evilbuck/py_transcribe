"""
Parallel processing and chunk management for audio transcription
"""
import math
import time
import tempfile
import shutil
import subprocess
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional

from .utils import get_audio_duration, format_time
from .logger import get_logger


def should_use_parallel_processing(duration_seconds: float, min_duration_minutes: int = 30) -> bool:
    """Determine if file should be processed with parallel chunking"""
    min_duration_seconds = min_duration_minutes * 60
    return duration_seconds > min_duration_seconds


def get_optimal_thread_count(user_threads: Optional[int] = None) -> int:
    """Determine optimal thread count for parallel processing"""
    if user_threads is not None:
        if user_threads < 1:
            raise ValueError("Thread count must be at least 1")
        return user_threads
    
    # Auto-detect based on CPU cores
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    # Use 75% of CPU cores, minimum 2, maximum 8 for reasonable performance
    optimal_threads = max(2, min(8, int(cpu_count * 0.75)))
    return optimal_threads


def get_model_memory_requirements(model_size: str) -> int:
    """Get estimated memory requirements for different model sizes"""
    # Memory estimates based on model size (in MB)
    memory_map = {
        "tiny": 150,    # ~39M parameters
        "base": 500,    # ~74M parameters  
        "small": 1000,  # ~244M parameters
        "medium": 2500, # ~769M parameters
        "large": 4500   # ~1550M parameters
    }
    return memory_map.get(model_size, 500)


def optimize_thread_count_for_gpu(requested_threads: int, gpu_memory_mb: int, model_size: str = "base") -> int:
    """Optimize thread count based on GPU memory constraints"""
    
    # Estimate memory usage per model instance
    model_memory_mb = get_model_memory_requirements(model_size)
    
    if gpu_memory_mb > 0:
        # Calculate max threads based on available GPU memory
        # Leave 2GB buffer for system and other processes
        available_memory = max(0, gpu_memory_mb - 2048)
        max_threads_by_memory = max(1, available_memory // model_memory_mb)
        
        # Don't exceed requested threads, but warn if memory constrained
        optimal_threads = min(requested_threads, max_threads_by_memory)
        
        logger = get_logger()
        if optimal_threads < requested_threads:
            logger.warning(f"GPU memory constraint: Using {optimal_threads} threads instead of {requested_threads}")
            logger.info(f"   Model memory: ~{model_memory_mb}MB x {requested_threads} = {model_memory_mb * requested_threads}MB")
            logger.info(f"   Available GPU memory: ~{gpu_memory_mb}MB")
        
        return optimal_threads
    
    return requested_threads


def calculate_optimal_chunk_size(duration_seconds: float, num_threads: int, 
                                gpu_memory_mb: int, default_minutes: int = 10, 
                                model_size: str = "base") -> int:
    """Calculate optimal chunk size based on GPU memory and file duration"""
    
    # Get model memory requirements
    model_memory_mb = get_model_memory_requirements(model_size)
    
    # Base chunk size in minutes
    optimal_minutes = default_minutes
    
    if gpu_memory_mb > 0:
        # Calculate total memory needed for parallel models
        total_model_memory = model_memory_mb * num_threads
        
        # Adjust chunk size based on memory availability
        memory_ratio = gpu_memory_mb / total_model_memory
        
        if memory_ratio >= 3.0:
            # Abundant memory: use larger chunks for efficiency
            optimal_minutes = int(default_minutes * 1.5)
        elif memory_ratio >= 2.0:
            # Good memory: use default chunks
            optimal_minutes = default_minutes
        elif memory_ratio >= 1.5:
            # Adequate memory: slightly smaller chunks
            optimal_minutes = int(default_minutes * 0.8)
        else:
            # Constrained memory: use smaller chunks
            optimal_minutes = max(5, int(default_minutes * 0.5))
        
        # For very long files, adjust chunk size to balance parallelism
        duration_hours = duration_seconds / 3600
        if duration_hours > 3:  # Very long files (>3 hours)
            # Use slightly larger chunks to reduce overhead
            optimal_minutes = int(optimal_minutes * 1.2)
        elif duration_hours < 0.5:  # Short files (<30 min)
            # Use smaller chunks for better parallelism
            optimal_minutes = max(5, int(optimal_minutes * 0.7))
    
    # Ensure chunk size is reasonable
    optimal_minutes = max(5, min(30, optimal_minutes))
    
    return optimal_minutes


def create_audio_chunks(input_file: Path, chunk_duration_minutes: int, temp_dir: Path) -> List[Dict[str, Any]]:
    """Split audio file into chunks using ffmpeg"""
    logger = get_logger()
    chunk_duration_seconds = chunk_duration_minutes * 60
    chunks = []
    
    try:
        # Get total duration
        total_duration = get_audio_duration(input_file)
        num_chunks = math.ceil(total_duration / chunk_duration_seconds)
        
        logger.info(f"Splitting audio into {num_chunks} chunks of {chunk_duration_minutes} minutes each...")
        
        for i in range(num_chunks):
            start_time = i * chunk_duration_seconds
            chunk_file = Path(temp_dir) / f"chunk_{i:03d}.wav"
            
            # Use ffmpeg to extract chunk
            cmd = [
                'ffmpeg', '-y', '-i', str(input_file),
                '-ss', str(start_time),
                '-t', str(chunk_duration_seconds),
                '-acodec', 'pcm_s16le',  # Use WAV format for reliability
                '-ar', '16000',  # Standard sample rate for whisper
                str(chunk_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to create chunk {i}: {result.stderr}")
            
            # Get actual duration of this chunk
            chunk_duration = get_audio_duration(chunk_file)
            
            chunks.append({
                'index': i,
                'file': chunk_file,
                'start_time': start_time,
                'duration': chunk_duration,
                'end_time': start_time + chunk_duration
            })
            
            logger.success(f"Created chunk {i+1}/{num_chunks}")
        
        return chunks
        
    except Exception as e:
        # Clean up any created chunks on error
        for chunk in chunks:
            if chunk['file'].exists():
                chunk['file'].unlink()
        raise RuntimeError(f"Failed to create audio chunks: {e}")


def transcribe_chunk_with_model(chunk_info: Dict[str, Any], model, transcription_params: Dict[str, Any]) -> Dict[str, Any]:
    """Transcribe a single audio chunk using provided model"""
    try:
        # Transcribe the chunk with optimized parameters
        segments, info = model.transcribe(
            str(chunk_info['file']),
            **transcription_params
        )
        
        # Collect all segments with adjusted timestamps
        chunk_segments = []
        for segment in segments:
            # Adjust timestamps to reflect position in original file
            adjusted_segment = {
                'start': segment.start + chunk_info['start_time'],
                'end': segment.end + chunk_info['start_time'],
                'text': segment.text.strip()
            }
            chunk_segments.append(adjusted_segment)
        
        result = {
            'chunk_index': chunk_info['index'],
            'segments': chunk_segments,
            'language': info.language,
            'language_probability': info.language_probability
        }
        
        return result
        
    except Exception as e:
        return {
            'chunk_index': chunk_info['index'],
            'error': str(e),
            'segments': []
        }


def assemble_transcripts(chunk_results: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Assemble transcripts from chunks in correct chronological order"""
    # Sort by chunk index to ensure correct order
    sorted_results = sorted(chunk_results, key=lambda x: x['chunk_index'])
    
    # Check for errors
    errors = [r for r in sorted_results if 'error' in r]
    if errors:
        error_msgs = [f"Chunk {r['chunk_index']}: {r['error']}" for r in errors]
        raise RuntimeError(f"Transcription errors in chunks:\n" + "\n".join(error_msgs))
    
    # Combine all segments in chronological order
    all_segments = []
    for result in sorted_results:
        all_segments.extend(result['segments'])
    
    # Sort segments by start time for safety (should already be in order)
    all_segments.sort(key=lambda x: x['start'])
    
    # Extract language info from first successful chunk
    language_info = None
    for result in sorted_results:
        if result['segments'] and 'language' in result:
            language_info = {
                'language': result['language'],
                'language_probability': result['language_probability']
            }
            break
    
    return all_segments, language_info


class ParallelTranscriber:
    """Handles parallel transcription processing"""
    
    def __init__(self, model_size: str, device: str, compute_type: str, num_threads: int):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.num_threads = num_threads
        self.logger = get_logger()
        
    def transcribe_file_parallel(self, input_path: Path, output_path: Path, 
                                chunk_duration_minutes: int, 
                                transcription_params: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe audio file using parallel chunk processing"""
        temp_dir = None
        
        try:
            # Create temporary directory for chunks
            temp_dir = tempfile.mkdtemp(prefix="whisper_chunks_")
            temp_path = Path(temp_dir)
            
            # Get audio duration and optimize chunk size
            duration = get_audio_duration(input_path)
            
            # For this simplified version, we'll use a basic approach
            # In a full implementation, we'd integrate with GPU memory optimization
            gpu_memory_mb = 8192  # Fallback estimate
            
            optimal_chunk_size = calculate_optimal_chunk_size(
                duration, self.num_threads, gpu_memory_mb, chunk_duration_minutes, self.model_size
            )
            
            self.logger.info(f"Using {self.num_threads} threads for parallel processing")
            self.logger.info(f"Optimal chunk size: {optimal_chunk_size} minutes (requested: {chunk_duration_minutes})")
            
            # Create audio chunks
            chunks = create_audio_chunks(input_path, optimal_chunk_size, temp_path)
            
            self.logger.info(f"\nProcessing {len(chunks)} chunks in parallel...")
            start_time = time.time()
            
            # Process chunks in parallel using simple model loading per thread
            # Note: In a full implementation, we'd use the model pool from models.py
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Submit all chunk transcription tasks
                future_to_chunk = {}
                
                for chunk in chunks:
                    future = executor.submit(self._transcribe_chunk_simple, chunk, transcription_params)
                    future_to_chunk[future] = chunk
                
                # Collect results as they complete
                chunk_results = []
                completed = 0
                
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        result = future.result()
                        chunk_results.append(result)
                        completed += 1
                        
                        progress = (completed / len(chunks)) * 100
                        elapsed = time.time() - start_time
                        self.logger.progress(f"✓ Chunk {chunk['index'] + 1}/{len(chunks)} completed ({progress:.1f}%) - Elapsed: {format_time(elapsed)}")
                        
                    except Exception as e:
                        self.logger.error(f"✗ Chunk {chunk['index'] + 1} failed: {e}")
                        chunk_results.append({
                            'chunk_index': chunk['index'],
                            'error': str(e),
                            'segments': []
                        })
            
            # Assemble final transcript
            self.logger.info("\nAssembling final transcript...")
            all_segments, language_info = assemble_transcripts(chunk_results)
            
            # Write assembled transcript to file
            with open(output_path, 'w', encoding='utf-8') as f:
                for segment in all_segments:
                    f.write(segment['text'])
                    f.write('\n')
            
            total_time = time.time() - start_time
            total_duration = sum(chunk['duration'] for chunk in chunks)
            speed_ratio = total_duration / total_time if total_time > 0 else 0
            
            results = {
                'segments': all_segments,
                'language': language_info['language'] if language_info else 'unknown',
                'language_probability': language_info['language_probability'] if language_info else 0.0,
                'audio_duration': total_duration,
                'processing_time': total_time,
                'speed_ratio': speed_ratio,
                'segment_count': len(all_segments),
                'output_file': str(output_path)
            }
            
            self.logger.success("Parallel transcription completed!")
            if language_info:
                self.logger.info(f"  Language: {language_info['language']} (probability: {language_info['language_probability']:.2f})")
            self.logger.info(f"  Audio duration: {format_time(total_duration)}")
            self.logger.info(f"  Processing time: {format_time(total_time)}")
            self.logger.info(f"  Speed ratio: {speed_ratio:.1f}x faster than real-time")
            self.logger.info(f"  Total segments: {len(all_segments)}")
            self.logger.info(f"  Output saved to: {output_path}")
            
            return results
            
        finally:
            # Clean up temporary directory
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
    
    def _transcribe_chunk_simple(self, chunk_info: Dict[str, Any], transcription_params: Dict[str, Any]) -> Dict[str, Any]:
        """Simple chunk transcription with individual model loading"""
        try:
            # Import here to avoid circular imports
            from .transcriber import AudioTranscriber
            
            # Create a transcriber for this chunk
            transcriber = AudioTranscriber(
                model_size=self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            
            # Initialize model
            transcriber._initialize_model()
            
            # Transcribe using the core method
            result = transcribe_chunk_with_model(chunk_info, transcriber.model, transcription_params)
            
            # Clean up
            transcriber.close()
            
            return result
            
        except Exception as e:
            return {
                'chunk_index': chunk_info['index'],
                'error': str(e),
                'segments': []
            }