#!/usr/bin/env python3
"""
Offline Audio Transcriber - Command-line tool for transcribing audio files using faster-whisper
"""
import argparse
import sys
import os
from pathlib import Path
import time
import math
import subprocess
import json
import multiprocessing
import concurrent.futures
import tempfile
import shutil
import threading
from queue import Queue, Empty
import atexit
import weakref
import platform


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files to text using faster-whisper (offline)",
        prog="transcribe"
    )
    
    parser.add_argument(
        "input_file",
        nargs="?",  # Make optional for cache-stats and preload commands
        help="Path to the audio file to transcribe"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Path for the output text file"
    )
    
    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size to use (default: base)"
    )
    
    parser.add_argument(
        "--threads",
        type=int,
        help="Number of threads for parallel processing (default: auto-detect CPU cores)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Chunk size in minutes for parallel processing (default: 10)"
    )
    
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing even for long files"
    )
    
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Pre-load model for faster subsequent usage"
    )
    
    parser.add_argument(
        "--cache-stats",
        action="store_true",
        help="Show model cache statistics and exit"
    )
    
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Force specific device for inference (default: auto-detect)"
    )
    
    parser.add_argument(
        "--compute-type",
        choices=["auto", "float32", "float16", "int8", "int8_float16"],
        default="auto",
        help="Force specific compute type (default: auto-detect based on device)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed system information and optimization hints"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable performance benchmarking and show detailed metrics"
    )
    
    return parser.parse_args()


def validate_input_file(file_path):
    """Validate input audio file"""
    supported_formats = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma'}
    
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    if path.suffix.lower() not in supported_formats:
        raise ValueError(f"Unsupported audio format: {path.suffix}. Supported: {', '.join(supported_formats)}")
    
    return path


def validate_output_path(output_path):
    """Validate output file path and directory"""
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


def get_audio_duration(file_path):
    """Get audio file duration in seconds using ffprobe"""
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


def should_use_parallel_processing(duration_seconds, min_duration_minutes=30):
    """Determine if file should be processed with parallel chunking"""
    min_duration_seconds = min_duration_minutes * 60
    return duration_seconds > min_duration_seconds


def get_optimal_thread_count(user_threads=None):
    """Determine optimal thread count for parallel processing"""
    if user_threads is not None:
        if user_threads < 1:
            raise ValueError("Thread count must be at least 1")
        return user_threads
    
    # Auto-detect based on CPU cores
    cpu_count = multiprocessing.cpu_count()
    # Use 75% of CPU cores, minimum 2, maximum 8 for reasonable performance
    optimal_threads = max(2, min(8, int(cpu_count * 0.75)))
    return optimal_threads


def detect_x86_features():
    """Detect x86 CPU features for optimization"""
    features = []
    
    try:
        # Try using cpuinfo if available
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        flags = info.get('flags', [])
        
        # Check for important SIMD instructions
        if 'avx512' in flags or 'avx512f' in flags:
            features.append('avx512')
        if 'avx2' in flags:
            features.append('avx2')
        elif 'avx' in flags:
            features.append('avx')
        if 'sse4_2' in flags:
            features.append('sse4.2')
            
    except ImportError:
        # Fallback to platform-specific detection
        import platform
        
        if platform.system() == 'Linux':
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    if 'avx512' in cpuinfo:
                        features.append('avx512')
                    if 'avx2' in cpuinfo:
                        features.append('avx2')
                    elif 'avx' in cpuinfo:
                        features.append('avx')
            except:
                pass
                
        elif platform.system() == 'Windows':
            # Windows detection would require WMI or other methods
            pass
    
    return features


def detect_device_capabilities():
    """Detect and report GPU/device capabilities for optimal performance"""
    import platform
    
    capabilities = {
        'platform': platform.system(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'cpu_count': multiprocessing.cpu_count(),
        'cpu_brand': '',
        'has_mps': False,
        'has_cuda': False,
        'has_rocm': False,
        'has_openvino': False,
        'gpu_name': '',
        'gpu_memory_mb': 0,
        'system_memory_gb': 0,
        'recommended_device': 'cpu',
        'recommended_compute_type': 'float32',
        'cpu_features': [],
        'optimization_hints': []
    }
    
    # Get system memory
    try:
        import psutil
        capabilities['system_memory_gb'] = psutil.virtual_memory().total // (1024**3)
    except ImportError:
        # Fallback memory detection
        try:
            if platform.system() == 'Darwin':
                result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
                if result.returncode == 0:
                    mem_bytes = int(result.stdout.split(':')[1].strip())
                    capabilities['system_memory_gb'] = mem_bytes // (1024**3)
        except:
            pass
    
    # Detect CPU features
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        # Apple Silicon detection
        try:
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                capabilities['cpu_brand'] = result.stdout.strip()
        except:
            pass
        
        # Get actual GPU capabilities at runtime
        gpu_info = get_apple_silicon_gpu_info()
        capabilities['gpu_cores'] = gpu_info['gpu_cores']
        capabilities['performance_cores'] = gpu_info['performance_cores']
        capabilities['efficiency_cores'] = gpu_info['efficiency_cores']
        
        # Provide optimization hints based on actual capabilities
        if gpu_info['gpu_cores'] > 0:
            capabilities['optimization_hints'].append(f'Apple Silicon GPU detected: {gpu_info["gpu_cores"]} cores')
            if gpu_info['gpu_cores'] >= 32:
                capabilities['optimization_hints'].append('High-performance GPU: Maximum optimization available')
            elif gpu_info['gpu_cores'] >= 16:
                capabilities['optimization_hints'].append('Pro-level GPU: Enhanced optimization available')
            else:
                capabilities['optimization_hints'].append('Efficient GPU: Balanced optimization available')
        
        if gpu_info['performance_cores'] > 0:
            capabilities['optimization_hints'].append(f'CPU: {gpu_info["performance_cores"]} performance + {gpu_info["efficiency_cores"]} efficiency cores')
        
        capabilities['cpu_features'] = ['neon', 'arm64']
    elif platform.machine() in ['x86_64', 'AMD64']:
        # x86 CPU feature detection
        capabilities['cpu_features'] = detect_x86_features()
    
    # Check for PyTorch MPS (Metal Performance Shaders) support
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            capabilities['has_mps'] = True
            capabilities['recommended_device'] = 'auto'  # Let faster-whisper auto-detect Metal
            capabilities['gpu_name'] = 'Apple Silicon GPU (Metal)'
            # For M-series chips with MPS, use float32 for stability initially
            # Can be optimized to float16 later if testing shows it's stable
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                capabilities['recommended_compute_type'] = 'float32'
                capabilities['optimization_hints'].append('Metal Performance Shaders available for GPU acceleration')
    except ImportError:
        # On Apple Silicon, faster-whisper can still use Metal via CoreML even without PyTorch
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            # faster-whisper automatically uses CoreML on Apple Silicon
            capabilities['has_mps'] = True  # We use MPS flag to indicate Metal availability
            capabilities['recommended_device'] = 'auto'  # faster-whisper will auto-detect
            capabilities['gpu_name'] = 'Apple Silicon GPU (CoreML)'
            capabilities['recommended_compute_type'] = 'float32'
            capabilities['optimization_hints'].append('CoreML/Metal acceleration available via faster-whisper')
            
        # Check if CoreML is available as additional indicator
        if platform.system() == "Darwin":
            try:
                import coremltools
                capabilities['optimization_hints'].append('CoreML tools installed for model optimization')
            except ImportError:
                pass
    
    # Check for CUDA support
    try:
        import torch
        if torch.cuda.is_available():
            capabilities['has_cuda'] = True
            capabilities['recommended_device'] = 'cuda'
            device_props = torch.cuda.get_device_properties(0)
            capabilities['gpu_name'] = device_props.name
            capabilities['gpu_memory_mb'] = device_props.total_memory // (1024 * 1024)
            capabilities['recommended_compute_type'] = 'float16'
            capabilities['optimization_hints'].append(f'CUDA GPU detected: {device_props.name}')
            
            # Check compute capability for optimization hints
            compute_capability = f"{device_props.major}.{device_props.minor}"
            if float(compute_capability) >= 7.0:
                capabilities['optimization_hints'].append('Tensor Core support available for faster inference')
    except (ImportError, AssertionError):
        pass
    
    # Check for AMD ROCm support
    try:
        import torch
        if hasattr(torch, 'hip') and torch.hip.is_available():
            capabilities['has_rocm'] = True
            capabilities['optimization_hints'].append('AMD ROCm GPU acceleration available')
    except:
        pass
    
    # Check for Intel OpenVINO support
    try:
        import openvino
        capabilities['has_openvino'] = True
        capabilities['optimization_hints'].append('Intel OpenVINO acceleration available')
    except ImportError:
        pass
    
    # Fallback compute type selection
    if not capabilities['has_mps'] and not capabilities['has_cuda']:
        capabilities['recommended_compute_type'] = 'float32'  # Safer for CPU
        
        # CPU-specific optimizations
        if 'avx2' in capabilities['cpu_features']:
            capabilities['optimization_hints'].append('AVX2 instructions available for CPU optimization')
        elif 'avx' in capabilities['cpu_features']:
            capabilities['optimization_hints'].append('AVX instructions available for CPU optimization')
    
    return capabilities


def get_optimal_compute_type(override=None):
    """Determine optimal compute type to avoid warnings"""
    if override and override != "auto":
        return override
    capabilities = detect_device_capabilities()
    return capabilities['recommended_compute_type']


def get_optimal_device(override=None):
    """Determine optimal device for faster-whisper"""
    if override and override != "auto":
        # Validate override against system capabilities
        capabilities = detect_device_capabilities()
        if override == "cuda" and not capabilities['has_cuda']:
            print(f"⚠️  Warning: CUDA requested but not available, falling back to auto-detect")
            return capabilities['recommended_device']
        elif override == "mps" and not capabilities['has_mps']:
            print(f"⚠️  Warning: MPS requested but not available, falling back to auto-detect")
            return capabilities['recommended_device']
        return override
    capabilities = detect_device_capabilities()
    return capabilities['recommended_device']


def get_gpu_memory_info():
    """Get GPU memory information for optimization"""
    memory_info = {
        'total_mb': 0,
        'available_mb': 0,
        'has_unified_memory': False,
        'gpu_cores': 0,
        'memory_bandwidth_gbps': 0
    }
    
    try:
        import torch
        if torch.backends.mps.is_available():
            # M-series chips have unified memory architecture
            memory_info['has_unified_memory'] = True
            # For M-series, we estimate based on system memory
            # since GPU and CPU share unified memory
            try:
                import psutil
                total_ram = psutil.virtual_memory().total // (1024 * 1024)
                # Estimate 60% of system RAM is available for GPU tasks
                memory_info['total_mb'] = int(total_ram * 0.6)
                memory_info['available_mb'] = int(total_ram * 0.4)  # Conservative estimate
            except ImportError:
                # Fallback estimates for M4 (typical configs)
                memory_info['total_mb'] = 12288  # 12GB estimate
                memory_info['available_mb'] = 8192   # 8GB conservative
            
            # Get Apple Silicon GPU info
            gpu_info = get_apple_silicon_gpu_info()
            memory_info['gpu_cores'] = gpu_info['gpu_cores']
        elif torch.cuda.is_available():
            # CUDA GPU memory
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            memory_info['total_mb'] = properties.total_memory // (1024 * 1024)
            memory_info['available_mb'] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            # Get compute capability
            memory_info['compute_capability'] = f"{properties.major}.{properties.minor}"
            memory_info['gpu_cores'] = properties.multi_processor_count
    except ImportError:
        # Without torch, still try to get Apple Silicon info on macOS
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            memory_info['has_unified_memory'] = True
            # Get system memory
            try:
                import psutil
                total_ram = psutil.virtual_memory().total // (1024 * 1024)
                memory_info['total_mb'] = int(total_ram * 0.6)
                memory_info['available_mb'] = int(total_ram * 0.4)
            except:
                pass
            # Get GPU info
            gpu_info = get_apple_silicon_gpu_info()
            memory_info['gpu_cores'] = gpu_info['gpu_cores']
    
    return memory_info


def get_apple_silicon_gpu_info():
    """Get Apple Silicon GPU core count and capabilities at runtime"""
    gpu_info = {
        'gpu_cores': 0,
        'performance_cores': 0,
        'efficiency_cores': 0,
        'neural_engine_cores': 0,
        'memory_bandwidth_gbps': 0
    }
    
    try:
        # Get GPU core count using system_profiler (plain text is more reliable)
        result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            import re
            # Look for "Total Number of Cores: X" pattern
            cores_match = re.search(r'Total Number of Cores:\s*(\d+)', result.stdout)
            if cores_match:
                gpu_info['gpu_cores'] = int(cores_match.group(1))
            else:
                # Alternative pattern for older macOS versions
                cores_match = re.search(r'(\d+)[-\s]core', result.stdout.lower())
                if cores_match:
                    gpu_info['gpu_cores'] = int(cores_match.group(1))
    except:
        pass
    
    # Try alternative method using ioreg
    if gpu_info['gpu_cores'] == 0:
        try:
            result = subprocess.run(['ioreg', '-l', '-w0'], capture_output=True, text=True)
            if result.returncode == 0:
                # Look for GPU core information in ioreg output
                import re
                # This is a simplified search - actual parsing would be more complex
                gpu_matches = re.findall(r'gpu-core-count["\s]=\s*(\d+)', result.stdout)
                if gpu_matches:
                    gpu_info['gpu_cores'] = int(gpu_matches[0])
        except:
            pass
    
    # Get performance metrics using sysctl
    try:
        # Get performance and efficiency core counts
        perf_result = subprocess.run(['sysctl', '-n', 'hw.perflevel0.physicalcpu'], 
                                   capture_output=True, text=True)
        if perf_result.returncode == 0:
            gpu_info['performance_cores'] = int(perf_result.stdout.strip())
        
        eff_result = subprocess.run(['sysctl', '-n', 'hw.perflevel1.physicalcpu'], 
                                  capture_output=True, text=True)
        if eff_result.returncode == 0:
            gpu_info['efficiency_cores'] = int(eff_result.stdout.strip())
    except:
        pass
    
    return gpu_info


def get_model_memory_requirements(model_size):
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


def calculate_optimal_batch_size(model_size, available_memory_mb, device_type="cpu"):
    """Calculate optimal batch size based on available memory"""
    model_memory = get_model_memory_requirements(model_size)
    
    # Reserve memory for system and other processes
    if device_type in ["cuda", "mps"]:
        reserved_memory = 2048  # Reserve 2GB for GPU
    else:
        reserved_memory = 4096  # Reserve 4GB for system on CPU
    
    usable_memory = max(0, available_memory_mb - reserved_memory)
    
    # Calculate batch size based on available memory
    # Each batch needs approximately model_memory + overhead
    overhead_factor = 1.5  # Account for activations and temporary buffers
    memory_per_batch = model_memory * overhead_factor
    
    max_batch_size = max(1, int(usable_memory / memory_per_batch))
    
    # Cap batch size for practical reasons
    if device_type in ["cuda", "mps"]:
        return min(max_batch_size, 16)  # GPU can handle larger batches
    else:
        return min(max_batch_size, 4)   # CPU should use smaller batches


def calculate_optimal_chunk_size(duration_seconds, num_threads, gpu_memory_mb, default_minutes=10, model_size="base"):
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


class ModelPool:
    """Pool of Whisper models for efficient parallel processing"""
    
    def __init__(self, model_size, device, compute_type, pool_size):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.pool_size = pool_size
        self.models = Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        self._initialized = False
        self._use_cache = True  # Enable cache integration
        
    def initialize(self):
        """Initialize the model pool using cached models"""
        if self._initialized:
            return
            
        with self.lock:
            if self._initialized:
                return
                
            print(f"📦 Initializing model pool with {self.pool_size} instances...")
            start_time = time.time()
            
            # Check if we can reuse cached model
            cache_stats = _global_model_cache.get_cache_stats()
            cache_key = f"{self.model_size}_{self.device}_{self.compute_type}"
            
            if cache_stats['cached_models'] > 0:
                print(f"🔄 Leveraging cached model for pool initialization...")
            
            for i in range(self.pool_size):
                try:
                    # Use cached model for faster pool initialization
                    model = initialize_whisper_model(
                        self.model_size, self.device, self.compute_type, use_cache=self._use_cache
                    )
                    self.models.put(model)
                    print(f"✓ Model {i+1}/{self.pool_size} ready")
                except Exception as e:
                    print(f"⚠️  Failed to load model {i+1}: {e}")
                    # Put None as placeholder to maintain pool size
                    self.models.put(None)
            
            load_time = time.time() - start_time
            print(f"✓ Model pool initialized in {load_time:.1f}s")
            self._initialized = True
    
    def get_model(self, timeout=30):
        """Get a model from the pool"""
        try:
            model = self.models.get(timeout=timeout)
            if model is None:
                # Fallback: create a new model if pool had failures
                model = initialize_whisper_model(self.model_size, self.device, self.compute_type)
            return model
        except Empty:
            # Emergency fallback: create new model if pool is exhausted
            print("⚠️  Model pool exhausted, creating temporary model")
            return initialize_whisper_model(self.model_size, self.device, self.compute_type)
    
    def return_model(self, model):
        """Return a model to the pool"""
        try:
            self.models.put_nowait(model)
        except:
            # Pool is full, model will be garbage collected
            pass
    
    def cleanup(self):
        """Clean up all models in the pool"""
        while not self.models.empty():
            try:
                model = self.models.get_nowait()
                if model is not None:
                    del model
            except Empty:
                break


def monitor_system_resources():
    """Monitor system resources for optimization feedback"""
    resource_info = {
        'cpu_percent': 0,
        'memory_percent': 0,
        'memory_available_gb': 0
    }
    
    try:
        import psutil
        resource_info['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        resource_info['memory_percent'] = memory.percent
        resource_info['memory_available_gb'] = memory.available / (1024**3)
    except ImportError:
        pass
    
    return resource_info


def optimize_thread_count_for_gpu(requested_threads, gpu_memory_mb, model_size="base"):
    """Optimize thread count based on GPU memory constraints"""
    
    # Estimate memory usage per model instance
    model_memory_estimates = {
        "tiny": 150,   # MB
        "base": 500,   # MB  
        "small": 1000, # MB
        "medium": 2500, # MB
        "large": 4500   # MB
    }
    
    model_memory_mb = model_memory_estimates.get(model_size, 500)
    
    if gpu_memory_mb > 0:
        # Calculate max threads based on available GPU memory
        # Leave 2GB buffer for system and other processes
        available_memory = max(0, gpu_memory_mb - 2048)
        max_threads_by_memory = max(1, available_memory // model_memory_mb)
        
        # Don't exceed requested threads, but warn if memory constrained
        optimal_threads = min(requested_threads, max_threads_by_memory)
        
        if optimal_threads < requested_threads:
            print(f"⚠️  GPU memory constraint: Using {optimal_threads} threads instead of {requested_threads}")
            print(f"   Model memory: ~{model_memory_mb}MB x {requested_threads} = {model_memory_mb * requested_threads}MB")
            print(f"   Available GPU memory: ~{gpu_memory_mb}MB")
        
        return optimal_threads
    
    return requested_threads


def get_optimal_transcription_params(device_capabilities, model_size="base", audio_duration_seconds=0):
    """Get optimal transcription parameters for the detected hardware"""
    params = {
        'beam_size': 5,
        'vad_filter': True,
        'vad_parameters': dict(min_silence_duration_ms=1000),
        'temperature': 0,
        'compression_ratio_threshold': 2.4,
        'no_speech_threshold': 0.6,
        'condition_on_previous_text': True,
        'log_prob_threshold': -1.0
    }
    
    # Get system memory for additional optimizations
    system_memory_gb = device_capabilities.get('system_memory_gb', 8)
    
    # Optimize for Apple Silicon with Metal
    if device_capabilities.get('has_mps'):
        # Use runtime-detected GPU cores for optimization
        gpu_cores = device_capabilities.get('gpu_cores', 0)
        perf_cores = device_capabilities.get('performance_cores', 0)
        
        # Scale parameters based on actual GPU core count
        if gpu_cores >= 64:  # Ultra-level (M1 Ultra has 64, M2 Ultra has 76)
            params['beam_size'] = 14 if model_size in ['tiny', 'base'] else 12
            params['best_of'] = 6
        elif gpu_cores >= 32:  # Max-level (M1 Max = 32, M2 Max = 38, M3 Max = 40)
            params['beam_size'] = 12 if model_size in ['tiny', 'base'] else 10
            params['best_of'] = 5
        elif gpu_cores >= 16:  # Pro-level (M1 Pro = 16, M2 Pro = 19, M3 Pro = 18)
            params['beam_size'] = 10 if model_size in ['tiny', 'base'] else 8
            params['best_of'] = 4
        elif gpu_cores >= 10:  # Base M2/M3 (M2 = 10, M3 = 10)
            params['beam_size'] = 9 if model_size in ['tiny', 'base'] else 7
            params['best_of'] = 3
        elif gpu_cores >= 8:   # Base M1 (M1 = 8)
            params['beam_size'] = 8 if model_size in ['tiny', 'base'] else 6
            params['best_of'] = 3
        else:
            # Unknown/fallback: use detected cores or conservative settings
            if gpu_cores > 0:
                # Scale linearly with GPU cores
                params['beam_size'] = min(5 + gpu_cores // 4, 12)
            else:
                params['beam_size'] = 7 if model_size in ['tiny', 'base'] else 5
                
        # Further optimize based on performance core count
        if perf_cores >= 8:  # High-performance CPU (Pro/Max/Ultra)
            params['beam_size'] = min(params['beam_size'] + 1, 14)
        
        # Adjust for system memory availability
        if system_memory_gb >= 64:  # Ultra systems typically have 64GB+
            params['beam_size'] = min(params['beam_size'] + 1, 14)
            
        # Optimize VAD for Metal (more aggressive silence detection)
        params['vad_parameters'] = dict(
            min_silence_duration_ms=500,
            speech_pad_ms=400,
            threshold=0.5
        )
        
        # Enable word timestamps for better segmentation on long files
        if audio_duration_seconds > 1800:  # 30+ minutes
            params['word_timestamps'] = True
        
        # Lower temperature for consistent results
        params['temperature'] = 0.0
        
    # Optimize for NVIDIA CUDA
    elif device_capabilities.get('has_cuda'):
        gpu_memory_mb = device_capabilities.get('gpu_memory_mb', 0)
        
        # Scale beam size based on GPU memory
        if gpu_memory_mb >= 8192:  # 8GB+ VRAM
            params['beam_size'] = 10
            params['best_of'] = 5
        elif gpu_memory_mb >= 4096:  # 4GB+ VRAM
            params['beam_size'] = 8
            params['best_of'] = 3
        else:
            params['beam_size'] = 6
            
        # CUDA-specific optimizations
        params['temperature'] = 0.0
        
        # Check for Tensor Core support
        if 'Tensor Core support' in device_capabilities.get('optimization_hints', []):
            # Can use more aggressive settings with Tensor Cores
            params['beam_size'] = min(params['beam_size'] + 2, 12)
            
    # Optimize for AMD ROCm
    elif device_capabilities.get('has_rocm'):
        params['beam_size'] = 6
        params['temperature'] = 0.1
        
    # CPU optimizations
    else:
        cpu_features = device_capabilities.get('cpu_features', [])
        
        # Scale based on CPU features
        if 'avx512' in cpu_features:
            params['beam_size'] = 5
            params['best_of'] = 2
        elif 'avx2' in cpu_features:
            params['beam_size'] = 4
        else:
            params['beam_size'] = 3
            
        # Conservative VAD for CPU
        params['vad_parameters'] = dict(
            min_silence_duration_ms=1500,
            speech_pad_ms=500
        )
        
        # Disable word timestamps on CPU for performance
        params['word_timestamps'] = False
        
    # Memory-based optimizations
    if system_memory_gb < 8:
        # Low memory: reduce beam size
        params['beam_size'] = max(2, params['beam_size'] - 2)
        params.pop('best_of', None)  # Remove best_of to save memory
    elif system_memory_gb >= 32:
        # High memory: can be more aggressive
        params['beam_size'] = min(params['beam_size'] + 1, 12)
        
    # Model-specific adjustments
    if model_size == 'large':
        # Large model: slightly reduce beam size for speed
        params['beam_size'] = max(3, params['beam_size'] - 1)
    
    return params


class GlobalModelCache:
    """Global cache for Whisper models to avoid repeated loading"""
    
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
        self._access_count = {}
        self._last_access = {}
        
        # Register cleanup on exit
        atexit.register(self.cleanup_all)
    
    def _get_cache_key(self, model_size, device, compute_type):
        """Generate cache key for model configuration"""
        return f"{model_size}_{device}_{compute_type}"
    
    def get_model(self, model_size, device="auto", compute_type="auto"):
        """Get model from cache or create new one"""
        cache_key = self._get_cache_key(model_size, device, compute_type)
        
        with self._lock:
            if cache_key in self._cache:
                model = self._cache[cache_key]
                if model is not None:
                    # Update access tracking
                    self._access_count[cache_key] = self._access_count.get(cache_key, 0) + 1
                    self._last_access[cache_key] = time.time()
                    print(f"📦 Using cached model '{model_size}' (accessed {self._access_count[cache_key]} times)")
                    return model
            
            # Model not in cache or invalid, create new one
            print(f"🔄 Loading and caching model '{model_size}'...")
            start_time = time.time()
            
            try:
                model = self._create_model(model_size, device, compute_type)
                self._cache[cache_key] = model
                self._access_count[cache_key] = 1
                self._last_access[cache_key] = time.time()
                
                load_time = time.time() - start_time
                print(f"✓ Model cached in {load_time:.1f}s")
                return model
                
            except Exception as e:
                print(f"⚠️  Failed to cache model: {e}")
                # Don't cache failed models
                return self._create_model(model_size, device, compute_type)
    
    def _create_model(self, model_size, device, compute_type):
        """Create a new model instance"""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("faster-whisper is not installed. Run: pip install faster-whisper")
        
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            return model
        except Exception as e:
            print(f"⚠️  Falling back to CPU due to: {e}")
            return WhisperModel(model_size, device="cpu", compute_type="float32")
    
    def preload_model(self, model_size, device="auto", compute_type="auto"):
        """Preload a model for faster access later"""
        cache_key = self._get_cache_key(model_size, device, compute_type)
        
        with self._lock:
            if cache_key not in self._cache:
                print(f"🚀 Pre-warming model '{model_size}'...")
                self.get_model(model_size, device, compute_type)
    
    def get_cache_stats(self):
        """Get cache statistics"""
        with self._lock:
            return {
                'cached_models': len(self._cache),
                'total_accesses': sum(self._access_count.values()),
                'models': {k: {'accesses': self._access_count.get(k, 0), 
                              'last_access': self._last_access.get(k, 0)} 
                          for k in self._cache.keys()}
            }
    
    def cleanup_old_models(self, max_age_seconds=3600):
        """Clean up models that haven't been accessed recently"""
        current_time = time.time()
        to_remove = []
        
        with self._lock:
            for cache_key, last_access in self._last_access.items():
                if current_time - last_access > max_age_seconds:
                    to_remove.append(cache_key)
            
            for cache_key in to_remove:
                if cache_key in self._cache:
                    print(f"🧹 Cleaning up old model: {cache_key}")
                    del self._cache[cache_key]
                    del self._access_count[cache_key]
                    del self._last_access[cache_key]
    
    def cleanup_all(self):
        """Clean up all cached models"""
        with self._lock:
            self._cache.clear()
            self._access_count.clear()
            self._last_access.clear()


# Global model cache instance
_global_model_cache = GlobalModelCache()


def preload_model_for_config(model_size, device=None, compute_type=None):
    """Preload a model based on system configuration for faster startup"""
    if device is None:
        device = get_optimal_device()
    if compute_type is None:
        compute_type = get_optimal_compute_type()
    
    print(f"🚀 Pre-loading model '{model_size}' for faster startup...")
    _global_model_cache.preload_model(model_size, device, compute_type)


def get_model_cache_stats():
    """Get current model cache statistics"""
    return _global_model_cache.get_cache_stats()


def suggest_model_preload(device_capabilities, model_size="base"):
    """Suggest whether to preload models based on system capabilities"""
    # For GPU systems with good performance, preloading is beneficial
    if device_capabilities.get('has_mps') or device_capabilities.get('has_cuda'):
        return True
    
    # For CPU systems, preloading might not be worth the startup cost
    return False


def report_device_capabilities(capabilities=None, verbose=False):
    """Report detected device capabilities to user"""
    if capabilities is None:
        capabilities = detect_device_capabilities()
    
    print(f"🖥️  System: {capabilities['platform']} {capabilities['machine']}")
    
    # Show CPU info
    if capabilities['cpu_brand']:
        print(f"🔧 CPU: {capabilities['cpu_brand']} ({capabilities['cpu_count']} cores)")
    else:
        print(f"🔧 CPU: {capabilities['processor']} ({capabilities['cpu_count']} cores)")
    
    # Show memory info
    if capabilities['system_memory_gb'] > 0:
        print(f"💾 Memory: {capabilities['system_memory_gb']}GB system RAM")
    
    # Get GPU memory info for detailed reporting
    memory_info = get_gpu_memory_info()
    
    # Show GPU capabilities
    if capabilities['has_mps']:
        gpu_cores = capabilities.get('gpu_cores', 0)
        if gpu_cores > 0:
            print(f"⚡ GPU: {capabilities['gpu_name']} ({gpu_cores} cores)")
        else:
            print(f"⚡ GPU: {capabilities['gpu_name']}")
        if memory_info['has_unified_memory'] and memory_info['available_mb'] > 0:
            print(f"🧠 Unified Memory: ~{memory_info['available_mb']//1024}GB available for GPU tasks")
        print(f"🎯 Using GPU acceleration via Metal")
    elif capabilities['has_cuda']:
        print(f"⚡ GPU: {capabilities['gpu_name']} ({capabilities['gpu_memory_mb']}MB)")
        print(f"🎯 Using GPU acceleration via CUDA")
    elif capabilities['has_rocm']:
        print(f"⚡ GPU: AMD GPU with ROCm support")
        print(f"🎯 Using GPU acceleration via ROCm")
    else:
        print("💻 GPU: Not available - Using optimized CPU processing")
        if capabilities['platform'] == "Darwin" and capabilities['machine'] == "arm64":
            print("💡 Tip: Install PyTorch with MPS support for 2-4x speedup")
    
    # Show optimization configuration
    print(f"⚙️  Config: device={capabilities['recommended_device']}, compute={capabilities['recommended_compute_type']}")
    
    # Show optimization hints
    if verbose and capabilities['optimization_hints']:
        print("\n📋 Optimization hints:")
        for hint in capabilities['optimization_hints']:
            print(f"   • {hint}")
    
    print()


def create_audio_chunks(input_file, chunk_duration_minutes, temp_dir):
    """Split audio file into chunks using ffmpeg"""
    chunk_duration_seconds = chunk_duration_minutes * 60
    chunks = []
    
    try:
        # Get total duration
        total_duration = get_audio_duration(input_file)
        num_chunks = math.ceil(total_duration / chunk_duration_seconds)
        
        print(f"Splitting audio into {num_chunks} chunks of {chunk_duration_minutes} minutes each...")
        
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
            
            print(f"✓ Created chunk {i+1}/{num_chunks}")
        
        return chunks
        
    except Exception as e:
        # Clean up any created chunks on error
        for chunk in chunks:
            if chunk['file'].exists():
                chunk['file'].unlink()
        raise RuntimeError(f"Failed to create audio chunks: {e}")


def transcribe_chunk_with_pool(chunk_info, model_pool, transcription_params=None):
    """Transcribe a single audio chunk using model pool"""
    model = None
    try:
        # Get model from pool
        model = model_pool.get_model()
        
        # Use optimized parameters if provided, otherwise use defaults
        if transcription_params is None:
            transcription_params = {
                'beam_size': 5,
                'vad_filter': True,
                'vad_parameters': dict(min_silence_duration_ms=1000)
            }
        
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
    finally:
        # Always return model to pool
        if model is not None:
            model_pool.return_model(model)


def transcribe_chunk(chunk_info, model_size, device=None, compute_type=None):
    """Transcribe a single audio chunk (legacy function for backward compatibility)"""
    try:
        # Initialize model for this worker thread with optimal settings
        model = initialize_whisper_model(model_size, device, compute_type)
        
        # Transcribe the chunk
        segments, info = model.transcribe(
            str(chunk_info['file']),
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=1000)
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
        
        return {
            'chunk_index': chunk_info['index'],
            'segments': chunk_segments,
            'language': info.language,
            'language_probability': info.language_probability
        }
        
    except Exception as e:
        return {
            'chunk_index': chunk_info['index'],
            'error': str(e),
            'segments': []
        }


def assemble_transcripts(chunk_results):
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


def transcribe_audio_parallel(model_size, input_path, output_path, chunk_duration_minutes, num_threads, device=None, compute_type=None):
    """Transcribe audio file using parallel chunk processing with model pooling"""
    temp_dir = None
    model_pool = None
    
    try:
        # Create temporary directory for chunks
        temp_dir = tempfile.mkdtemp(prefix="whisper_chunks_")
        
        # Get GPU memory info for optimization
        memory_info = get_gpu_memory_info()
        
        # Calculate optimal chunk size based on system capabilities
        duration = get_audio_duration(input_path)
        optimal_chunk_size = calculate_optimal_chunk_size(
            duration, num_threads, memory_info['available_mb'], chunk_duration_minutes
        )
        
        print(f"Using {num_threads} threads for parallel processing")
        print(f"Optimal chunk size: {optimal_chunk_size} minutes (requested: {chunk_duration_minutes})")
        if memory_info['has_unified_memory']:
            print(f"🧠 GPU Memory: ~{memory_info['available_mb']//1024}GB unified memory available")
        
        # Use optimal chunk size
        final_chunk_size = optimal_chunk_size
        
        # Create audio chunks
        chunks = create_audio_chunks(input_path, final_chunk_size, temp_dir)
        
        # Get optimized transcription parameters for this hardware
        capabilities = detect_device_capabilities()
        transcription_params = get_optimal_transcription_params(capabilities, model_size, duration)
        print(f"🎯 Optimized parameters: beam_size={transcription_params['beam_size']}, VAD threshold={transcription_params['vad_parameters']['min_silence_duration_ms']}ms")
        
        # Pre-warm the model cache if beneficial (before creating pool)
        cache_stats = get_model_cache_stats()
        if cache_stats['cached_models'] == 0 and suggest_model_preload(capabilities, model_size):
            print("🚀 Pre-warming model cache for parallel processing...")
            preload_model_for_config(model_size, device, compute_type)
        
        # Initialize model pool for efficient GPU utilization
        # Use smaller pool size to avoid memory exhaustion
        pool_size = min(num_threads, 3)  # Limit to 3 models max for GPU efficiency
        model_pool = ModelPool(model_size, device, compute_type, pool_size)
        model_pool.initialize()
        
        print(f"\nProcessing {len(chunks)} chunks in parallel...")
        start_time = time.time()
        
        # Process chunks in parallel using model pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all chunk transcription tasks
            future_to_chunk = {
                executor.submit(transcribe_chunk_with_pool, chunk, model_pool, transcription_params): chunk
                for chunk in chunks
            }
            
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
                    print(f"✓ Chunk {chunk['index'] + 1}/{len(chunks)} completed ({progress:.1f}%) - Elapsed: {format_time(elapsed)}")
                    
                except Exception as e:
                    print(f"✗ Chunk {chunk['index'] + 1} failed: {e}")
                    chunk_results.append({
                        'chunk_index': chunk['index'],
                        'error': str(e),
                        'segments': []
                    })
        
        # Assemble final transcript
        print("\nAssembling final transcript...")
        all_segments, language_info = assemble_transcripts(chunk_results)
        
        # Write assembled transcript to file
        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in all_segments:
                f.write(segment['text'])
                f.write('\n')
        
        total_time = time.time() - start_time
        total_duration = sum(chunk['duration'] for chunk in chunks)
        speed_ratio = total_duration / total_time if total_time > 0 else 0
        
        print(f"✓ Parallel transcription completed!")
        if language_info:
            print(f"  Language: {language_info['language']} (probability: {language_info['language_probability']:.2f})")
        print(f"  Audio duration: {format_time(total_duration)}")
        print(f"  Processing time: {format_time(total_time)}")
        print(f"  Speed ratio: {speed_ratio:.1f}x faster than real-time")
        print(f"  Total segments: {sum(len(r['segments']) for r in chunk_results)}")
        print(f"  Output saved to: {output_path}")
        
    finally:
        # Clean up model pool
        if model_pool:
            model_pool.cleanup()
        
        # Clean up temporary directory
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)


def initialize_whisper_model(model_size, device=None, compute_type=None, use_cache=True):
    """Initialize Whisper model with optimal device and compute type selection"""
    # Use optimal device and compute type if not specified
    if device is None:
        device = get_optimal_device()
    if compute_type is None:
        compute_type = get_optimal_compute_type()
    
    if use_cache:
        # Use global model cache for faster access
        return _global_model_cache.get_model(model_size, device, compute_type)
    else:
        # Direct model creation (for legacy compatibility)
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("faster-whisper is not installed. Run: pip install faster-whisper")
        
        print(f"Loading Whisper model '{model_size}' (device: {device}, compute: {compute_type})...")
        start_time = time.time()
        
        try:
            # Try optimal configuration first
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as e:
            print(f"⚠️  Falling back to CPU due to: {e}")
            # Fallback to CPU with safe compute type
            model = WhisperModel(model_size, device="cpu", compute_type="float32")
        
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.1f}s")
        
        return model


def format_time(seconds):
    """Format seconds into human-readable time string"""
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


def create_progress_bar(percentage, width=40):
    """Create a visual progress bar string"""
    filled = int(width * percentage / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


def transcribe_audio(model, input_path, output_path, transcription_params=None):
    """Transcribe audio file using Whisper model with streaming segments"""
    print(f"Transcribing audio: {input_path.name}")
    start_time = time.time()
    
    # Use optimized parameters if provided, otherwise use defaults
    if transcription_params is None:
        transcription_params = {
            'beam_size': 5,
            'vad_filter': True,
            'vad_parameters': dict(min_silence_duration_ms=1000)
        }
    
    # Use streaming segments for memory efficiency with large files
    segments, info = model.transcribe(
        str(input_path),
        **transcription_params
    )
    
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    audio_duration = info.duration
    print(f"Audio duration: {format_time(audio_duration)}")
    print()  # Empty line before progress
    
    # Write transcription to file with enhanced progress tracking
    with open(output_path, 'w', encoding='utf-8') as f:
        segment_count = 0
        last_segment_end = 0
        
        for segment in segments:
            # Write segment text
            f.write(segment.text.strip())
            f.write('\n')
            
            segment_count += 1
            last_segment_end = segment.end
            
            # Calculate progress based on audio time processed
            progress_percentage = min((last_segment_end / audio_duration) * 100, 100)
            elapsed_time = time.time() - start_time
            
            # Estimate total time and remaining time
            if progress_percentage > 0:
                estimated_total_time = elapsed_time * (100 / progress_percentage)
                estimated_remaining = max(estimated_total_time - elapsed_time, 0)
            else:
                estimated_remaining = 0
            
            # Update progress bar every segment (but clear previous line)
            progress_bar = create_progress_bar(progress_percentage)
            elapsed_str = format_time(elapsed_time)
            remaining_str = format_time(estimated_remaining)
            
            # Clear line and print progress
            print(f"\r{progress_bar} {progress_percentage:.1f}% | "
                  f"Elapsed: {elapsed_str} | ETA: {remaining_str}", end="", flush=True)
    
    # Final newline and completion message
    print()  # Move to next line after progress bar
    
    total_time = time.time() - start_time
    speed_ratio = audio_duration / total_time if total_time > 0 else 0
    
    print(f"✓ Transcription completed!")
    print(f"  Audio duration: {format_time(audio_duration)}")
    print(f"  Processing time: {format_time(total_time)}")
    print(f"  Speed ratio: {speed_ratio:.1f}x faster than real-time")
    print(f"  Total segments: {segment_count}")
    print(f"  Output saved to: {output_path}")


def main():
    """Main entry point"""
    try:
        args = parse_arguments()
        
        # Handle cache stats option
        if args.cache_stats:
            stats = get_model_cache_stats()
            print("📊 Model Cache Statistics:")
            print(f"   Cached models: {stats['cached_models']}")
            print(f"   Total accesses: {stats['total_accesses']}")
            if stats['models']:
                print("   Model details:")
                for model_key, details in stats['models'].items():
                    print(f"     {model_key}: {details['accesses']} accesses")
            else:
                print("   No models currently cached")
            return 0
        
        # Detect and report device capabilities early
        capabilities = detect_device_capabilities()
        report_device_capabilities(capabilities, verbose=args.verbose)
        
        # Handle preload option
        if args.preload:
            device = get_optimal_device(args.device)
            compute_type = get_optimal_compute_type(args.compute_type)
            preload_model_for_config(args.model, device, compute_type)
            print(f"✓ Model '{args.model}' pre-loaded and cached")
            return 0
        
        # Validate required arguments for transcription
        if not args.input_file:
            print("Error: input_file is required for transcription", file=sys.stderr)
            return 1
        if not args.output:
            print("Error: -o/--output is required for transcription", file=sys.stderr)
            return 1
        
        # Validate input file
        input_path = validate_input_file(args.input_file)
        print(f"✓ Input file validated: {input_path}")
        
        # Validate output path
        output_path = validate_output_path(args.output)
        print(f"✓ Output path validated: {output_path}")
        
        # Check audio duration to decide on processing method
        duration = get_audio_duration(input_path)
        print(f"✓ Audio duration: {format_time(duration)}")
        
        # Get optimal device and compute settings with user overrides
        device = get_optimal_device(args.device)
        compute_type = get_optimal_compute_type(args.compute_type)
        
        # Determine processing method
        use_parallel = (
            not args.no_parallel and 
            should_use_parallel_processing(duration)
        )
        
        if use_parallel:
            # Get thread count and optimize for GPU memory
            initial_threads = get_optimal_thread_count(args.threads)
            memory_info = get_gpu_memory_info()
            
            # Optimize thread count based on GPU memory constraints
            num_threads = optimize_thread_count_for_gpu(
                initial_threads, memory_info['available_mb'], args.model
            )
            
            print(f"✓ Using parallel processing ({num_threads} threads)")
            
            # Process with parallel chunking
            transcribe_audio_parallel(
                args.model, input_path, output_path, 
                args.chunk_size, num_threads, device, compute_type
            )
        else:
            # Use traditional single-threaded processing
            if args.no_parallel:
                print("✓ Parallel processing disabled by user")
            else:
                print("✓ Using sequential processing (file under 30 minutes)")
            
            # Get optimized transcription parameters
            transcription_params = get_optimal_transcription_params(capabilities, args.model, duration)
            print(f"🎯 Optimized parameters: beam_size={transcription_params['beam_size']}, VAD threshold={transcription_params['vad_parameters']['min_silence_duration_ms']}ms")
            
            # Check if we should preload for future usage
            cache_stats = get_model_cache_stats()
            if cache_stats['cached_models'] == 0 and suggest_model_preload(capabilities, args.model):
                print("💡 Pre-loading model for future usage...")
            
            # Initialize Whisper model (will use cache if available)
            model = initialize_whisper_model(args.model, device, compute_type)
            
            # Transcribe audio file
            transcribe_audio(model, input_path, output_path, transcription_params)
        
        return 0
        
    except (FileNotFoundError, ValueError, PermissionError, ImportError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nTranscription interrupted by user", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())