"""
Device detection and optimization for audio transcription
"""
import platform
import subprocess
import multiprocessing
from typing import Dict, List, Any

from .logger import get_logger


def detect_x86_features() -> List[str]:
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


def get_apple_silicon_gpu_info() -> Dict[str, int]:
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


def detect_device_capabilities() -> Dict[str, Any]:
    """Detect and report GPU/device capabilities for optimal performance"""
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
        capabilities.update(gpu_info)
        
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


def get_optimal_device(override: str = None) -> str:
    """Determine optimal device for faster-whisper"""
    if override and override != "auto":
        # Validate override against system capabilities
        capabilities = detect_device_capabilities()
        if override == "cuda" and not capabilities['has_cuda']:
            logger = get_logger()
            logger.warning("CUDA requested but not available, falling back to auto-detect")
            return capabilities['recommended_device']
        elif override == "mps" and not capabilities['has_mps']:
            logger = get_logger()
            logger.warning("MPS requested but not available, falling back to auto-detect")
            return capabilities['recommended_device']
        return override
    capabilities = detect_device_capabilities()
    return capabilities['recommended_device']


def get_optimal_compute_type(override: str = None) -> str:
    """Determine optimal compute type to avoid warnings"""
    if override and override != "auto":
        return override
    capabilities = detect_device_capabilities()
    return capabilities['recommended_compute_type']


def get_gpu_memory_info() -> Dict[str, Any]:
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


def get_optimal_transcription_params(device_capabilities: Dict[str, Any], model_size: str = "base", 
                                   audio_duration_seconds: float = 0) -> Dict[str, Any]:
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


def report_device_capabilities(capabilities: Dict[str, Any] = None, verbose: bool = False) -> None:
    """Report detected device capabilities to user"""
    logger = get_logger(verbose=verbose)
    
    if capabilities is None:
        capabilities = detect_device_capabilities()
    
    logger.system_info(f"System: {capabilities['platform']} {capabilities['machine']}")
    
    # Show CPU info
    if capabilities['cpu_brand']:
        logger.info(f"🔧 CPU: {capabilities['cpu_brand']} ({capabilities['cpu_count']} cores)")
    else:
        logger.info(f"🔧 CPU: {capabilities['processor']} ({capabilities['cpu_count']} cores)")
    
    # Show memory info
    if capabilities['system_memory_gb'] > 0:
        logger.info(f"💾 Memory: {capabilities['system_memory_gb']}GB system RAM")
    
    # Get GPU memory info for detailed reporting
    memory_info = get_gpu_memory_info()
    
    # Show GPU capabilities
    if capabilities['has_mps']:
        gpu_cores = capabilities.get('gpu_cores', 0)
        if gpu_cores > 0:
            logger.gpu_info(f"GPU: {capabilities['gpu_name']} ({gpu_cores} cores)")
        else:
            logger.gpu_info(f"GPU: {capabilities['gpu_name']}")
        if memory_info['has_unified_memory'] and memory_info['available_mb'] > 0:
            logger.info(f"🧠 Unified Memory: ~{memory_info['available_mb']//1024}GB available for GPU tasks")
        logger.info(f"🎯 Using GPU acceleration via Metal")
    elif capabilities['has_cuda']:
        logger.gpu_info(f"GPU: {capabilities['gpu_name']} ({capabilities['gpu_memory_mb']}MB)")
        logger.info(f"🎯 Using GPU acceleration via CUDA")
    elif capabilities['has_rocm']:
        logger.gpu_info(f"GPU: AMD GPU with ROCm support")
        logger.info(f"🎯 Using GPU acceleration via ROCm")
    else:
        logger.info("💻 GPU: Not available - Using optimized CPU processing")
        if capabilities['platform'] == "Darwin" and capabilities['machine'] == "arm64":
            logger.info("💡 Tip: Install PyTorch with MPS support for 2-4x speedup")
    
    # Show optimization configuration
    logger.optimization_info(f"Config: device={capabilities['recommended_device']}, compute={capabilities['recommended_compute_type']}")
    
    # Show optimization hints
    if verbose and capabilities['optimization_hints']:
        logger.info("\n📋 Optimization hints:")
        for hint in capabilities['optimization_hints']:
            logger.info(f"   • {hint}")
    
    logger.info("")