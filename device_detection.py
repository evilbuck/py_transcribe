"""
Device detection and optimization module for faster-whisper
"""
import os
import platform
import subprocess
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class DeviceCapabilities:
    """Information about available device capabilities"""
    has_cuda: bool = False
    has_mps: bool = False  # Apple Metal Performance Shaders
    cuda_version: Optional[str] = None
    gpu_memory: Optional[int] = None  # MB
    cpu_cores: int = 1
    optimal_threads: int = 1


class DeviceDetector:
    """Detects and optimizes device configuration for faster-whisper"""
    
    def __init__(self):
        self._capabilities = None
        
    def get_capabilities(self) -> DeviceCapabilities:
        """Get cached device capabilities"""
        if self._capabilities is None:
            self._capabilities = self._detect_capabilities()
        return self._capabilities
    
    def _detect_capabilities(self) -> DeviceCapabilities:
        """Detect available device capabilities"""
        caps = DeviceCapabilities()
        
        # Detect CPU cores and optimal thread count
        caps.cpu_cores = os.cpu_count() or 1
        caps.optimal_threads = min(caps.cpu_cores, 8)  # Cap at 8 for whisper efficiency
        
        # Detect CUDA availability
        caps.has_cuda = self._check_cuda()
        if caps.has_cuda:
            caps.cuda_version = self._get_cuda_version()
            caps.gpu_memory = self._get_gpu_memory()
        
        # Detect Apple MPS (Metal Performance Shaders)
        caps.has_mps = self._check_mps()
        
        return caps
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            # Fallback: check nvidia-smi
            try:
                result = subprocess.run(['nvidia-smi'], 
                                      capture_output=True, 
                                      check=True)
                return result.returncode == 0
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False
    
    def _check_mps(self) -> bool:
        """Check if Apple MPS is available"""
        if platform.system() != "Darwin":
            return False
        
        try:
            import torch
            return torch.backends.mps.is_available()
        except (ImportError, AttributeError):
            # Fallback: check for Apple Silicon
            try:
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                      capture_output=True, text=True)
                return 'Apple' in result.stdout
            except subprocess.CalledProcessError:
                return False
    
    def _get_cuda_version(self) -> Optional[str]:
        """Get CUDA version if available"""
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    # Extract version like "V11.8.89"
                    parts = line.split()
                    for part in parts:
                        if part.startswith('V'):
                            return part[1:]  # Remove 'V' prefix
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        return None
    
    def _get_gpu_memory(self) -> Optional[int]:
        """Get GPU memory in MB"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            memory_mb = int(result.stdout.strip())
            return memory_mb
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            return None


class OptimalConfiguration:
    """Determines optimal configuration for faster-whisper based on device capabilities"""
    
    def __init__(self, detector: Optional[DeviceDetector] = None):
        self.detector = detector or DeviceDetector()
    
    def get_optimal_device(self, override: Optional[str] = None) -> str:
        """
        Determine optimal device for faster-whisper
        
        Args:
            override: User-specified device override ('cpu', 'cuda', 'mps', 'auto')
            
        Returns:
            Device string for faster-whisper
        """
        if override and override != "auto":
            return override
        
        caps = self.detector.get_capabilities()
        
        # Prefer GPU acceleration when available
        if caps.has_cuda:
            return "cuda"
        elif caps.has_mps:
            return "mps"
        else:
            return "cpu"
    
    def get_optimal_compute_type(self, device: str, override: Optional[str] = None) -> str:
        """
        Determine optimal compute type for given device
        
        Args:
            device: Target device ('cpu', 'cuda', 'mps')
            override: User-specified compute type override
            
        Returns:
            Compute type string for faster-whisper
        """
        if override and override != "auto":
            return override
        
        caps = self.detector.get_capabilities()
        
        if device == "cuda":
            # Use float16 for GPU performance, fallback to int8_float16 for low memory
            if caps.gpu_memory and caps.gpu_memory >= 4000:  # 4GB+
                return "float16"
            else:
                return "int8_float16"
        elif device == "mps":
            # Apple Metal prefers float16
            return "float16"
        else:  # CPU
            # For CPU, int8 provides good speed/quality tradeoff
            return "int8"
    
    def get_optimal_threads(self, override: Optional[int] = None) -> int:
        """
        Get optimal thread count for CPU processing
        
        Args:
            override: User-specified thread count
            
        Returns:
            Optimal thread count
        """
        if override:
            return override
        
        caps = self.detector.get_capabilities()
        return caps.optimal_threads
    
    def get_optimal_batch_size(self, device: str, model_size: str = "base") -> int:
        """
        Get optimal batch size based on device and model
        
        Args:
            device: Target device
            model_size: Whisper model size
            
        Returns:
            Optimal batch size
        """
        caps = self.detector.get_capabilities()
        
        if device == "cuda" and caps.gpu_memory:
            # Scale batch size based on GPU memory and model size
            model_memory_factor = {
                "tiny": 1,
                "base": 2, 
                "small": 4,
                "medium": 8,
                "large": 16
            }
            
            base_batch_size = max(1, caps.gpu_memory // (1000 * model_memory_factor.get(model_size, 8)))
            return min(base_batch_size, 32)  # Cap at 32
        elif device == "mps":
            # Apple Metal - conservative batch sizes
            return {"tiny": 16, "base": 8, "small": 4, "medium": 2, "large": 1}.get(model_size, 4)
        else:  # CPU
            # CPU processing - smaller batches
            return {"tiny": 4, "base": 2, "small": 1, "medium": 1, "large": 1}.get(model_size, 1)
    
    def get_configuration_summary(self, device: str, compute_type: str, threads: int, batch_size: int) -> Dict[str, Any]:
        """Get a summary of the current configuration"""
        caps = self.detector.get_capabilities()
        
        return {
            "device": device,
            "compute_type": compute_type,
            "threads": threads,
            "batch_size": batch_size,
            "capabilities": {
                "has_cuda": caps.has_cuda,
                "has_mps": caps.has_mps,
                "cuda_version": caps.cuda_version,
                "gpu_memory_mb": caps.gpu_memory,
                "cpu_cores": caps.cpu_cores
            }
        }


def setup_environment(threads: int) -> None:
    """
    Set up optimal environment variables for performance
    
    Args:
        threads: Number of threads to use for CPU processing
    """
    # Set OpenMP thread count for faster-whisper
    os.environ["OMP_NUM_THREADS"] = str(threads)
    
    # Additional performance optimizations
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)