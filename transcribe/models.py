"""
Model caching and pool management for audio transcription
"""
import time
import threading
import atexit
import weakref
from queue import Queue, Empty
from typing import Optional, Dict, Any

from .logger import get_logger


class GlobalModelCache:
    """Global cache for Whisper models to avoid repeated loading"""
    
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
        self._access_count = {}
        self._last_access = {}
        
        # Register cleanup on exit
        atexit.register(self.cleanup_all)
    
    def _get_cache_key(self, model_size: str, device: str, compute_type: str) -> str:
        """Generate cache key for model configuration"""
        return f"{model_size}_{device}_{compute_type}"
    
    def get_model(self, model_size: str, device: str = "auto", compute_type: str = "auto"):
        """Get model from cache or create new one"""
        cache_key = self._get_cache_key(model_size, device, compute_type)
        logger = get_logger()
        
        with self._lock:
            if cache_key in self._cache:
                model = self._cache[cache_key]
                if model is not None:
                    # Update access tracking
                    self._access_count[cache_key] = self._access_count.get(cache_key, 0) + 1
                    self._last_access[cache_key] = time.time()
                    logger.model_info(f"Using cached model '{model_size}' (accessed {self._access_count[cache_key]} times)")
                    return model
            
            # Model not in cache or invalid, create new one
            logger.info(f"🔄 Loading and caching model '{model_size}'...")
            start_time = time.time()
            
            try:
                model = self._create_model(model_size, device, compute_type)
                self._cache[cache_key] = model
                self._access_count[cache_key] = 1
                self._last_access[cache_key] = time.time()
                
                load_time = time.time() - start_time
                logger.success(f"Model cached in {load_time:.1f}s")
                return model
                
            except Exception as e:
                logger.error(f"Failed to cache model: {e}")
                # Don't cache failed models
                return self._create_model(model_size, device, compute_type)
    
    def _create_model(self, model_size: str, device: str, compute_type: str):
        """Create a new model instance"""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("faster-whisper is not installed. Run: pip install faster-whisper")
        
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            return model
        except Exception as e:
            logger = get_logger()
            logger.warning(f"Falling back to CPU due to: {e}")
            return WhisperModel(model_size, device="cpu", compute_type="float32")
    
    def preload_model(self, model_size: str, device: str = "auto", compute_type: str = "auto") -> None:
        """Preload a model for faster access later"""
        cache_key = self._get_cache_key(model_size, device, compute_type)
        logger = get_logger()
        
        with self._lock:
            if cache_key not in self._cache:
                logger.info(f"🚀 Pre-warming model '{model_size}'...")
                self.get_model(model_size, device, compute_type)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                'cached_models': len(self._cache),
                'total_accesses': sum(self._access_count.values()),
                'models': {k: {'accesses': self._access_count.get(k, 0), 
                              'last_access': self._last_access.get(k, 0)} 
                          for k in self._cache.keys()}
            }
    
    def cleanup_old_models(self, max_age_seconds: int = 3600) -> None:
        """Clean up models that haven't been accessed recently"""
        current_time = time.time()
        to_remove = []
        logger = get_logger()
        
        with self._lock:
            for cache_key, last_access in self._last_access.items():
                if current_time - last_access > max_age_seconds:
                    to_remove.append(cache_key)
            
            for cache_key in to_remove:
                if cache_key in self._cache:
                    logger.info(f"🧹 Cleaning up old model: {cache_key}")
                    del self._cache[cache_key]
                    del self._access_count[cache_key]
                    del self._last_access[cache_key]
    
    def cleanup_all(self) -> None:
        """Clean up all cached models"""
        with self._lock:
            self._cache.clear()
            self._access_count.clear()
            self._last_access.clear()


class ModelPool:
    """Pool of Whisper models for efficient parallel processing"""
    
    def __init__(self, model_size: str, device: str, compute_type: str, pool_size: int):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.pool_size = pool_size
        self.models = Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        self._initialized = False
        self._use_cache = True  # Enable cache integration
        self.logger = get_logger()
        
    def initialize(self) -> None:
        """Initialize the model pool using cached models"""
        if self._initialized:
            return
            
        with self.lock:
            if self._initialized:
                return
                
            self.logger.model_info(f"Initializing model pool with {self.pool_size} instances...")
            start_time = time.time()
            
            # Check if we can reuse cached model
            cache_stats = _global_model_cache.get_cache_stats()
            cache_key = f"{self.model_size}_{self.device}_{self.compute_type}"
            
            if cache_stats['cached_models'] > 0:
                self.logger.info(f"🔄 Leveraging cached model for pool initialization...")
            
            for i in range(self.pool_size):
                try:
                    # Use cached model for faster pool initialization
                    model = _global_model_cache.get_model(
                        self.model_size, self.device, self.compute_type
                    )
                    self.models.put(model)
                    self.logger.success(f"Model {i+1}/{self.pool_size} ready")
                except Exception as e:
                    self.logger.warning(f"Failed to load model {i+1}: {e}")
                    # Put None as placeholder to maintain pool size
                    self.models.put(None)
            
            load_time = time.time() - start_time
            self.logger.success(f"Model pool initialized in {load_time:.1f}s")
            self._initialized = True
    
    def get_model(self, timeout: int = 30):
        """Get a model from the pool"""
        try:
            model = self.models.get(timeout=timeout)
            if model is None:
                # Fallback: create a new model if pool had failures
                model = _global_model_cache.get_model(self.model_size, self.device, self.compute_type)
            return model
        except Empty:
            # Emergency fallback: create new model if pool is exhausted
            self.logger.warning("Model pool exhausted, creating temporary model")
            return _global_model_cache.get_model(self.model_size, self.device, self.compute_type)
    
    def return_model(self, model) -> None:
        """Return a model to the pool"""
        try:
            self.models.put_nowait(model)
        except:
            # Pool is full, model will be garbage collected
            pass
    
    def cleanup(self) -> None:
        """Clean up all models in the pool"""
        while not self.models.empty():
            try:
                model = self.models.get_nowait()
                if model is not None:
                    del model
            except Empty:
                break


# Global model cache instance
_global_model_cache = GlobalModelCache()


def get_model_cache_stats() -> Dict[str, Any]:
    """Get current model cache statistics"""
    return _global_model_cache.get_cache_stats()


def preload_model_for_config(model_size: str, device: Optional[str] = None, 
                           compute_type: Optional[str] = None) -> None:
    """Preload a model based on system configuration for faster startup"""
    if device is None:
        try:
            from .device_detection import get_optimal_device
            device = get_optimal_device()
        except ImportError:
            device = "auto"
    
    if compute_type is None:
        try:
            from .device_detection import get_optimal_compute_type
            compute_type = get_optimal_compute_type()
        except ImportError:
            compute_type = "auto"
    
    logger = get_logger()
    logger.info(f"🚀 Pre-loading model '{model_size}' for faster startup...")
    _global_model_cache.preload_model(model_size, device, compute_type)


def suggest_model_preload(device_capabilities: Dict[str, Any], model_size: str = "base") -> bool:
    """Suggest whether to preload models based on system capabilities"""
    # For GPU systems with good performance, preloading is beneficial
    if device_capabilities.get('has_mps') or device_capabilities.get('has_cuda'):
        return True
    
    # For CPU systems, preloading might not be worth the startup cost
    return False


def initialize_whisper_model(model_size: str, device: Optional[str] = None, 
                           compute_type: Optional[str] = None, use_cache: bool = True):
    """Initialize Whisper model with optimal device and compute type selection"""
    # Use optimal device and compute type if not specified
    if device is None:
        try:
            from .device_detection import get_optimal_device
            device = get_optimal_device()
        except ImportError:
            device = "auto"
    
    if compute_type is None:
        try:
            from .device_detection import get_optimal_compute_type
            compute_type = get_optimal_compute_type()
        except ImportError:
            compute_type = "auto"
    
    if use_cache:
        # Use global model cache for faster access
        return _global_model_cache.get_model(model_size, device, compute_type)
    else:
        # Direct model creation (for legacy compatibility)
        logger = get_logger()
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("faster-whisper is not installed. Run: pip install faster-whisper")
        
        logger.info(f"Loading Whisper model '{model_size}' (device: {device}, compute: {compute_type})...")
        start_time = time.time()
        
        try:
            # Try optimal configuration first
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as e:
            logger.warning(f"Falling back to CPU due to: {e}")
            # Fallback to CPU with safe compute type
            model = WhisperModel(model_size, device="cpu", compute_type="float32")
        
        load_time = time.time() - start_time
        logger.success(f"Model loaded in {load_time:.1f}s")
        
        return model


def cleanup_model_cache(max_age_seconds: int = 3600) -> None:
    """Clean up old models from the cache"""
    _global_model_cache.cleanup_old_models(max_age_seconds)


def cleanup_all_models() -> None:
    """Clean up all cached models"""
    _global_model_cache.cleanup_all()