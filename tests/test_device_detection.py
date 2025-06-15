"""
Unit tests for device detection module
"""
import pytest
from unittest.mock import patch, MagicMock
import platform

from transcribe.device_detection import (
    detect_x86_features,
    get_apple_silicon_gpu_info,
    detect_device_capabilities,
    get_optimal_device,
    get_optimal_compute_type,
    get_gpu_memory_info,
    get_optimal_transcription_params
)


class TestX86Features:
    """Test x86 CPU feature detection"""
    
    @patch('transcribe.device_detection.cpuinfo')
    def test_x86_features_with_cpuinfo(self, mock_cpuinfo):
        """Test x86 feature detection with cpuinfo library"""
        mock_cpuinfo.get_cpu_info.return_value = {
            'flags': ['avx2', 'avx', 'sse4_2']
        }
        
        features = detect_x86_features()
        assert 'avx2' in features
        assert 'sse4.2' in features
    
    @patch('transcribe.device_detection.cpuinfo', side_effect=ImportError())
    @patch('platform.system')
    @patch('builtins.open')
    def test_x86_features_linux_fallback(self, mock_open, mock_system):
        """Test x86 feature detection on Linux without cpuinfo"""
        mock_system.return_value = 'Linux'
        mock_open.return_value.__enter__.return_value.read.return_value = 'flags: avx2 avx sse4_2'
        
        features = detect_x86_features()
        assert 'avx2' in features
    
    @patch('transcribe.device_detection.cpuinfo', side_effect=ImportError())
    @patch('platform.system')
    def test_x86_features_no_detection(self, mock_system):
        """Test x86 feature detection with no available methods"""
        mock_system.return_value = 'Unknown'
        
        features = detect_x86_features()
        assert isinstance(features, list)


class TestAppleSiliconGPU:
    """Test Apple Silicon GPU detection"""
    
    @patch('subprocess.run')
    def test_apple_gpu_info_success(self, mock_run):
        """Test successful Apple GPU info detection"""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Total Number of Cores: 32\nOther info..."
        )
        
        gpu_info = get_apple_silicon_gpu_info()
        assert gpu_info['gpu_cores'] == 32
        assert isinstance(gpu_info, dict)
    
    @patch('subprocess.run')
    def test_apple_gpu_info_alternative_pattern(self, mock_run):
        """Test Apple GPU info with alternative pattern"""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Apple M1 Max with 32-core GPU"
        )
        
        gpu_info = get_apple_silicon_gpu_info()
        assert gpu_info['gpu_cores'] == 32
    
    @patch('subprocess.run')
    def test_apple_gpu_info_failure(self, mock_run):
        """Test Apple GPU info detection failure"""
        mock_run.side_effect = Exception("Command failed")
        
        gpu_info = get_apple_silicon_gpu_info()
        assert gpu_info['gpu_cores'] == 0
        assert isinstance(gpu_info, dict)


class TestDeviceCapabilities:
    """Test device capability detection"""
    
    @patch('platform.system')
    @patch('platform.machine')
    @patch('multiprocessing.cpu_count')
    def test_basic_capabilities(self, mock_cpu_count, mock_machine, mock_system):
        """Test basic device capability detection"""
        mock_system.return_value = 'Darwin'
        mock_machine.return_value = 'arm64'
        mock_cpu_count.return_value = 10
        
        capabilities = detect_device_capabilities()
        
        assert capabilities['platform'] == 'Darwin'
        assert capabilities['machine'] == 'arm64'
        assert capabilities['cpu_count'] == 10
        assert 'recommended_device' in capabilities
        assert 'recommended_compute_type' in capabilities
        assert isinstance(capabilities['optimization_hints'], list)
    
    @patch('torch.backends.mps.is_available')
    @patch('torch.backends.mps.is_built')
    def test_mps_detection(self, mock_is_built, mock_is_available):
        """Test MPS detection"""
        mock_is_available.return_value = True
        mock_is_built.return_value = True
        
        capabilities = detect_device_capabilities()
        assert capabilities['has_mps'] == True
    
    @patch('torch.cuda.is_available')
    def test_cuda_detection(self, mock_cuda_available):
        """Test CUDA detection"""
        mock_cuda_available.return_value = True
        
        # Mock device properties
        mock_props = MagicMock()
        mock_props.name = "RTX 4090"
        mock_props.total_memory = 24 * 1024 * 1024 * 1024  # 24GB
        mock_props.major = 8
        mock_props.minor = 9
        
        with patch('torch.cuda.get_device_properties', return_value=mock_props):
            capabilities = detect_device_capabilities()
            assert capabilities['has_cuda'] == True
            assert capabilities['gpu_name'] == "RTX 4090"


class TestOptimalSettings:
    """Test optimal device and compute type selection"""
    
    @patch('transcribe.device_detection.detect_device_capabilities')
    def test_get_optimal_device_auto(self, mock_capabilities):
        """Test automatic device selection"""
        mock_capabilities.return_value = {
            'recommended_device': 'mps',
            'has_mps': True,
            'has_cuda': False
        }
        
        device = get_optimal_device("auto")
        assert device == 'mps'
    
    def test_get_optimal_device_override(self):
        """Test device override"""
        device = get_optimal_device("cpu")
        assert device == "cpu"
    
    @patch('transcribe.device_detection.detect_device_capabilities')
    def test_get_optimal_device_invalid_override(self, mock_capabilities):
        """Test invalid device override falls back"""
        mock_capabilities.return_value = {
            'recommended_device': 'cpu',
            'has_cuda': False,
            'has_mps': False
        }
        
        # Request CUDA on system without CUDA
        device = get_optimal_device("cuda")
        assert device == 'cpu'  # Should fall back
    
    @patch('transcribe.device_detection.detect_device_capabilities')
    def test_get_optimal_compute_type(self, mock_capabilities):
        """Test compute type selection"""
        mock_capabilities.return_value = {
            'recommended_compute_type': 'float16'
        }
        
        compute_type = get_optimal_compute_type("auto")
        assert compute_type == 'float16'
        
        # Test override
        compute_type = get_optimal_compute_type("float32")
        assert compute_type == "float32"


class TestGPUMemoryInfo:
    """Test GPU memory information"""
    
    @patch('torch.backends.mps.is_available')
    def test_mps_memory_info(self, mock_mps_available):
        """Test MPS memory info"""
        mock_mps_available.return_value = True
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.total = 32 * 1024 * 1024 * 1024  # 32GB
            
            memory_info = get_gpu_memory_info()
            assert memory_info['has_unified_memory'] == True
            assert memory_info['total_mb'] > 0
            assert memory_info['available_mb'] > 0
    
    @patch('torch.cuda.is_available')
    def test_cuda_memory_info(self, mock_cuda_available):
        """Test CUDA memory info"""
        mock_cuda_available.return_value = True
        
        mock_props = MagicMock()
        mock_props.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
        mock_props.major = 8
        mock_props.minor = 6
        mock_props.multi_processor_count = 68
        
        with patch('torch.cuda.get_device_properties', return_value=mock_props):
            with patch('torch.cuda.current_device', return_value=0):
                memory_info = get_gpu_memory_info()
                assert memory_info['total_mb'] == 8192
                assert memory_info['gpu_cores'] == 68
    
    def test_no_gpu_memory_info(self):
        """Test memory info with no GPU"""
        # This should not crash and return default values
        memory_info = get_gpu_memory_info()
        assert isinstance(memory_info, dict)
        assert 'total_mb' in memory_info
        assert 'available_mb' in memory_info


class TestTranscriptionParams:
    """Test transcription parameter optimization"""
    
    def test_mps_optimization(self):
        """Test parameter optimization for MPS"""
        capabilities = {
            'has_mps': True,
            'gpu_cores': 32,
            'performance_cores': 8,
            'system_memory_gb': 32
        }
        
        params = get_optimal_transcription_params(capabilities, "base", 1800)
        
        assert params['beam_size'] >= 5
        assert params['vad_filter'] == True
        assert params['temperature'] == 0.0
        assert 'vad_parameters' in params
    
    def test_cuda_optimization(self):
        """Test parameter optimization for CUDA"""
        capabilities = {
            'has_cuda': True,
            'has_mps': False,
            'gpu_memory_mb': 8192,
            'optimization_hints': ['Tensor Core support available']
        }
        
        params = get_optimal_transcription_params(capabilities, "medium", 600)
        
        assert params['beam_size'] >= 5
        assert params['temperature'] == 0.0
    
    def test_cpu_optimization(self):
        """Test parameter optimization for CPU"""
        capabilities = {
            'has_mps': False,
            'has_cuda': False,
            'cpu_features': ['avx2'],
            'system_memory_gb': 16
        }
        
        params = get_optimal_transcription_params(capabilities, "small", 300)
        
        assert params['beam_size'] >= 3
        assert params['word_timestamps'] == False  # Disabled for CPU
    
    def test_low_memory_optimization(self):
        """Test parameter optimization for low memory"""
        capabilities = {
            'has_mps': False,
            'has_cuda': False,
            'system_memory_gb': 4  # Low memory
        }
        
        params = get_optimal_transcription_params(capabilities, "large", 0)
        
        # Should reduce beam size for low memory
        assert params['beam_size'] >= 2
        assert 'best_of' not in params  # Should be removed for memory savings
    
    def test_high_memory_optimization(self):
        """Test parameter optimization for high memory"""
        capabilities = {
            'has_mps': True,
            'gpu_cores': 64,  # Ultra-level
            'system_memory_gb': 64  # High memory
        }
        
        params = get_optimal_transcription_params(capabilities, "tiny", 0)
        
        # Should use more aggressive settings
        assert params['beam_size'] >= 10
        assert 'best_of' in params