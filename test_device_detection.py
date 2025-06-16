#!/usr/bin/env python3
"""
Unit tests for device detection module
"""
import unittest
import os
from unittest.mock import patch, MagicMock
from device_detection import DeviceDetector, OptimalConfiguration, DeviceCapabilities, setup_environment


class TestDeviceCapabilities(unittest.TestCase):
    """Test DeviceCapabilities dataclass"""
    
    def test_default_initialization(self):
        """Test default initialization of DeviceCapabilities"""
        caps = DeviceCapabilities()
        self.assertFalse(caps.has_cuda)
        self.assertFalse(caps.has_mps)
        self.assertIsNone(caps.cuda_version)
        self.assertIsNone(caps.gpu_memory)
        self.assertEqual(caps.cpu_cores, 1)
        self.assertEqual(caps.optimal_threads, 1)


class TestDeviceDetector(unittest.TestCase):
    """Test DeviceDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = DeviceDetector()
    
    def test_initialization(self):
        """Test DeviceDetector initialization"""
        self.assertIsNone(self.detector._capabilities)
    
    @patch('os.cpu_count')
    def test_detect_capabilities_cpu_cores(self, mock_cpu_count):
        """Test CPU core detection"""
        mock_cpu_count.return_value = 8
        caps = self.detector._detect_capabilities()
        self.assertEqual(caps.cpu_cores, 8)
        self.assertEqual(caps.optimal_threads, 8)  # Should be min(8, 8)
    
    @patch('os.cpu_count')
    def test_detect_capabilities_high_cpu_cores(self, mock_cpu_count):
        """Test CPU core detection with high core count"""
        mock_cpu_count.return_value = 16
        caps = self.detector._detect_capabilities()
        self.assertEqual(caps.cpu_cores, 16)
        self.assertEqual(caps.optimal_threads, 8)  # Should be capped at 8
    
    @patch('device_detection.DeviceDetector._check_cuda')
    @patch('device_detection.DeviceDetector._check_mps')
    def test_detect_capabilities_no_gpu(self, mock_mps, mock_cuda):
        """Test capabilities detection with no GPU"""
        mock_cuda.return_value = False
        mock_mps.return_value = False
        caps = self.detector._detect_capabilities()
        self.assertFalse(caps.has_cuda)
        self.assertFalse(caps.has_mps)
    
    def test_check_cuda_no_torch(self):
        """Test CUDA check when torch is not available"""
        with patch.dict('sys.modules', {'torch': None}):
            result = self.detector._check_cuda()
            # Should fallback to nvidia-smi check or return False
            self.assertIsInstance(result, bool)
    
    @patch('platform.system')
    def test_check_mps_non_darwin(self, mock_system):
        """Test MPS check on non-Darwin system"""
        mock_system.return_value = "Linux"
        result = self.detector._check_mps()
        self.assertFalse(result)
    
    def test_get_capabilities_caching(self):
        """Test that capabilities are cached"""
        caps1 = self.detector.get_capabilities()
        caps2 = self.detector.get_capabilities()
        self.assertIs(caps1, caps2)  # Should be the same object


class TestOptimalConfiguration(unittest.TestCase):
    """Test OptimalConfiguration class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = OptimalConfiguration()
    
    def test_initialization(self):
        """Test OptimalConfiguration initialization"""
        self.assertIsInstance(self.config.detector, DeviceDetector)
    
    def test_get_optimal_device_override(self):
        """Test device selection with override"""
        self.assertEqual(self.config.get_optimal_device("cuda"), "cuda")
        self.assertEqual(self.config.get_optimal_device("cpu"), "cpu")
        self.assertEqual(self.config.get_optimal_device("mps"), "mps")
    
    @patch('device_detection.DeviceDetector.get_capabilities')
    def test_get_optimal_device_cuda_preferred(self, mock_caps):
        """Test device selection prefers CUDA when available"""
        mock_caps.return_value = DeviceCapabilities(has_cuda=True, has_mps=True)
        device = self.config.get_optimal_device()
        self.assertEqual(device, "cuda")
    
    @patch('device_detection.DeviceDetector.get_capabilities')
    def test_get_optimal_device_mps_fallback(self, mock_caps):
        """Test device selection falls back to MPS"""
        mock_caps.return_value = DeviceCapabilities(has_cuda=False, has_mps=True)
        device = self.config.get_optimal_device()
        self.assertEqual(device, "mps")
    
    @patch('device_detection.DeviceDetector.get_capabilities')
    def test_get_optimal_device_cpu_fallback(self, mock_caps):
        """Test device selection falls back to CPU"""
        mock_caps.return_value = DeviceCapabilities(has_cuda=False, has_mps=False)
        device = self.config.get_optimal_device()
        self.assertEqual(device, "cpu")
    
    def test_get_optimal_compute_type_override(self):
        """Test compute type selection with override"""
        self.assertEqual(self.config.get_optimal_compute_type("cuda", "float32"), "float32")
        self.assertEqual(self.config.get_optimal_compute_type("cpu", "int8"), "int8")
    
    @patch('device_detection.DeviceDetector.get_capabilities')
    def test_get_optimal_compute_type_cuda_high_memory(self, mock_caps):
        """Test compute type for CUDA with high memory"""
        mock_caps.return_value = DeviceCapabilities(gpu_memory=8000)
        compute_type = self.config.get_optimal_compute_type("cuda")
        self.assertEqual(compute_type, "float16")
    
    @patch('device_detection.DeviceDetector.get_capabilities')
    def test_get_optimal_compute_type_cuda_low_memory(self, mock_caps):
        """Test compute type for CUDA with low memory"""
        mock_caps.return_value = DeviceCapabilities(gpu_memory=2000)
        compute_type = self.config.get_optimal_compute_type("cuda")
        self.assertEqual(compute_type, "int8_float16")
    
    def test_get_optimal_compute_type_mps(self):
        """Test compute type for Apple MPS"""
        compute_type = self.config.get_optimal_compute_type("mps")
        self.assertEqual(compute_type, "float16")
    
    def test_get_optimal_compute_type_cpu(self):
        """Test compute type for CPU"""
        compute_type = self.config.get_optimal_compute_type("cpu")
        self.assertEqual(compute_type, "int8")
    
    @patch('device_detection.DeviceDetector.get_capabilities')
    def test_get_optimal_threads(self, mock_caps):
        """Test optimal thread count"""
        mock_caps.return_value = DeviceCapabilities(optimal_threads=6)
        threads = self.config.get_optimal_threads()
        self.assertEqual(threads, 6)
    
    def test_get_optimal_threads_override(self):
        """Test optimal thread count with override"""
        threads = self.config.get_optimal_threads(4)
        self.assertEqual(threads, 4)
    
    @patch('device_detection.DeviceDetector.get_capabilities')
    def test_get_optimal_batch_size_cuda(self, mock_caps):
        """Test batch size for CUDA"""
        mock_caps.return_value = DeviceCapabilities(gpu_memory=8000)
        batch_size = self.config.get_optimal_batch_size("cuda", "base")
        self.assertGreater(batch_size, 0)
        self.assertLessEqual(batch_size, 32)
    
    def test_get_optimal_batch_size_mps(self):
        """Test batch size for MPS"""
        batch_size = self.config.get_optimal_batch_size("mps", "base")
        self.assertEqual(batch_size, 8)
    
    def test_get_optimal_batch_size_cpu(self):
        """Test batch size for CPU"""
        batch_size = self.config.get_optimal_batch_size("cpu", "base")
        self.assertEqual(batch_size, 2)
    
    @patch('device_detection.DeviceDetector.get_capabilities')
    def test_get_configuration_summary(self, mock_caps):
        """Test configuration summary"""
        mock_caps.return_value = DeviceCapabilities(
            has_cuda=True,
            has_mps=False,
            cuda_version="11.8",
            gpu_memory=8000,
            cpu_cores=8
        )
        
        summary = self.config.get_configuration_summary("cuda", "float16", 4, 16)
        
        self.assertEqual(summary["device"], "cuda")
        self.assertEqual(summary["compute_type"], "float16")
        self.assertEqual(summary["threads"], 4)
        self.assertEqual(summary["batch_size"], 16)
        self.assertTrue(summary["capabilities"]["has_cuda"])
        self.assertFalse(summary["capabilities"]["has_mps"])


class TestSetupEnvironment(unittest.TestCase):
    """Test setup_environment function"""
    
    def test_setup_environment(self):
        """Test environment variable setup"""
        original_omp = os.environ.get("OMP_NUM_THREADS")
        original_mkl = os.environ.get("MKL_NUM_THREADS")
        original_numexpr = os.environ.get("NUMEXPR_NUM_THREADS")
        
        try:
            setup_environment(4)
            
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "4")
            self.assertEqual(os.environ["MKL_NUM_THREADS"], "4")
            self.assertEqual(os.environ["NUMEXPR_NUM_THREADS"], "4")
        
        finally:
            # Restore original environment
            if original_omp is not None:
                os.environ["OMP_NUM_THREADS"] = original_omp
            elif "OMP_NUM_THREADS" in os.environ:
                del os.environ["OMP_NUM_THREADS"]
            
            if original_mkl is not None:
                os.environ["MKL_NUM_THREADS"] = original_mkl
            elif "MKL_NUM_THREADS" in os.environ:
                del os.environ["MKL_NUM_THREADS"]
            
            if original_numexpr is not None:
                os.environ["NUMEXPR_NUM_THREADS"] = original_numexpr
            elif "NUMEXPR_NUM_THREADS" in os.environ:
                del os.environ["NUMEXPR_NUM_THREADS"]


if __name__ == "__main__":
    unittest.main()