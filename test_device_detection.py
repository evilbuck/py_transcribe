#!/usr/bin/env python3
"""
Test script for device detection module
"""
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, '.')

from transcribe.device_detection import (
    detect_device_capabilities, 
    get_optimal_device, 
    get_optimal_compute_type,
    get_gpu_memory_info,
    get_optimal_transcription_params,
    report_device_capabilities
)


def test_device_detection():
    """Test device detection functionality"""
    print("Testing device detection...")
    
    # Test basic device detection
    capabilities = detect_device_capabilities()
    
    # Verify required fields exist
    required_fields = [
        'platform', 'machine', 'cpu_count', 'has_mps', 'has_cuda', 
        'recommended_device', 'recommended_compute_type', 'optimization_hints'
    ]
    
    for field in required_fields:
        assert field in capabilities, f"Missing field: {field}"
    
    # Verify basic values make sense
    assert capabilities['cpu_count'] > 0
    assert capabilities['platform'] in ['Darwin', 'Linux', 'Windows']
    assert capabilities['recommended_device'] in ['auto', 'cpu', 'cuda', 'mps']
    assert capabilities['recommended_compute_type'] in ['auto', 'float32', 'float16', 'int8', 'int8_float16']
    assert isinstance(capabilities['optimization_hints'], list)
    
    print("✓ Device detection tests passed")
    return capabilities


def test_optimal_settings():
    """Test optimal device and compute type selection"""
    print("Testing optimal settings selection...")
    
    # Test default settings
    device = get_optimal_device()
    compute_type = get_optimal_compute_type()
    
    assert device in ['auto', 'cpu', 'cuda', 'mps']
    assert compute_type in ['auto', 'float32', 'float16', 'int8', 'int8_float16']
    
    # Test override settings
    device_override = get_optimal_device('cpu')
    compute_override = get_optimal_compute_type('float32')
    
    assert device_override == 'cpu'
    assert compute_override == 'float32'
    
    print("✓ Optimal settings tests passed")


def test_gpu_memory_info():
    """Test GPU memory information retrieval"""
    print("Testing GPU memory info...")
    
    memory_info = get_gpu_memory_info()
    
    # Verify required fields
    required_fields = ['total_mb', 'available_mb', 'has_unified_memory', 'gpu_cores']
    for field in required_fields:
        assert field in memory_info, f"Missing field: {field}"
    
    # Verify values are reasonable
    assert memory_info['total_mb'] >= 0
    assert memory_info['available_mb'] >= 0
    assert isinstance(memory_info['has_unified_memory'], bool)
    assert memory_info['gpu_cores'] >= 0
    
    print("✓ GPU memory info tests passed")


def test_transcription_params():
    """Test transcription parameter optimization"""
    print("Testing transcription parameter optimization...")
    
    # Get capabilities
    capabilities = detect_device_capabilities()
    
    # Test parameter generation
    params = get_optimal_transcription_params(capabilities, model_size="tiny")
    
    # Verify required parameters exist
    required_params = ['beam_size', 'vad_filter', 'vad_parameters', 'temperature']
    for param in required_params:
        assert param in params, f"Missing parameter: {param}"
    
    # Verify reasonable values
    assert 1 <= params['beam_size'] <= 20
    assert isinstance(params['vad_filter'], bool)
    assert isinstance(params['vad_parameters'], dict)
    assert 0.0 <= params['temperature'] <= 1.0
    
    # Test with different model sizes
    for model_size in ['tiny', 'base', 'small', 'medium', 'large']:
        size_params = get_optimal_transcription_params(capabilities, model_size=model_size)
        assert 'beam_size' in size_params
        
    print("✓ Transcription parameter tests passed")


def test_capabilities_reporting():
    """Test capabilities reporting"""
    print("Testing capabilities reporting...")
    
    capabilities = detect_device_capabilities()
    
    # Test basic reporting (should not crash)
    try:
        report_device_capabilities(capabilities, verbose=False)
        print("  ✓ Basic reporting works")
    except Exception as e:
        print(f"  ❌ Basic reporting failed: {e}")
        
    # Test verbose reporting
    try:
        report_device_capabilities(capabilities, verbose=True)
        print("  ✓ Verbose reporting works")
    except Exception as e:
        print(f"  ❌ Verbose reporting failed: {e}")
    
    print("✓ Capabilities reporting tests passed")


def main():
    """Run all device detection tests"""
    print("Running device detection module tests...\n")
    
    capabilities = test_device_detection()
    test_optimal_settings()
    test_gpu_memory_info()
    test_transcription_params()
    test_capabilities_reporting()
    
    print(f"\n✅ Device detection tests completed!")
    print(f"Detected platform: {capabilities['platform']} {capabilities['machine']}")
    print(f"Recommended device: {capabilities['recommended_device']}")
    print(f"Recommended compute type: {capabilities['recommended_compute_type']}")


if __name__ == "__main__":
    main()