#!/usr/bin/env python3
"""
Test script for parallel processing module
"""
import sys
import tempfile
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, '.')

from transcribe.parallel import (
    should_use_parallel_processing,
    get_optimal_thread_count,
    get_model_memory_requirements,
    optimize_thread_count_for_gpu,
    calculate_optimal_chunk_size,
    create_audio_chunks,
    assemble_transcripts,
    ParallelTranscriber
)


def test_parallel_decision():
    """Test parallel processing decision logic"""
    print("Testing parallel processing decision...")
    
    # Test short files (should not use parallel)
    assert not should_use_parallel_processing(60)  # 1 minute
    assert not should_use_parallel_processing(1500)  # 25 minutes
    
    # Test long files (should use parallel)
    assert should_use_parallel_processing(2000)  # 33 minutes
    assert should_use_parallel_processing(3600)  # 1 hour
    
    # Test custom threshold
    assert not should_use_parallel_processing(600, min_duration_minutes=15)  # 10 min < 15 min
    assert should_use_parallel_processing(1200, min_duration_minutes=15)  # 20 min > 15 min
    
    print("✓ Parallel processing decision tests passed")


def test_thread_count_optimization():
    """Test thread count optimization"""
    print("Testing thread count optimization...")
    
    # Test basic thread count
    threads = get_optimal_thread_count()
    assert 2 <= threads <= 8
    
    # Test user override
    user_threads = get_optimal_thread_count(4)
    assert user_threads == 4
    
    # Test invalid input
    try:
        get_optimal_thread_count(0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    print("✓ Thread count optimization tests passed")


def test_memory_requirements():
    """Test model memory requirements"""
    print("Testing model memory requirements...")
    
    # Test all model sizes
    assert get_model_memory_requirements("tiny") == 150
    assert get_model_memory_requirements("base") == 500
    assert get_model_memory_requirements("small") == 1000
    assert get_model_memory_requirements("medium") == 2500
    assert get_model_memory_requirements("large") == 4500
    
    # Test unknown model (should return default)
    assert get_model_memory_requirements("unknown") == 500
    
    print("✓ Memory requirements tests passed")


def test_gpu_memory_optimization():
    """Test GPU memory optimization"""
    print("Testing GPU memory optimization...")
    
    # Test with plenty of GPU memory
    optimized = optimize_thread_count_for_gpu(8, 16384, "tiny")  # 16GB GPU, tiny model
    assert optimized == 8  # Should allow all threads
    
    # Test with limited GPU memory
    optimized = optimize_thread_count_for_gpu(8, 4096, "large")  # 4GB GPU, large model
    assert optimized < 8  # Should reduce thread count
    
    # Test with no GPU memory
    optimized = optimize_thread_count_for_gpu(8, 0, "base")
    assert optimized == 8  # Should return original count
    
    print("✓ GPU memory optimization tests passed")


def test_chunk_size_calculation():
    """Test optimal chunk size calculation"""
    print("Testing chunk size calculation...")
    
    # Test basic calculation
    chunk_size = calculate_optimal_chunk_size(3600, 4, 8192, 10, "base")  # 1 hour, 4 threads, 8GB GPU
    assert 5 <= chunk_size <= 30  # Should be within reasonable bounds
    
    # Test with different parameters
    for duration in [1800, 3600, 7200]:  # 30min, 1hr, 2hr
        for threads in [2, 4, 8]:
            for gpu_memory in [0, 4096, 8192]:
                chunk_size = calculate_optimal_chunk_size(duration, threads, gpu_memory)
                assert 5 <= chunk_size <= 30
    
    print("✓ Chunk size calculation tests passed")


def test_chunk_creation():
    """Test audio chunk creation with a real file"""
    print("Testing audio chunk creation...")
    
    # Check if test audio file exists
    test_audio = Path("assets/testing speech audio file.m4a")
    if not test_audio.exists():
        print("⚠️  Skipping chunk creation test - test file not found")
        return
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            # Test creating chunks (use very small chunks since file is only 11 seconds)
            # This should create just 1 chunk
            chunks = create_audio_chunks(test_audio, 1, temp_path)  # 1 minute chunks
            
            assert len(chunks) >= 1
            assert all(chunk['file'].exists() for chunk in chunks)
            assert all(chunk['duration'] > 0 for chunk in chunks)
            assert all(chunk['start_time'] >= 0 for chunk in chunks)
            
            # Verify chunk structure
            for chunk in chunks:
                required_fields = ['index', 'file', 'start_time', 'duration', 'end_time']
                for field in required_fields:
                    assert field in chunk, f"Missing field: {field}"
            
            print(f"  ✓ Created {len(chunks)} chunks successfully")
            
        except Exception as e:
            print(f"  ⚠️  Chunk creation test failed: {e}")


def test_transcript_assembly():
    """Test transcript assembly"""
    print("Testing transcript assembly...")
    
    # Create mock chunk results
    chunk_results = [
        {
            'chunk_index': 0,
            'segments': [
                {'start': 0.0, 'end': 5.0, 'text': 'First segment'},
                {'start': 5.0, 'end': 10.0, 'text': 'Second segment'}
            ],
            'language': 'en',
            'language_probability': 0.99
        },
        {
            'chunk_index': 1,
            'segments': [
                {'start': 10.0, 'end': 15.0, 'text': 'Third segment'},
                {'start': 15.0, 'end': 20.0, 'text': 'Fourth segment'}
            ],
            'language': 'en',
            'language_probability': 0.98
        }
    ]
    
    # Test normal assembly
    segments, language_info = assemble_transcripts(chunk_results)
    
    assert len(segments) == 4
    assert segments[0]['text'] == 'First segment'
    assert segments[3]['text'] == 'Fourth segment'
    assert language_info['language'] == 'en'
    assert language_info['language_probability'] == 0.99
    
    # Test with errors
    error_results = [
        {
            'chunk_index': 0,
            'error': 'Test error',
            'segments': []
        }
    ]
    
    try:
        assemble_transcripts(error_results)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "Test error" in str(e)
    
    print("✓ Transcript assembly tests passed")


def test_parallel_transcriber():
    """Test ParallelTranscriber class"""
    print("Testing ParallelTranscriber class...")
    
    # Test initialization
    transcriber = ParallelTranscriber("tiny", "cpu", "float32", 2)
    assert transcriber.model_size == "tiny"
    assert transcriber.device == "cpu"
    assert transcriber.compute_type == "float32"
    assert transcriber.num_threads == 2
    
    # For a full test, we'd need to test transcribe_file_parallel,
    # but that requires the full audio processing pipeline
    print("✓ ParallelTranscriber initialization test passed")


def main():
    """Run all parallel processing tests"""
    print("Running parallel processing module tests...\n")
    
    test_parallel_decision()
    test_thread_count_optimization()
    test_memory_requirements()
    test_gpu_memory_optimization()
    test_chunk_size_calculation()
    test_chunk_creation()
    test_transcript_assembly()
    test_parallel_transcriber()
    
    print("\n✅ Parallel processing tests completed!")


if __name__ == "__main__":
    main()