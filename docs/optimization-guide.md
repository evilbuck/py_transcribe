# Optimization Guide

This guide explains the runtime optimization features of the py_transcribe tool and how it automatically adapts to different hardware configurations.

## Automatic Hardware Detection

The tool automatically detects and optimizes for:

- **CPU Architecture**: ARM64 (Apple Silicon), x86_64 (Intel/AMD)
- **GPU Acceleration**: Metal (macOS), CUDA (NVIDIA), ROCm (AMD)
- **CPU Features**: AVX512, AVX2, AVX, SSE4.2, NEON
- **System Memory**: Adjusts parameters based on available RAM
- **GPU Memory**: Optimizes batch sizes and parallelism

## Platform-Specific Optimizations

### Apple Silicon (M1/M2/M3/M4)

- Automatically uses Metal/CoreML acceleration via faster-whisper
- Unified memory architecture allows efficient GPU utilization
- Runtime detection of GPU cores for optimal settings:
  - 64+ cores (Ultra): Maximum optimization (beam size 14)
  - 32+ cores (Max): High performance (beam size 12)
  - 16+ cores (Pro): Enhanced performance (beam size 10)
  - 8-10 cores (Base): Balanced performance (beam size 8-9)
- Performance scales with actual hardware, not model names

### NVIDIA GPUs

- CUDA acceleration with automatic memory scaling
- Tensor Core utilization on supported GPUs (compute capability 7.0+)
- Dynamic batch size based on VRAM:
  - 8GB+: Large batches, beam size 10
  - 4-8GB: Medium batches, beam size 8
  - <4GB: Conservative settings

### CPU-Only Systems

- Optimized for SIMD instructions:
  - AVX512: Enhanced parallelism
  - AVX2: Vectorized operations
  - Fallback for older CPUs
- Thread count optimization (75% of cores)
- Conservative memory usage

## Memory-Based Optimization

### Dynamic Chunk Size

For parallel processing of long audio files:
- Calculates optimal chunk size based on:
  - Available GPU/system memory
  - Model size requirements
  - Number of parallel threads
- Automatically adjusts for memory constraints

### Batch Size Optimization

- Estimates memory per model instance
- Reserves memory for system stability
- Scales batch size based on available resources

## Transcription Parameters

The tool automatically adjusts these parameters:

- **Beam Size**: Search breadth (2-12, based on hardware)
- **VAD Settings**: Voice activity detection sensitivity
- **Temperature**: Sampling temperature (0.0 for consistency)
- **Word Timestamps**: Enabled for long files on capable hardware

## Manual Control

Override automatic detection when needed:

```bash
# Force CPU processing
./transcribe audio.mp3 -o text.txt --device cpu

# Force specific compute type
./transcribe audio.mp3 -o text.txt --compute-type float16

# Show optimization details
./transcribe audio.mp3 -o text.txt --verbose
```

## Performance Tips

1. **Model Selection**: Start with 'base' model for good speed/accuracy balance
2. **Parallel Processing**: Automatically enabled for files >30 minutes
3. **Memory**: Close other applications for maximum performance
4. **GPU Drivers**: Keep GPU drivers updated for best performance

## Benchmarking

Use `--benchmark` flag to measure performance:

```bash
./transcribe audio.mp3 -o text.txt --benchmark
```

This provides detailed metrics including:
- Processing speed ratio (vs real-time)
- Memory usage statistics
- Device utilization
- Optimization effectiveness