# Speed Optimization Implementation Plan

## Context
- **Hardware**: Apple M4 GPU (10 cores) - excellent for Metal Performance Shaders (MPS)
- **Use Case**: Long podcasts/lectures (>30 minutes)
- **Current Performance**: Using CPU-based processing with parallel chunking
- **Goal**: Maximize GPU utilization for 2-4x speed improvement

## Phase 1: GPU Acceleration (IN PROGRESS)
**Target**: 2-4x speed improvement through Metal/MPS optimization

### 1.1 Enhanced Device Detection ✅ PLANNED
- [ ] Improve device selection logic to explicitly prioritize Metal/MPS
- [ ] Add device capability detection and reporting  
- [ ] Implement fallback hierarchy: Metal → CUDA → CPU
- [ ] Add GPU memory detection and reporting

### 1.2 Metal Performance Shaders (MPS) Optimization ✅ PLANNED
- [ ] Add PyTorch dependency for MPS support detection
- [ ] Explicitly configure faster-whisper to use Metal backend
- [ ] Optimize compute_type for M4 GPU (float16 vs float32)
- [ ] Test and benchmark different configurations

### 1.3 Model Loading Strategy ✅ PLANNED
- [ ] Pre-load models in GPU memory for parallel processing
- [ ] Implement model pooling for worker threads
- [ ] Add GPU memory management for multiple model instances
- [ ] Optimize model initialization for Metal backend

### 1.4 Long File Optimization ✅ PLANNED
- [ ] GPU-aware chunk size optimization for M4
- [ ] Minimize model loading overhead in parallel processing
- [ ] Optimize temporary file handling for GPU workflows

## Phase 2: Parallel Processing Enhancements (PLANNED)
**Status**: Not started - will begin after Phase 1 completion

## Phase 3: Model and Configuration Optimizations (NEXT TARGET)
**Status**: Will implement after Phase 1
- Model selection intelligence for M4
- Processing parameter tuning for long files
- Memory management optimization

## Phase 4: Advanced Features (FUTURE)
**Status**: Future consideration

## Implementation Notes
- Focus on long podcast/lecture use case (>30 minutes)
- Maintain backward compatibility
- Test with existing long audio file: `assets/nih_huberman_long.mp3` (266 minutes)
- Measure performance improvements at each step

## Performance Tracking
**Baseline**: Current performance on 266-minute file
**Target**: 2-4x improvement in Phase 1

## Resumption Instructions
1. Check `.tasks/phase1_progress.md` for current step status
2. Run existing tests to ensure no regressions: `python3 -m pytest tests/ -v`
3. Benchmark with test file before/after changes
4. Continue with next unchecked item in current phase