# Architecture Overview

## Design Philosophy

The Offline Audio Transcriber follows a single-file architecture designed for simplicity, offline operation, and performance. The implementation prioritizes straightforward deployment and minimal dependencies while maximizing transcription speed.

## Core Components

### 1. CLI Interface (`parse_arguments()`)
- **Purpose**: Handle command-line argument parsing and validation
- **Design Decision**: Uses Python's `argparse` for robust option handling
- **Key Features**:
  - Required input/output file arguments
  - Optional model selection with sensible defaults
  - Built-in help and error messages

### 2. Input Validation (`validate_input_file()`, `validate_output_path()`)
- **Purpose**: Ensure file system integrity before processing
- **Design Decisions**:
  - Fail fast with clear error messages
  - Automatic output directory creation
  - Comprehensive format validation
- **Supported Formats**: `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.aac`, `.wma`

### 3. Model Management (`initialize_whisper_model()`)
- **Purpose**: Handle Whisper model loading and device selection
- **Design Decisions**:
  - Automatic device selection (Metal/CUDA/CPU)
  - Lazy loading with first-use model download
  - Performance timing for user feedback
- **Caching**: Models stored in `~/.cache/huggingface/hub/`

### 4. Transcription Engine (`transcribe_audio()`)
- **Purpose**: Core transcription processing with progress feedback
- **Key Optimizations**:
  - Streaming segments for memory efficiency
  - Voice Activity Detection (VAD) to skip silence
  - Progress reporting every 10 segments
- **Performance Metrics**: Real-time speed ratio calculation

## Data Flow

```
Audio File → Validation → Model Loading → Streaming Transcription → Text Output
     ↓              ↓            ↓               ↓                    ↓
File Check → Format Check → Device Select → VAD Processing → File Write
```

## Memory Management Strategy

### Streaming Segments
- **Problem**: Large audio files can consume excessive memory
- **Solution**: Process audio in streaming segments rather than loading entire file
- **Implementation**: `model.transcribe()` returns generator for memory-efficient iteration

### Model Caching
- **Strategy**: One-time download with persistent local cache
- **Location**: Standard HuggingFace cache directory
- **Benefit**: No repeated downloads for offline operation

## Performance Optimizations

### 1. Device Selection
- **Auto-detection**: Automatically selects best available device
- **Priority**: Metal (macOS) > CUDA (GPU) > CPU
- **Fallback**: Graceful degradation to CPU if GPU unavailable

### 2. Voice Activity Detection
- **Purpose**: Skip silent portions to improve processing speed
- **Configuration**: 1-second minimum silence duration
- **Benefit**: Significant speed improvement for audio with pauses

### 3. Beam Search
- **Setting**: beam_size=5 for balanced speed/accuracy
- **Trade-off**: Higher beam size = better accuracy but slower processing

## Error Handling Architecture

### Exception Hierarchy
1. **File System Errors**: `FileNotFoundError`, `PermissionError`
2. **Validation Errors**: `ValueError` for format/path issues
3. **Dependency Errors**: `ImportError` for missing libraries
4. **User Interruption**: `KeyboardInterrupt` for graceful shutdown

### Error Reporting
- **Philosophy**: Clear, actionable error messages
- **Implementation**: Specific error types with context
- **User Experience**: Suggestions for resolution when possible

## Testing Strategy

### Unit Test Coverage
- **CLI Parsing**: Argument validation and edge cases
- **File Validation**: Format checking and permission handling
- **Model Initialization**: Mocking external dependencies
- **Transcription Logic**: Mock-based testing of core functionality

### Mock Strategy
- **External Dependencies**: Mock faster-whisper to avoid installation requirements
- **File System**: Use temporary directories for isolated testing
- **Performance**: Mock timing functions for consistent test results

## Security Considerations

### File System Access
- **Input Validation**: Strict path validation to prevent directory traversal
- **Output Safety**: Directory creation with proper error handling
- **Permissions**: Explicit permission checks before file operations

### Model Security
- **Source**: Models downloaded from official HuggingFace repositories
- **Verification**: Automatic checksum validation by faster-whisper
- **Offline Operation**: No network access required post-installation

## Deployment Considerations

### Single-File Design
- **Benefit**: Easy deployment and distribution
- **Trade-off**: All functionality in one module
- **Maintenance**: Clear function separation for readability

### Dependency Management
- **Minimal Dependencies**: Only faster-whisper required
- **Version Pinning**: Specify minimum version for compatibility
- **Optional Dependencies**: No optional features to maintain simplicity

## Parallel Processing Architecture

### 5. Parallel Chunking System (`transcribe_audio_parallel()`)
- **Purpose**: Handle large audio files (>30 minutes) with parallel processing
- **Key Components**:
  - Audio duration detection via ffprobe
  - Automatic chunking using ffmpeg
  - Thread pool for concurrent chunk processing
  - Chronological transcript reassembly

### 6. Audio Chunking (`create_audio_chunks()`)
- **Purpose**: Split large audio files into manageable segments
- **Design Decisions**:
  - Use ffmpeg for reliable audio splitting
  - Convert to WAV format for whisper compatibility
  - Standard 16kHz sample rate
  - Configurable chunk duration (default: 10 minutes)

### 7. Worker Thread Processing (`transcribe_chunk()`)
- **Purpose**: Process individual audio chunks in parallel
- **Implementation**:
  - Separate model instances per thread
  - Timestamp adjustment to maintain chronological order
  - Error isolation per chunk
  - Progress reporting per completed chunk

### 8. Transcript Assembly (`assemble_transcripts()`)
- **Purpose**: Combine parallel results into final transcript
- **Features**:
  - Chronological ordering by timestamp
  - Error aggregation and reporting
  - Language detection from first successful chunk
  - Segment continuity verification

## Enhanced Data Flow

### Parallel Processing Path
```
Large Audio (>30min) → Duration Check → Chunking → Parallel Processing → Assembly → Output
       ↓                    ↓              ↓              ↓              ↓         ↓
   ffprobe check → 30min threshold → ffmpeg split → Thread Pool → Timestamp sort → File Write
```

### Sequential Processing Path (Existing)
```
Small Audio (<30min) → Model Loading → Streaming Transcription → Output
       ↓                    ↓                    ↓               ↓
   Direct process → Single model → VAD segments → File Write
```

## Future Extensibility

### Modular Functions
Each major component is implemented as a separate function, allowing for:
- Individual testing
- Future refactoring
- Feature enhancement without architectural changes

### Configuration Points
- Model selection via command line
- Device selection (automatic but could be manual)
- Output format (currently text, could support JSON/SRT)
- VAD parameters (currently fixed, could be configurable)
- **Parallel Processing**:
  - Thread count (auto-detect or manual)
  - Chunk duration (configurable)
  - Parallel processing threshold (default: 30 minutes)
  - Enable/disable parallel processing

## Performance Targets

### Speed Requirements
- **Target**: 4x faster than real-time processing
- **Measurement**: Audio duration / processing time ratio
- **Reporting**: Displayed to user for transparency

### Memory Efficiency
- **Strategy**: Streaming processing prevents memory bloat
- **Target**: Constant memory usage regardless of audio file size
- **Implementation**: Generator-based segment processing

## Conclusion

This architecture prioritizes simplicity and performance while maintaining extensibility. The single-file design reduces deployment complexity while the modular function structure enables future enhancements. The streaming approach ensures scalability to large files, and comprehensive error handling provides a robust user experience.