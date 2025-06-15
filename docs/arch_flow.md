# Architecture Flow Diagram

This document describes the code flow from CLI input to transcription output.

## Main Flow

```mermaid
graph TD
    A[CLI Input] --> B[parse_arguments]
    B --> C{Special Commands?}
    
    C -->|--cache-stats| D[Show Cache Stats]
    C -->|--show-config| E[Show Saved Config]
    C -->|--reset-config| F[Reset Config]
    C -->|--preload| G[Preload Model]
    
    C -->|Transcription| H[Load User Config]
    H --> I{Config Exists?}
    
    I -->|Yes| J[Use Saved Config]
    I -->|No| K[Attempt Auto Detection]
    
    K --> L{Detection Success?}
    L -->|Yes| M[Use Detected Config]
    L -->|No| N[Prompt User for Config]
    N --> O[Save User Config]
    
    J --> P[Validate Input File]
    M --> P
    O --> P
    
    P --> Q[Validate Output Path]
    Q --> R[Get Audio Duration]
    R --> S{Duration > 30 min?}
    
    S -->|Yes| T[Parallel Processing]
    S -->|No| U[Sequential Processing]
    
    T --> V[Create Audio Chunks]
    V --> W[Initialize Model Pool]
    W --> X[Process Chunks in Parallel]
    X --> Y[Assemble Results]
    
    U --> Z[Initialize Single Model]
    Z --> AA[Process Full Audio]
    
    Y --> BB[Write Output File]
    AA --> BB
    BB --> CC[Show Results]
    
    D --> DD[Exit]
    E --> DD
    F --> DD
    G --> DD
    CC --> DD
```

## Key Components

### Configuration System

```mermaid
graph LR
    A[System Detection] --> B{Detection Success?}
    B -->|No| C[Prompt User]
    C --> D[Save to ~/.transcribe/config.json]
    B -->|Yes| E[Use Detected Values]
    D --> F[Load on Next Run]
    E --> G[Continue Processing]
    F --> G
```

### Model Management

```mermaid
graph TD
    A[Model Request] --> B[Global Model Cache]
    B --> C{Model Cached?}
    C -->|Yes| D[Return Cached Model]
    C -->|No| E[Download/Load Model]
    E --> F[Cache Model]
    F --> G[Return Model]
    D --> H[Update Access Count]
    G --> H
```

### Parallel Processing Flow

```mermaid
graph TD
    A[Long Audio File] --> B[Calculate Optimal Chunk Size]
    B --> C[Create Audio Chunks with ffmpeg]
    C --> D[Initialize Model Pool]
    D --> E[ThreadPoolExecutor]
    E --> F[Process Chunks Concurrently]
    F --> G[Collect Results]
    G --> H[Sort by Timestamp]
    H --> I[Merge Transcripts]
    I --> J[Write Final Output]
```

### Error Handling & Timeout Prevention

```mermaid
graph TD
    A[System Call] --> B[Add Timeout]
    B --> C{Timeout Reached?}
    C -->|Yes| D[Use Conservative Defaults]
    C -->|No| E[Use Detected Values]
    D --> F[Alert User]
    F --> G[Prompt for Manual Config]
    E --> H[Continue Normal Flow]
    G --> I[Save Config for Future]
    I --> H
```

## Key Design Decisions

1. **Single File Architecture**: All functionality in `transcribe.py` for simplicity
2. **Configuration Persistence**: Save user input to avoid repeated timeouts
3. **Conservative Defaults**: Fail safely with CPU processing when detection fails
4. **Model Caching**: Global cache to avoid repeated model loading
5. **Timeout Protection**: All system calls have timeouts to prevent hangs
6. **Graceful Degradation**: Fall back to simpler processing modes when needed

## File Processing Paths

### Small Files (< 30 minutes)
```
Input → Validation → Single Model → Transcribe → Output
```

### Large Files (> 30 minutes)  
```
Input → Validation → Chunking → Model Pool → Parallel Transcribe → Assembly → Output
```

### Error Recovery
```
Detection Failure → User Prompt → Config Save → Retry Processing
```

This architecture ensures reliable operation even when system detection fails, while providing optimal performance for different file sizes and hardware configurations.