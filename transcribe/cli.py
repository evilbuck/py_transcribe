"""
Command-line interface for the audio transcriber using Typer
"""
import sys
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from .transcriber import AudioTranscriber
from .utils import validate_input_file, validate_output_path, get_audio_duration, format_time
from .logger import get_logger

app = typer.Typer(
    name="transcribe",
    help="Transcribe audio files to text using faster-whisper (offline)",
    no_args_is_help=True,
)


@app.command()
def transcribe(
    input_file: Annotated[Path, typer.Argument(help="Path to the audio file to transcribe")],
    output: Annotated[Path, typer.Option("-o", "--output", help="Path for the output text file")],
    model: Annotated[str, typer.Option(help="Whisper model size to use")] = "base",
    device: Annotated[str, typer.Option(help="Device for inference (auto/cpu/cuda/mps)")] = "auto",
    compute_type: Annotated[str, typer.Option("--compute-type", help="Compute type (auto/float32/float16/int8/int8_float16)")] = "auto",
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed information")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Enable debug logging")] = False,
    beam_size: Annotated[int, typer.Option(help="Beam size for transcription")] = 5,
    temperature: Annotated[float, typer.Option(help="Temperature for transcription")] = 0.0,
):
    """Transcribe an audio file to text"""
    
    # Initialize logger
    logger = get_logger(debug=debug, verbose=verbose)
    
    try:
        # Validate input file
        logger.debug(f"Validating input file: {input_file}")
        input_path = validate_input_file(input_file)
        logger.success(f"Input file validated: {input_path}")
        
        # Validate output path
        logger.debug(f"Validating output path: {output}")
        output_path = validate_output_path(output)
        logger.success(f"Output path validated: {output_path}")
        
        # Get audio duration
        logger.debug("Getting audio duration")
        duration = get_audio_duration(input_path)
        logger.info(f"Audio duration: {format_time(duration)}")
        
        # Create transcriber
        logger.debug(f"Creating transcriber with model={model}, device={device}, compute_type={compute_type}")
        transcriber = AudioTranscriber(
            model_size=model,
            device=device,
            compute_type=compute_type,
            debug=debug,
            verbose=verbose
        )
        
        # Set up transcription parameters
        transcription_params = {
            'beam_size': beam_size,
            'vad_filter': True,
            'vad_parameters': dict(min_silence_duration_ms=1000),
            'temperature': temperature
        }
        
        logger.optimization_info(f"Using parameters: beam_size={beam_size}, temperature={temperature}")
        
        # Transcribe the file
        results = transcriber.transcribe_file(
            input_path,
            output_path,
            transcription_params=transcription_params
        )
        
        # Clean up
        transcriber.close()
        
        # Success message
        logger.success("Transcription completed successfully!")
        if verbose:
            logger.info(f"Language: {results['language']} (confidence: {results['language_probability']:.2f})")
            logger.info(f"Segments: {results['segment_count']}")
            if results['speed_ratio'] > 0:
                logger.info(f"Processing speed: {results['speed_ratio']:.1f}x real-time")
        
    except (FileNotFoundError, ValueError, PermissionError, ImportError, RuntimeError) as e:
        logger.error(str(e))
        raise typer.Exit(1)
    except KeyboardInterrupt:
        logger.error("Transcription interrupted by user")
        raise typer.Exit(1)


@app.command()
def cache_stats():
    """Show model cache statistics"""
    logger = get_logger()
    
    try:
        from .models import get_model_cache_stats
        
        stats = get_model_cache_stats()
        logger.info("📊 Model Cache Statistics:")
        logger.info(f"   Cached models: {stats['cached_models']}")
        logger.info(f"   Total accesses: {stats['total_accesses']}")
        
        if stats['models']:
            logger.info("   Model details:")
            for model_key, details in stats['models'].items():
                logger.info(f"     {model_key}: {details['accesses']} accesses")
        else:
            logger.info("   No models currently cached")
            
    except Exception as e:
        logger.error(f"Failed to get cache statistics: {e}")


@app.command()
def preload(
    model: Annotated[str, typer.Argument(help="Model size to preload")] = "base",
    device: Annotated[str, typer.Option(help="Device for inference")] = "auto",
    compute_type: Annotated[str, typer.Option("--compute-type", help="Compute type")] = "auto",
):
    """Pre-load a model for faster subsequent usage"""
    logger = get_logger()
    
    try:
        from .models import preload_model_for_config
        
        logger.info(f"🚀 Pre-loading model '{model}' for faster startup...")
        preload_model_for_config(model, device, compute_type)
        logger.success(f"Model '{model}' pre-loaded and ready")
        
    except Exception as e:
        logger.error(f"Failed to preload model: {e}")
        raise typer.Exit(1)


@app.command()
def info():
    """Show system information and capabilities"""
    logger = get_logger(verbose=True)
    
    try:
        from .device_detection import detect_device_capabilities, report_device_capabilities
        capabilities = detect_device_capabilities()
        report_device_capabilities(capabilities, verbose=True)
            
    except Exception as e:
        logger.error(f"Failed to get system information: {e}")


def main():
    """Main entry point for the CLI"""
    app()


if __name__ == "__main__":
    main()