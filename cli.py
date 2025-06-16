"""
Click-based command line interface for audio transcription
"""
import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)

from transcriber import AudioTranscriber
from utils import format_time, estimate_processing_time


# Create console for rich output
console = Console()


class ProgressTracker:
    """Track and display transcription progress"""
    
    def __init__(self, show_progress: bool = True):
        self.show_progress = show_progress
        self.progress = None
        self.task_id = None
        self.last_update = time.time()
        
    def __enter__(self):
        if self.show_progress:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console
            )
            self.progress.__enter__()
            self.task_id = self.progress.add_task("Transcribing audio...", total=100)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress:
            self.progress.__exit__(exc_type, exc_val, exc_tb)
    
    def update(self, progress_percentage: float, segment_count: int, elapsed_time: float):
        """Update progress display"""
        if self.progress and self.task_id is not None:
            # Update only every 0.5 seconds to avoid too frequent updates
            current_time = time.time()
            if current_time - self.last_update >= 0.5:
                self.progress.update(
                    self.task_id,
                    completed=progress_percentage,
                    description=f"Transcribing audio... ({segment_count} segments)"
                )
                self.last_update = current_time


@click.command(name="transcribe")
@click.argument(
    'input_file',
    type=click.Path(exists=True, readable=True, path_type=Path),
    metavar='INPUT_FILE'
)
@click.option(
    '-o', '--output',
    required=True,
    type=click.Path(path_type=Path),
    help='Path for the output text file'
)
@click.option(
    '--model',
    type=click.Choice(['tiny', 'base', 'small', 'medium', 'large'], case_sensitive=False),
    default='base',
    show_default=True,
    help='Whisper model size to use'
)
@click.option(
    '--device',
    type=click.Choice(['auto', 'cpu', 'cuda', 'mps'], case_sensitive=False),
    default='auto',
    show_default=True,
    help='Force specific device for inference'
)
@click.option(
    '--compute-type',
    type=click.Choice(['auto', 'float32', 'float16', 'int8', 'int8_float16'], case_sensitive=False),
    default='auto',
    show_default=True,
    help='Force specific compute type'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Show detailed system information and optimization hints'
)
@click.option(
    '--quiet', '-q',
    is_flag=True,
    help='Suppress progress display and non-essential output'
)
@click.version_option(version="1.0.0", prog_name="transcribe")
def main(input_file, output, model, device, compute_type, verbose, quiet):
    """
    Transcribe audio files to text using faster-whisper (offline)
    
    INPUT_FILE: Path to the audio file to transcribe
    
    Supported audio formats: MP3, WAV, M4A, OGG, FLAC, AAC, WMA
    """
    try:
        # Set console quiet mode
        if quiet:
            console.quiet = True
        
        # Create transcriber instance
        transcriber = AudioTranscriber(
            model_size=model.lower(),
            device=device.lower() if device != 'auto' else None,
            compute_type=compute_type.lower() if compute_type != 'auto' else None
        )
        
        # Validate inputs and get basic info
        if not quiet:
            console.print(f"✓ Input file validated: [green]{input_file}[/green]")
        
        # Get audio duration for estimation
        try:
            duration = transcriber.get_audio_duration(str(input_file))
            if not quiet:
                console.print(f"✓ Audio duration: [blue]{format_time(duration)}[/blue]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not determine audio duration: {e}[/yellow]")
            duration = 0
        
        # Show model information if verbose
        if verbose:
            model_info = transcriber.get_model_info()
            console.print("\n[bold]Model Configuration:[/bold]")
            console.print(f"  Model size: [cyan]{model_info['model_size']}[/cyan]")
            console.print(f"  Device: [cyan]{model_info['device']}[/cyan]")
            console.print(f"  Compute type: [cyan]{model_info['compute_type']}[/cyan]")
            
            if duration > 0:
                estimated_time = estimate_processing_time(duration, model_info['model_size'])
                console.print(f"  Estimated processing time: [yellow]{format_time(estimated_time)}[/yellow]")
            console.print()
        
        # Validate output path
        transcriber.validate_output_path(str(output))
        if not quiet:
            console.print(f"✓ Output path validated: [green]{output}[/green]")
        
        # Set up progress tracking
        progress_tracker = ProgressTracker(show_progress=not quiet)
        
        # Start transcription
        if not quiet:
            console.print(f"\n[bold]Loading Whisper model '{model}'...[/bold]")
        
        start_time = time.time()
        
        with progress_tracker:
            # Transcribe the file
            result = transcriber.transcribe_file(
                str(input_file),
                str(output),
                progress_callback=progress_tracker.update if not quiet else None
            )
        
        # Show results
        if not quiet:
            console.print(f"\n[bold green]✓ Transcription completed![/bold green]")
            console.print(f"  Language: [cyan]{result['language']}[/cyan] (probability: {result['language_probability']:.2f})")
            console.print(f"  Audio duration: [blue]{format_time(result['audio_duration'])}[/blue]")
            console.print(f"  Processing time: [blue]{format_time(result['processing_time'])}[/blue]")
            console.print(f"  Speed ratio: [yellow]{result['speed_ratio']:.1f}x[/yellow] faster than real-time")
            console.print(f"  Total segments: [cyan]{result['segment_count']}[/cyan]")
            console.print(f"  Output saved to: [green]{result['output_path']}[/green]")
        else:
            # Minimal output for quiet mode
            console.print(f"Transcription completed: {result['output_path']}")
        
        return 0
        
    except FileNotFoundError as e:
        error_msg = f"Error: {e}"
        console.print(f"[red]{error_msg}[/red]", err=True)
        click.echo(error_msg, err=True)  # Also output plain text for testing
        return 1
    except ValueError as e:
        error_msg = f"Error: {e}"
        console.print(f"[red]{error_msg}[/red]", err=True)
        click.echo(error_msg, err=True)
        return 1
    except PermissionError as e:
        error_msg = f"Error: {e}"
        console.print(f"[red]{error_msg}[/red]", err=True)
        click.echo(error_msg, err=True)
        return 1
    except ImportError as e:
        error_msg = f"Error: {e}"
        console.print(f"[red]{error_msg}[/red]", err=True)
        console.print("[yellow]Hint: Install faster-whisper with: pip install faster-whisper[/yellow]", err=True)
        click.echo(error_msg, err=True)
        click.echo("Hint: Install faster-whisper with: pip install faster-whisper", err=True)
        return 1
    except RuntimeError as e:
        error_msg = f"Error: {e}"
        console.print(f"[red]{error_msg}[/red]", err=True)
        click.echo(error_msg, err=True)
        return 1
    except KeyboardInterrupt:
        error_msg = "Transcription interrupted by user"
        console.print(f"\n[yellow]{error_msg}[/yellow]", err=True)
        click.echo(error_msg, err=True)
        return 1
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        console.print(f"[red]{error_msg}[/red]", err=True)
        click.echo(error_msg, err=True)
        if verbose:
            import traceback
            trace = traceback.format_exc()
            console.print(f"[dim]{trace}[/dim]", err=True)
            click.echo(trace, err=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())