import os

# Suppress TQDM output from underneath pyannote and whisper
os.environ["TQDM_DISABLE"] = "1"

import glob
import torch
import warnings
import logging

# Suppress torchvision and lightning noise from pyannote
warnings.filterwarnings("ignore", module="torchvision")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

from dotenv import load_dotenv

# Initialize local modules
from config_loader import load_global_config
from utils import get_series_name, get_unique_filename
from core.pipeline import TranscriptionPipeline
import time
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text

load_dotenv()
console = Console()

class CustomTaskProgressColumn(TaskProgressColumn):
    """A custom column that hides the percentage when total is None (indeterminate)."""
    def render(self, task):
        if task.total is None:
            return Text("")
        return super().render(task)

def process_podcasts(time_limit=None):
    # Load master configuration
    global_config = load_global_config()
    
    input_dir = global_config.paths.input_dir
    output_dir = global_config.paths.output_dir
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Find active audio files
    supported_extensions = ("*.mp3", "*.m4a", "*.wav", "*.flac")
    audio_files = []
    for ext in supported_extensions:
        audio_files.extend(glob.glob(os.path.join(input_dir, ext)))
        
    if not audio_files:
        console.print(f"[yellow]No audio files found in '{input_dir}' directory.[/]")
        return

    console.print(f"[bold green]Audio files found to process: {len(audio_files)}[/]")
    
    # --- PHASE 1: SERIES ANALYSIS & SETUP ---
    console.rule("[bold cyan]Phase 1: Series Analysis & Setup")
    
    unique_series = {}
    for audio_path in audio_files:
        series = get_series_name(os.path.basename(audio_path))
        if series not in unique_series:
            unique_series[series] = []
        unique_series[series].append(audio_path)
        
    console.print(f"Found {len(unique_series)} unique podcast series in the batch.")
    
    series_models = {}
    import sys, select
    from speaker_manager import load_series_config, save_series_config

    def input_with_timeout(prompt, timeout=10.0):
        console.print(prompt, end="")
        i, o, e = select.select([sys.stdin], [], [], timeout)
        if i:
            return sys.stdin.readline().strip()
        console.print() 
        return None

    for series_name, files in unique_series.items():
        config_db = load_series_config(series_name)
        saved_model = config_db.get("diarization_model")
        
        console.print(f"\n[bold]Series:[/] [cyan]{series_name}[/] ({len(files)} files)")
        
        if not saved_model:
            console.print("[yellow]❓ No historical model selected for this series.[/]")
            console.print("Select Diarization Strategy:")
            console.print("  [1] Skip Diarization (1 Speaker only) - Fastest")
            console.print("  [2] Fast Model (pyannote/speaker-diarization-2.1)")
            console.print("  [3] Accurate Model - Overlapping speakers (pyannote/speaker-diarization-3.1)")
            
            choice = input_with_timeout(f"[bold yellow]Enter 1, 2, or 3 within 10s (default is 3): [/]", timeout=10.0)

            if choice == '1': series_models[series_name] = "skip"
            elif choice == '2': series_models[series_name] = "pyannote/speaker-diarization-2.1"
            elif choice == '3': series_models[series_name] = "pyannote/speaker-diarization-3.1"
            else:
                series_models[series_name] = "pyannote/speaker-diarization-3.1"
                console.print("[dim]Defaulting to Accurate Model (3).[/]")
                
            config_db["diarization_model"] = series_models[series_name]
            save_series_config(series_name, config_db)
        else:
            model_display = "Skip Diarization" if saved_model == "skip" else saved_model
            console.print(f"[dim]Historically selected model: {model_display}[/]")
            console.print("  [1] Skip Diarization | [2] Fast Model | [3] Accurate Model")
            
            choice = input_with_timeout(f"[yellow]Enter 1, 2, or 3 within 10s to Change (or wait): [/]", timeout=10.0)
            
            if choice in ['1', '2', '3']:
                if choice == '1': series_models[series_name] = "skip"
                elif choice == '2': series_models[series_name] = "pyannote/speaker-diarization-2.1"
                elif choice == '3': series_models[series_name] = "pyannote/speaker-diarization-3.1"
                config_db["diarization_model"] = series_models[series_name]
                save_series_config(series_name, config_db)
            else:
                series_models[series_name] = saved_model

    # --- PHASE 2: PRE-FLIGHT CHECKS ---
    console.rule("[bold cyan]Phase 2: Preparation & Pre-flight")
    
    if global_config.post_processing.enabled and global_config.post_processing.provider.lower() == "ollama":
        import requests, subprocess
        try:
            requests.get("http://localhost:11434/", timeout=2)
        except requests.exceptions.RequestException:
            console.print("\n[bold red]⚠️ Ollama is NOT running![/]")
            if console.input("[bold]Start Ollama automatically? [Y/N]: [/]").strip().upper() == 'Y':
                subprocess.run(["open", "-a", "Ollama"])
                time.sleep(10)
            else: return

    files_to_process = []
    for audio_path in audio_files:
        series_name = get_series_name(os.path.basename(audio_path))
        chosen_model = series_models[series_name]
        base_name, _ = os.path.splitext(os.path.basename(audio_path))
        output_txt = os.path.join(output_dir, f"{base_name}.txt")
        
        if os.path.exists(output_txt):
            choice = console.input(f"\n[bold yellow]'{output_txt}' exists.[/] [O]verwrite, [R]ename, [S]kip, [Q]uit: ").strip().upper()
            if choice == 'R': output_txt = get_unique_filename(output_txt)
            elif choice == 'S': continue
            elif choice == 'Q': return
                
        files_to_process.append((audio_path, series_name, chosen_model, base_name, output_txt))

    if not files_to_process:
        console.print("[bold yellow]No files to process. Exiting.[/]")
        return

    # --- PHASE 3: EXECUTION ---
    console.rule("[bold cyan]Phase 3: Audio Processing Execution")
    
    from concurrent.futures import ThreadPoolExecutor
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        CustomTaskProgressColumn(),
        console=console,
    ) as progress:
        job_task = progress.add_task("[cyan]Batch Progress", total=len(files_to_process))
        
        # Initialize the pipeline
        pipeline = TranscriptionPipeline(
            config=global_config,
            progress_callback=lambda msg: progress.update(file_task, description=f"[magenta]{msg}")
        )
        
        successfully_processed = []
        
        # Using 1 worker for LLM to avoid overwhelming local Ollama or API limits, 
        # but allowing it to run in background while GPU is busy with the next file.
        with ThreadPoolExecutor(max_workers=1) as executor:
            for audio_path, series_name, chosen_model, base_name, output_txt in files_to_process:
                file_task = progress.add_task(f"[magenta]{base_name}: Starting...", total=None)
                
                success = pipeline.process_file(
                    audio_path, series_name, chosen_model, base_name, output_txt, 
                    time_limit, executor=executor
                )
                
                if success:
                    successfully_processed.append(audio_path)
                    progress.remove_task(file_task)
                    progress.update(job_task, advance=1)
                else:
                    progress.console.print(f"   ❌ [bold red]Failed to process {base_name}[/]")
            
            if successfully_processed:
                progress.update(job_task, description="[cyan]Waiting for background tasks...")

    if successfully_processed:
        if console.input(f"\n[bold]Delete {len(successfully_processed)} original media files? [Y/N]: ").strip().upper() == 'Y':
            for path in successfully_processed:
                try: os.remove(path)
                except Exception as e: console.print(f"   ❌ [red]Could not delete {path}: {e}[/]")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Podcast Transcribe.")
    parser.add_argument("--test", type=int, help="Test run mode (seconds).")
    args, _ = parser.parse_known_args()
    
    console.print(Panel.fit("[bold violet]🎙️ PODCASTS TRANSCRIBE 🤖[/]", border_style="violet"))
    process_podcasts(time_limit=args.test)
    console.print("\n[bold green]🎉 ALL FILES PROCESSED SUCCESSFULLY![/]")
