import os
import glob
import torch
import mlx_whisper
from pyannote.audio import Pipeline
from dotenv import load_dotenv

# Initialize local modules
from config_loader import load_global_config
from utils import get_series_name, get_unique_filename
from audio_converter import convert_to_wav
from speaker_manager import load_series_config, load_series_embeddings, save_series_config, save_series_embeddings, get_global_speaker_mapping, get_speaker, merge_duplicate_speakers
from post_processing import run_post_processing
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()

def process_podcasts(time_limit=None):
    # Load master configuration
    global_config = load_global_config()
    paths_cfg = global_config.get("paths", {})
    
    input_dir = paths_cfg.get("input_dir", "input")
    output_dir = paths_cfg.get("output_dir", "output")
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Find active audio extensions in the input folder
    supported_extensions = ("*.mp3", "*.m4a", "*.wav", "*.flac")
    audio_files = []
    
    for ext in supported_extensions:
        audio_files.extend(glob.glob(os.path.join(input_dir, ext)))
        
    if not audio_files:
        console.print(f"[yellow]No audio files found in '{input_dir}' directory.[/]")
        return

    console.print(f"[bold green]Audio files found to process: {len(audio_files)}[/]")
    
    whisper_kwargs = global_config["transcription"]
    diarization_cfg = global_config["diarization"]
    cache_cfg = global_config["cache"]
    
    similarity_threshold = diarization_cfg.get("similarity_threshold", 0.35)
    ema_alpha = diarization_cfg.get("ema_alpha", 0.1)
    auto_merge_duplicates = diarization_cfg.get("auto_merge_duplicates", True)
    
    console.print("[dim]Settings loaded from config.yaml[/]")

    # Load speaker recognition model
    with console.status("[bold cyan]Loading speaker diarization model...", spinner="dots"):
        try:
            device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            diarization_pipeline = Pipeline.from_pretrained(
                diarization_cfg.get("model", "pyannote/speaker-diarization-3.1"),
                token=os.environ.get("HF_TOKEN")
            )
            diarization_pipeline.to(device)
        except Exception as e:
            console.print(f"[bold red]Error loading diarization model:[/] {e}")
            console.print("[yellow]💡 Make sure you have accepted the pyannote model terms:[/]")
            console.print("   1. https://huggingface.co/pyannote/speaker-diarization-3.1")
            console.print("   2. https://huggingface.co/pyannote/segmentation-3.0")
            console.print("   And verify that your HF_TOKEN is specified in the .env file.")
            return

    for audio_path in audio_files:
        # Determine podcast series from filename
        series_name = get_series_name(os.path.basename(audio_path))
        
        # Get base name without extension and path
        base_name, _ = os.path.splitext(os.path.basename(audio_path))
        output_txt = os.path.join(output_dir, f"{base_name}.txt")
        
        if os.path.exists(output_txt):
            # Interactive menu if file exists
            console.print(f"\n[bold yellow][WARNING] Output file '{output_txt}' already exists.[/]")
            while True:
                choice = console.input("[bold]Action? [O]verwrite, [R]ename, [S]kip file, [Q]uit script: [/]").strip().upper()
                if choice == 'O':
                    console.print("[dim]File will be overwritten.[/]")
                    break
                elif choice == 'R':
                    output_txt = get_unique_filename(output_txt)
                    console.print(f"[dim]New filename: '{output_txt}'.[/]")
                    break
                elif choice == 'S':
                    console.print("[dim]Skipping this file.[/]")
                    break
                elif choice == 'Q':
                    console.print("[dim]Script stopped by user.[/]")
                    return
                else:
                    console.print("[red]Invalid choice. Please enter O, R, S, or Q.[/]")
            if choice == 'S':
                continue
            
        console.rule(f"[bold blue]Processing: {os.path.basename(audio_path)} (Series: {series_name})")
        
        # Load saved speaker profiles for this specific series
        config_db = load_series_config(series_name)
        embeddings_db = load_series_embeddings(series_name)
        
        # Automated duplicate speaker merging
        if auto_merge_duplicates:
            if merge_duplicate_speakers(series_name, config_db, embeddings_db):
                save_series_config(series_name, config_db)
                save_series_embeddings(series_name, embeddings_db)
                
                
        try:
            with console.status("[bold magenta]Step 1/3: Diarization (who speaks when)...", spinner="point"):
                temp_wav = convert_to_wav(audio_path, time_limit, cache_cfg=cache_cfg)
                
                # File is processed directly from the persistent cache
                diarization = diarization_pipeline(temp_wav)
                
                speaker_mapping = get_global_speaker_mapping(
                    diarization, series_name, config_db, embeddings_db, 
                    similarity_threshold, ema_alpha
                )
                
                # Save the actively updated speaker profiles back to disk immediately
                save_series_config(series_name, config_db)
                save_series_embeddings(series_name, embeddings_db)

            with console.status("[bold magenta]Step 2/3: Transcription via MLX Whisper...", spinner="point"):
                # Whisper parses the 16kHz cached WAV (faster than MP3 inference)
                transcribe_source = temp_wav
                
                # Transcription via MLX on Apple Silicon GPU
                result = mlx_whisper.transcribe(
                    transcribe_source,
                    verbose=False, # Hardcoded false to prevent CLI bloat
                    **whisper_kwargs
                )

            with console.status("[bold magenta]Step 3/3: Merging text and identifying globals...", spinner="point"):
                with open(output_txt, "w", encoding="utf-8") as f:
                    for segment in result["segments"]:
                        start_time = segment["start"]
                        end_time = segment["end"]
                        text = segment["text"].strip()
                        
                        # Format time as MM:SS.mmm for readable outputs
                        start_fmt = f"{int(start_time // 60):02d}:{start_time % 60:06.3f}"
                        end_fmt = f"{int(end_time // 60):02d}:{end_time % 60:06.3f}"
                        
                        # Interruption aware speaker resolving
                        speaker = get_speaker(diarization, start_time, end_time, speaker_mapping)
                        
                        line = f"[{start_fmt} -> {end_fmt}] {speaker}: {text}\n"
                        f.write(line)
            
            console.print(f"   ✅ [bold green]Raw transcription saved:[/] {output_txt}")
            
            # 4. LLM Post-Processing Magic
            if global_config.get("post_processing", {}).get("enabled", False):
                with console.status("[bold magenta]Step 4: LLM Post-Processing (Formatting & Summary)...", spinner="point"):
                    with open(output_txt, "r", encoding="utf-8") as f:
                        raw_text = f.read()
                    
                    formatted_text = run_post_processing(raw_text, global_config)
                    
                    final_output_path = os.path.join(output_dir, f"{base_name}_formatted.md")
                    with open(final_output_path, "w", encoding="utf-8") as f:
                        f.write(formatted_text)
                        
                console.print(f"   ✨ [bold green]LLM formatted transcript saved:[/] {final_output_path}")
                
            # Ask to delete the original audio
            console.print(f"\n[bold yellow]❓ Finished processing '{os.path.basename(audio_path)}'[/]")
            while True:
                del_choice = console.input("[bold]Delete original media file? [Y]es, [N]o: [/]").strip().upper()
                if del_choice == 'Y':
                    try:
                        os.remove(audio_path)
                        console.print(f"   🗑️ [dim]Deleted {audio_path}[/]")
                    except Exception as e:
                        console.print(f"   ❌ [red]Could not delete {audio_path}: {e}[/]")
                    break
                elif del_choice == 'N':
                    console.print("   💾 [dim]Kept original file.[/]")
                    break
                else:
                    console.print("[red]Invalid choice. Please enter Y or N.[/]")
                    
        except Exception as e:
            console.print(f"   ❌ [bold red]Failed to process file {audio_path}:[/] {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Podcast Transcribe. Converts audio to text partitioned by global speakers.")
    parser.add_argument("--test", type=int, help="Test run mode. Process only N seconds of the audio. Example: --test 60")
    parser.add_argument("--test-run", action='store_true', help="Hidden test flag wrapper") # For IDE silence
    
    args, unknown = parser.parse_known_args()
    
    console.print(Panel.fit("[bold violet]🎙️ PODCASTS TRANSCRIBE 🤖[/]", border_style="violet"))
    process_podcasts(time_limit=args.test)
    console.print("\n[bold green]🎉 ALL FILES PROCESSED SUCCESSFULLY![/]")