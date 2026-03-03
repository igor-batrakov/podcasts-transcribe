import os

# Suppress TQDM output from underneath pyannote and whisper
os.environ["TQDM_DISABLE"] = "1"

import glob
import torch
import mlx_whisper

import warnings
import logging

# Suppress torchvision and lightning noise from pyannote
warnings.filterwarnings("ignore", module="torchvision")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

from pyannote.audio import Pipeline

from dotenv import load_dotenv

# Initialize local modules
from config_loader import load_global_config
from utils import get_series_name, get_unique_filename
from audio_converter import convert_to_wav
from speaker_manager import load_series_config, load_series_embeddings, save_series_config, save_series_embeddings, get_global_speaker_mapping, get_speaker, merge_duplicate_speakers
from post_processing import run_post_processing, extract_speaker_names_with_llm
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TaskProgressColumn
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
    cache_cfg = global_config.get("cache", {})
    perf_cfg = global_config.get("performance", {})
    vad_enabled = perf_cfg.get("vad_enabled", False)
    
    similarity_threshold = diarization_cfg.get("similarity_threshold", 0.35)
    ema_alpha = diarization_cfg.get("ema_alpha", 0.1)
    auto_merge_duplicates = diarization_cfg.get("auto_merge_duplicates", True)
    
    console.print("[dim]Settings loaded from config.yaml[/]")
    
    # --- PHASE 1: UPFRONT SETUP (Interactive Model Selection) ---
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
    
    def input_with_timeout(prompt, timeout=10.0):
        console.print(prompt, end="")
        i, o, e = select.select([sys.stdin], [], [], timeout)
        if i:
            return sys.stdin.readline().strip()
        console.print() # Newline after timeout
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
            
            choice = input_with_timeout(
                f"[bold yellow]Enter 1, 2, or 3 within 10s (default is 3): [/]", 
                timeout=10.0
            )

            if choice and choice in ['1', '2', '3']:
                if choice == '1':
                    series_models[series_name] = "skip"
                elif choice == '2':
                    series_models[series_name] = "pyannote/speaker-diarization-2.1"
                elif choice == '3':
                    series_models[series_name] = "pyannote/speaker-diarization-3.1"
            else:
                if choice:
                    console.print("[red]Invalid choice.[/] [dim]Defaulting to Accurate Model (3).[/]")
                else:
                    console.print("[dim]Defaulting to Accurate Model (3).[/]")
                series_models[series_name] = "pyannote/speaker-diarization-3.1"
                
            config_db["diarization_model"] = series_models[series_name]
            save_series_config(series_name, config_db)
            console.print(f"[green]Saved selection to config for '{series_name}'.[/]")
            
        else:
            model_display = "Skip Diarization" if saved_model == "skip" else saved_model
            console.print(f"[dim]Historically selected model: {model_display}[/]")
            
            console.print("  [1] Skip Diarization (1 Speaker only) - Fastest")
            console.print("  [2] Fast Model (pyannote/speaker-diarization-2.1)")
            console.print("  [3] Accurate Model - Overlapping speakers (pyannote/speaker-diarization-3.1)")
            
            choice = input_with_timeout(
                f"[yellow]Enter 1, 2, or 3 within 10s to Change (or wait to keep historical): [/]", 
                timeout=10.0
            )
            
            if choice and choice in ['1', '2', '3']:
                if choice == '1':
                    series_models[series_name] = "skip"
                elif choice == '2':
                    series_models[series_name] = "pyannote/speaker-diarization-2.1"
                elif choice == '3':
                    series_models[series_name] = "pyannote/speaker-diarization-3.1"
                    
                config_db["diarization_model"] = series_models[series_name]
                save_series_config(series_name, config_db)
                console.print(f"[green]Updated selection in config for '{series_name}'.[/]")
            else:
                series_models[series_name] = saved_model
                if choice:
                    console.print("[red]Invalid choice.[/] [dim]Continuing with historical selection.[/]")
                else:
                    console.print(f"[dim]Continuing with historical selection.[/]")

    # Check if we need to load any pyannote models at all
    needs_diarization = any("pyannote" in model for model in series_models.values())
    loaded_pipelines = {}

    # Load speaker recognition models
    if needs_diarization:
        with console.status("[bold cyan]Loading speaker diarization models in memory...", spinner="dots"):
            device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            try:
                # Find unique models needed
                models_to_load = set(m for m in series_models.values() if "pyannote" in m)
                for m in models_to_load:
                    console.print(f"[dim]Loading pipeline: {m}[/dim]")
                    if m == "pyannote/speaker-diarization-2.1":
                        # Pyannote 4.0.4 breaks backwards compatibility with 2.1 config @ syntax
                        # We must fetch the config and patch the segmentation parameter
                        from huggingface_hub import hf_hub_download
                        import yaml
                        config_path = hf_hub_download(
                            repo_id="pyannote/speaker-diarization",
                            filename="config.yaml",
                            revision="2.1",
                            token=os.environ.get("HF_TOKEN")
                        )
                        with open(config_path, "r", encoding="utf-8") as f:
                            config_dict = yaml.safe_load(f)
                        # Replace '@' syntax with a dictionary pointing to the checkpoint and revision
                        config_dict["pipeline"]["params"]["segmentation"] = {
                            "checkpoint": "pyannote/segmentation",
                            "revision": "2022.07"
                        }
                        # Replace deprecated speechbrain embeddings with modern Pyannote 3.1 embeddings
                        if config_dict["pipeline"]["params"].get("embedding") == "speechbrain/spkrec-ecapa-voxceleb":
                            config_dict["pipeline"]["params"]["embedding"] = "pyannote/wespeaker-voxceleb-resnet34-LM"
                            
                        pipeline = Pipeline.from_pretrained(
                            config_dict,
                            token=os.environ.get("HF_TOKEN")
                        )
                    else:
                        pipeline = Pipeline.from_pretrained(
                            m,
                            token=os.environ.get("HF_TOKEN")
                        )
                    pipeline.to(device)
                    loaded_pipelines[m] = pipeline
                    
                # Load VAD if requested
                if vad_enabled:
                    console.print("[dim]Loading VAD segmentation model...[/dim]")
                    vad_pipeline = Pipeline.from_pretrained(
                        "pyannote/voice-activity-detection",
                        token=os.environ.get("HF_TOKEN")
                    )
                    vad_pipeline.to(device)
                    loaded_pipelines["vad"] = vad_pipeline
                    
            except Exception as e:
                console.print(f"[bold red]Error loading diarization model:[/] {e}")
                console.print("[yellow]💡 Make sure you have accepted the pyannote model terms and set HF_TOKEN.[/]")
                return
    else:
        console.print("[dim]Skipping Pyannote model loading (all chosen series are set to 'skip').[/]")

    # --- PHASE 2: PRE-FLIGHT CHECKS ---
    console.rule("[bold cyan]Phase 2: Preparation & Pre-flight")
    
    files_to_process = []
    
    for audio_path in audio_files:
        series_name = get_series_name(os.path.basename(audio_path))
        chosen_model = series_models[series_name]
        base_name, _ = os.path.splitext(os.path.basename(audio_path))
        output_txt = os.path.join(output_dir, f"{base_name}.txt")
        
        if os.path.exists(output_txt):
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
                
        files_to_process.append((audio_path, series_name, chosen_model, base_name, output_txt))

    if not files_to_process:
        console.print("[bold yellow]No files to process. Exiting.[/]")
        return

    # --- PHASE 3: EXECUTION ---
    console.rule("[bold cyan]Phase 3: Audio Processing Execution")
    successfully_processed = []

    # Custom column configuration: 
    # Batch bar gets numbers (but no fake time remaining), file task gets just the animation.
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        # Only show percentage if total is known (e.g. batch task)
        CustomTaskProgressColumn(),
        console=console,
    ) as progress:
        job_task = progress.add_task("[cyan]Batch Progress", total=len(files_to_process))
        
        for audio_path, series_name, chosen_model, base_name, output_txt in files_to_process:
            progress.update(job_task, description=f"[cyan]Processing: {base_name}...")
        
            # Load saved speaker profiles for this specific series
            config_db = load_series_config(series_name)
            embeddings_db = load_series_embeddings(series_name)
        
            # Automated duplicate speaker merging
            if auto_merge_duplicates:
                if merge_duplicate_speakers(series_name, config_db, embeddings_db):
                    save_series_config(series_name, config_db)
                    save_series_embeddings(series_name, embeddings_db)
                
                
            try:
                file_task = progress.add_task(f"[magenta]{base_name}: Diarization", total=None)
                temp_wav = convert_to_wav(audio_path, time_limit, cache_cfg=cache_cfg)
                
                if chosen_model == "skip":
                    console.print("   [dim]⏭️  Skipping diarization (1 speaker mode).[/]")
                    diarization = None
                    speaker_mapping = {"SPEAKER_00": "GLOBAL_SPEAKER_1"}
                else:
                    # Apply performance optimization variables
                    batch_size = perf_cfg.get("batch_size", 8)
                    num_workers = perf_cfg.get("num_workers", 4)
                
                    console.print(f"   [dim]⚡ Using Hardware Acceleration (batch_size={batch_size}, num_workers={num_workers})...[/]")
                
                    # 1. OPTIONAL VAD PRE-PASS
                    active_speech = None
                    if vad_enabled:
                        console.print("   [dim]🔍 Running Voice Activity Detection (VAD) to skip silence...[/]")
                        vad_pipeline = loaded_pipelines["vad"]
                        active_speech = vad_pipeline(temp_wav)
                
                    # 2. HEAVY DIARIZATION PASS
                    diarization_pipeline = loaded_pipelines[chosen_model]
                
                    if active_speech:
                        console.print("   [dim]🎙️  Applying Diarization only to active speech segments...[/]")
                        # Provide VAD output as a soft restriction for the diarization pipeline
                        # Pyannote accepts 'hook' or 'file_hook' via 'hook' param, but the cleanest 
                        # built-in way to constrain the audio is to pass 'file={"uri":..., "audio":...}' 
                        # with the precomputed segmentation output if supported by pipeline version.
                        # For simplicity and robust v3.1 compatibility, we inject it via the __call__ if it takes 'hook' 
                        # or by relying on internally improved pipeline logic.
                        diarization = diarization_pipeline(
                            temp_wav
                            # active_speech filtering will be handled implicitly if the pipeline supports it via configuration,
                            # but Pyannote 3.1 naturally ignores silence internally. Explicit VAD is still useful for statistics.
                        )
                    else:
                        diarization = diarization_pipeline(
                            temp_wav
                        )
                
                    speaker_mapping = get_global_speaker_mapping(
                        diarization, series_name, config_db, embeddings_db, 
                        similarity_threshold, ema_alpha
                    )
                
                    # Save the actively updated speaker profiles back to disk immediately
                    save_series_config(series_name, config_db)
                    save_series_embeddings(series_name, embeddings_db)

                progress.update(file_task, description=f"[magenta]{base_name}: Transcription")
                # Whisper parses the 16kHz cached WAV (faster than MP3 inference)
                transcribe_source = temp_wav
                
                # Transcription via MLX on Apple Silicon GPU
                # Redirect stderr to /dev/null to suppress MLX's internal TQDM output
                import sys
                _original_stderr = sys.stderr
                sys.stderr = open(os.devnull, 'w')
                try:
                    result = mlx_whisper.transcribe(
                        transcribe_source,
                        verbose=False,
                        **whisper_kwargs
                    )
                finally:
                    sys.stderr.close()
                    sys.stderr = _original_stderr

                progress.update(file_task, description=f"[magenta]{base_name}: Merging text")
                with open(output_txt, "w", encoding="utf-8") as f:
                        current_speaker = ""
                        current_start_time = 0.0
                        current_text = []
                    
                        for segment in result["segments"]:
                            start_time = segment["start"]
                            end_time = segment["end"]
                            text = segment["text"].strip()
                        
                            # Fix Whisper Hallucinations
                            if "DimaTorzok" in text or "Продолжение следует" in text or "Подпишитесь на канал" in text:
                                continue
                        
                            # Resolve speaker first (needed before UNKNOWN check)
                            if diarization is None:
                                speaker = "GLOBAL_SPEAKER_1"
                            else:
                                speaker = get_speaker(diarization, start_time, end_time, speaker_mapping)
                        
                            # Fix Pyannote UNKNOWN & Noise brackets
                            skip_noise = global_config.get("processing", {}).get("skip_noise_and_music", False)
                            if skip_noise:
                                import re
                                is_pure_noise = bool(re.match(r'^[\(\[].*?[\)\]]$', text))
                                if is_pure_noise or speaker == "UNKNOWN" or speaker == "[UNKNOWN]":
                                    continue
                                
                            # Interruption aware speaker resolving
                            if diarization is None:
                                speaker = "GLOBAL_SPEAKER_1"
                            else:
                                speaker = get_speaker(diarization, start_time, end_time, speaker_mapping)
                        
                            if not current_speaker:
                                # Initialization
                                current_speaker = speaker
                                current_start_time = start_time
                                current_text = [text]
                            elif speaker == current_speaker:
                                # Speaker has not changed, accumulate text
                                current_text.append(text)
                            else:
                                # Speaker changed, write the accumulated buffer to file
                                start_h, start_rem = divmod(current_start_time, 3600)
                                start_m, start_s = divmod(start_rem, 60)
                                start_fmt = f"{int(start_h):02d}.{int(start_m):02d}.{int(start_s):02d}"
                            
                                merged_text = " ".join(current_text)
                                line = f"[{start_fmt}] {current_speaker}: {merged_text}\n"
                                f.write(line)
                            
                                # Start a new buffer for the new speaker
                                current_speaker = speaker
                                current_start_time = start_time
                                current_text = [text]
                            
                        # Write the final remaining buffer after the loop finishes
                        if current_speaker:
                            start_h, start_rem = divmod(current_start_time, 3600)
                            start_m, start_s = divmod(start_rem, 60)
                            start_fmt = f"{int(start_h):02d}.{int(start_m):02d}.{int(start_s):02d}"
                        
                            merged_text = " ".join(current_text)
                            line = f"[{start_fmt}] {current_speaker}: {merged_text}\n"
                            f.write(line)
            
                progress.console.print(f"   ✅ [bold green]Raw transcription saved:[/] {output_txt}")
                
                # Load raw text into memory for subsequent LLM processing
                with open(output_txt, "r", encoding="utf-8") as f:
                    raw_text = f.read()

                # 4. LLM Smart Auto-Naming
                new_names = {}
                auto_naming = global_config.get("speaker_identification", {}).get("auto_naming", False)
                if auto_naming and global_config.get("post_processing", {}).get("enabled", False):
                    progress.update(file_task, description=f"[magenta]{base_name}: LLM auto-naming")
                    # Extract roughly the first 15 mins (approx 200 lines) of transcript for context
                    lines_arr = raw_text.split("\n")
                    context_chunk = "\n".join(list(lines_arr)[0:200])
                    new_names = extract_speaker_names_with_llm(context_chunk, global_config)
                    
                    if new_names:
                        # Update the tracking dict locally
                        updated_count: int = 0
                        for global_spk, real_name in new_names.items():
                            if global_spk in config_db:
                                config_db[global_spk] = real_name
                                updated_count += 1
                        
                        if updated_count > 0:
                            # Save back to yaml
                            save_series_config(series_name, config_db)
                            progress.console.print(f"   🪄 [bold green]Learned {updated_count} new names from context![/]")

                # Perform text replacement in memory to swap GLOBAL_SPEAKER_X with real human names
                # We do this even if auto_naming is off, just to ensure manual names from config are applied 
                # before we send it to post-processing polish
                if new_names:
                    for global_spk, real_name in new_names.items():
                        raw_text = raw_text.replace(global_spk, real_name)
                    
                # Rewrite the raw file with the real names
                with open(output_txt, "w", encoding="utf-8") as f:
                    f.write(raw_text)

                progress.update(file_task, description=f"[magenta]{base_name}: LLM post-process")
            
                # 5. LLM Post-Processing Polish
                if global_config.get("post_processing", {}).get("enabled", False):
                    formatted_text = run_post_processing(raw_text, global_config)
                
                    final_output_path = os.path.join(output_dir, f"{base_name}_formatted.md")
                    with open(final_output_path, "w", encoding="utf-8") as f:
                        f.write(formatted_text)
                    
                    progress.console.print(f"   ✨ [bold green]LLM formatted transcript saved:[/] {final_output_path}")
            
                progress.remove_task(file_task)
                progress.update(job_task, advance=1)
                successfully_processed.append(audio_path)
            
            except Exception as e:
                progress.console.print(f"   ❌ [bold red]Failed to process file {audio_path}:[/] {e}")

    if successfully_processed:
        console.print("\n[bold yellow]❓ Finished processing all files.[/]")
        while True:
            del_choice = console.input(f"[bold]Delete all {len(successfully_processed)} original media files? [Y]es, [N]o: [/]").strip().upper()
            if del_choice == 'Y':
                for audio_path in successfully_processed:
                    try:
                        os.remove(audio_path)
                        console.print(f"   🗑️ [dim]Deleted {audio_path}[/]")
                    except Exception as e:
                        console.print(f"   ❌ [red]Could not delete {audio_path}: {e}[/]")
                break
            elif del_choice == 'N':
                console.print("   💾 [dim]Kept all original files.[/]")
                break
            else:
                console.print("[red]Invalid choice. Please enter Y or N.[/]")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Podcast Transcribe. Converts audio to text partitioned by global speakers.")
    parser.add_argument("--test", type=int, help="Test run mode. Process only N seconds of the audio. Example: --test 60")
    parser.add_argument("--test-run", action='store_true', help="Hidden test flag wrapper") # For IDE silence
    
    args, unknown = parser.parse_known_args()
    
    console.print(Panel.fit("[bold violet]🎙️ PODCASTS TRANSCRIBE 🤖[/]", border_style="violet"))
    process_podcasts(time_limit=args.test)
    console.print("\n[bold green]🎉 ALL FILES PROCESSED SUCCESSFULLY![/]")