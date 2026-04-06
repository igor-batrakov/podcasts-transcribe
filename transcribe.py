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
from post_processing import run_post_processing, extract_podcast_metadata_with_llm, unload_ollama_model
import gc
import time
import datetime
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
    
    # Check total memory for optimization
    import subprocess
    try:
        mem_bytes = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize']).decode('utf-8').strip())
        mem_gb = mem_bytes / (1024**3)
        if mem_gb <= 18:  # Target 16GB/8GB Macs
            console.print(f"\n[bold yellow]⚠️ Detected {mem_gb:.0f}GB Unified Memory. Activating Low-Memory Optimizations.[/]")
            perf_cfg["batch_size"] = min(perf_cfg.get("batch_size", 8), 1)  # Force batch size to 1 to save memory
            perf_cfg["num_workers"] = 0  # Disable multiprocessing workers to save memory
            console.print("   [dim]Disabled Pyannote multiprocess workers and reduced batch_size to 1.[/]")
            if whisper_kwargs.get("path_or_hf_repo", "").endswith("large-v3-ru-podlodka"):
                 console.print("   [dim]💡 Tip: You are using the 'large' Whisper model. If it crashes, change it to 'medium' in config.yaml.[/]")
    except Exception:
        pass
    
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
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    def load_pyannote_pipeline(m, dev):
        console.print(f"   [dim]Loading pipeline: {m}[/dim]")
        if m == "pyannote/speaker-diarization-2.1":
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
            config_dict["pipeline"]["params"]["segmentation"] = {
                "checkpoint": "pyannote/segmentation",
                "revision": "2022.07"
            }
            if config_dict["pipeline"]["params"].get("embedding") == "speechbrain/spkrec-ecapa-voxceleb":
                config_dict["pipeline"]["params"]["embedding"] = "pyannote/wespeaker-voxceleb-resnet34-LM"
            pipeline = Pipeline.from_pretrained(config_dict, token=os.environ.get("HF_TOKEN"))
        else:
            pipeline = Pipeline.from_pretrained(m, token=os.environ.get("HF_TOKEN"))
        pipeline.to(dev)
        return pipeline

    # --- PHASE 2: PRE-FLIGHT CHECKS ---
    console.rule("[bold cyan]Phase 2: Preparation & Pre-flight")
    
    # Check Ollama connection if LLM features are enabled
    pp_config = global_config.get("post_processing", {})
    if pp_config.get("enabled", False) and pp_config.get("provider", "ollama").lower() == "ollama":
        import requests
        console.print("[dim]Checking Ollama connection...[/]")
        try:
            requests.get("http://localhost:11434/", timeout=2)
            console.print("   [green]✓ Ollama is running.[/]")
        except requests.exceptions.RequestException:
            console.print("\n[bold red]⚠️ Ollama is NOT running![/]")
            start_choice = console.input("[bold]Would you like to try starting the Ollama app automatically? [Y]es, [N]o: [/]").strip().upper()
            if start_choice == 'Y':
                try:
                    subprocess.run(["open", "-a", "Ollama"])
                    console.print("[dim]Waiting for Ollama to initialize (10s)...[/]")
                    time.sleep(10)
                    requests.get("http://localhost:11434/", timeout=5)
                    console.print("   [green]✓ Ollama started successfully![/]")
                except Exception as e:
                    console.print(f"   [bold red]❌ Could not connect to Ollama: {e}[/]")
                    console.print("   [yellow]Please open the Ollama app manually from Launchpad and run the script again.[/]")
                    return
            else:
                console.print("[dim]Ollama is required for LLM features. Exiting.[/]")
                return

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
                # Setup Time Tracking
                file_start_time = time.time()
                t_convert_start = time.time()
                
                file_task = progress.add_task(f"[magenta]{base_name}: Diarization", total=None)
                temp_wav = convert_to_wav(audio_path, time_limit, cache_cfg=cache_cfg)
                
                t_convert_end = time.time()
                t_diarization_start = time.time()
                
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
                        vad_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", token=os.environ.get("HF_TOKEN"))
                        vad_pipeline.to(device)
                        active_speech = vad_pipeline(temp_wav)
                        del vad_pipeline
                        gc.collect()
                        if torch.backends.mps.is_available():
                            torch.mps.empty_cache()
                
                    # 2. HEAVY DIARIZATION PASS
                    diarization_pipeline = load_pyannote_pipeline(chosen_model, device)
                
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
                        
                    del diarization_pipeline
                    gc.collect()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    time.sleep(2)
                
                    speaker_mapping = get_global_speaker_mapping(
                        diarization, series_name, config_db, embeddings_db, 
                        similarity_threshold, ema_alpha
                    )
                
                    # Save the actively updated speaker profiles back to disk immediately
                    save_series_config(series_name, config_db)
                    save_series_embeddings(series_name, embeddings_db)

                t_diarization_end = time.time()
                t_transcribe_start = time.time()

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
                    
                import mlx.core as mx
                mx.metal.clear_cache()
                
                # Jingle / Quotes Heuristic analysis
                total_podcast_duration = 0.0
                speaker_durations = {}
                
                if diarization is not None:
                    # Pre-calculate speaker times to find "inserts" (< 1% total duration)
                    for segment in result["segments"]:
                        start_time = segment["start"]
                        end_time = segment["end"]
                        text = segment["text"].strip()
                        
                        skip_noise = global_config.get("processing", {}).get("skip_noise_and_music", False)
                        if skip_noise:
                            import re
                            if bool(re.match(r'^[\(\[].*?[\)\]]$', text)):
                                continue
                                
                        total_podcast_duration = max(total_podcast_duration, end_time)
                        
                        speaker = get_speaker(diarization, start_time, end_time, speaker_mapping)
                        if speaker != "UNKNOWN" and speaker != "[UNKNOWN]":
                            speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + (end_time - start_time)
                            
                insert_speakers = set()
                if total_podcast_duration > 0:
                    for spk, duration in speaker_durations.items():
                        if (duration / total_podcast_duration) <= 0.01:
                            insert_speakers.add(spk)
                            
                if insert_speakers:
                    progress.console.print(f"   🧹 [dim]Found {len(insert_speakers)} short insert speakers (<1% duration). Labeling as inserts.[/dim]")

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
                                # Override as Insert if matched heuristic
                                if speaker in insert_speakers:
                                    speaker = "[Вставка]"
                        
                            # Fix Pyannote UNKNOWN & Noise brackets
                            skip_noise = global_config.get("processing", {}).get("skip_noise_and_music", False)
                            if skip_noise:
                                import re
                                is_pure_noise = bool(re.match(r'^[\(\[].*?[\)\]]$', text))
                                if is_pure_noise or speaker == "UNKNOWN" or speaker == "[UNKNOWN]":
                                    continue
                                
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

                t_transcribe_end = time.time()
                t_llm_start = time.time()

                podcast_metadata = {}
                # 4. LLM Smart Auto-Naming
                new_names = {}
                unidentified_speakers = []
                auto_naming = global_config.get("diarization", {}).get("auto_naming", False)
                if auto_naming and global_config.get("post_processing", {}).get("enabled", False):
                    progress.update(file_task, description=f"[magenta]{base_name}: LLM auto-naming & metadata")
                    # Extract roughly the first 15 mins (approx 200 lines) of transcript for context
                    lines_arr = raw_text.split("\n")
                    context_chunk = "\n".join(lines_arr[:200])
                    extracted_data = extract_podcast_metadata_with_llm(context_chunk, global_config)
                    
                    if extracted_data:
                        extracted_speakers = extracted_data.get("speakers", {})
                        podcast_metadata = extracted_data.get("metadata", {})
                        
                        # Update the tracking dict locally if confidence is high enough
                        updated_count: int = 0
                        for global_spk, data in extracted_speakers.items():
                            conf = data.get("confidence", 0)
                            real_name = data.get("name", "")
                            if conf >= 80:
                                new_names[global_spk] = real_name
                                if global_spk in config_db:
                                    config_db[global_spk] = real_name
                                    updated_count = updated_count + 1
                            else:
                                unidentified_speakers.append((global_spk, real_name, conf))
                        
                        if updated_count > 0:
                            # Save back to yaml
                            save_series_config(series_name, config_db)
                            progress.console.print(f"   🪄 [bold green]Learned {updated_count} new names from context![/]")

                # Perform text replacement in memory to swap GLOBAL_SPEAKER_X with real human names
                # We do this even if auto_naming is off, just to ensure manual names from config are applied 
                # before we send it to post-processing polish
                
                # Manual replacement from config
                for spk_key, human_name in config_db.items():
                    if spk_key.startswith("GLOBAL_SPEAKER_") and human_name != spk_key:
                         raw_text = raw_text.replace(spk_key, human_name)
                         
                # Log low confidence speakers
                for spk, guessed_name, conf in unidentified_speakers:
                    # Find first timestamp
                    first_ts = "Unknown"
                    for line in raw_text.split("\n"):
                        if spk in line:
                            import re
                            match = re.search(r'\[(.*?)\]', line)
                            if match:
                                first_ts = match.group(1)
                            break
                    progress.console.print(f"   ⚠️ [yellow]LLM guessed '{guessed_name}' for {spk} but confidence is low ({conf}%). First appearance: [{first_ts}].[/]")
                    
                # Rewrite the raw file with the real names
                with open(output_txt, "w", encoding="utf-8") as f:
                    f.write(raw_text)

                progress.update(file_task, description=f"[magenta]{base_name}: LLM post-process")
            
                # 5. Determine if it's a single speaker podcast
                lines_list = raw_text.strip().split("\n")
                speaker_counts = {}
                total_speech_lines = 0
                import re
                for line in lines_list:
                    match = re.match(r'\[\d{2}\.\d{2}\.\d{2}\] (.*?):', line)
                    if match:
                        spk = match.group(1).strip()
                        speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
                        total_speech_lines += 1
                
                is_single_speaker = False
                if total_speech_lines > 0:
                    max_spk_lines = max(speaker_counts.values())
                    if (max_spk_lines / total_speech_lines) >= 0.90 or chosen_model == "skip":
                         is_single_speaker = True
                         progress.console.print(f"   📝 [dim]Detected single-speaker majority ({max_spk_lines}/{total_speech_lines} lines). Using article formatting...[/]")

                # 6. LLM Post-Processing Polish
                if global_config.get("post_processing", {}).get("enabled", False):
                    formatted_text = run_post_processing(raw_text, global_config, is_single_speaker)
                
                    # Build Markdown Header
                    header_lines = []
                    show_name = podcast_metadata.get("show_name") or "Podcast"
                    ep_num = podcast_metadata.get("episode_number")
                    
                    if ep_num:
                        header_lines.append(f"# 🎙️ {show_name} - Выпуск {ep_num}")
                    else:
                        header_lines.append(f"# 🎙️ {show_name}")
                        
                    date_val = podcast_metadata.get("date")
                    if date_val:
                        header_lines.append(f"**📅 Дата:** {date_val}")
                        
                    topic_val = podcast_metadata.get("topic")
                    if topic_val:
                        header_lines.append(f"**💡 Тема:** {topic_val}")
                        
                    recognized_guests = [human_name for spk, human_name in config_db.items() if spk.startswith("GLOBAL_SPEAKER_") and human_name != spk]
                    if recognized_guests:
                        header_lines.append(f"**👥 Участники:** {', '.join(recognized_guests)}")
                        
                    header_lines.append("\n---")
                    header_str = "\n".join(header_lines)
                    
                    final_text = header_str + "\n\n" + formatted_text

                    final_output_path = os.path.join(output_dir, f"{base_name}_formatted.md")
                    with open(final_output_path, "w", encoding="utf-8") as f:
                        f.write(final_text)
                    
                    progress.console.print(f"   ✨ [bold green]LLM formatted transcript saved:[/] {final_output_path}")
                
                # Free Ollama memory so it doesn't overlap with Pyannote on the next iteration
                if global_config.get("post_processing", {}).get("enabled", False) and global_config.get("post_processing", {}).get("provider", "ollama").lower() == "ollama":
                    unload_ollama_model(global_config.get("post_processing", {}).get("model", "llama3.1"))
            
                t_llm_end = time.time()
                file_end_time = time.time()

                # --- 7. Write Execution Report ---
                def format_time(seconds: float) -> str:
                    "Converts seconds to 1h 23m 45s format cleanly"
                    if seconds < 0: return "0s"
                    m, s = divmod(int(seconds), 60)
                    h, m = divmod(m, 60)
                    if h > 0: return f"{h}h {m}m {s}s"
                    elif m > 0: return f"{m}m {s}s"
                    else: return f"{s}s"

                hardware_mode = "Hardware Accelerated" if perf_cfg.get("num_workers", 4) > 0 else "Low Memory Mode (<=16GB)"
                
                # Deduplicate speakers reliably directly from the saved file to ensure we don't count unknowns
                final_speakers = set()
                with open(output_txt, "r", encoding="utf-8") as f:
                    for line in f.readlines():
                        import re
                        m = re.match(r'\[\d{2}\.\d{2}\.\d{2}\] (.*?):', line)
                        if m: final_speakers.add(m.group(1).strip())
                
                normal_speakers = len([s for s in final_speakers if s != "[Вставка]"])
                insert_speakers_count = len(insert_speakers) if 'insert_speakers' in locals() else 0

                report_path = os.path.join(output_dir, "processing_report.md")
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                
                report_content = (
                    f"### 🎙️ {base_name}\n"
                    f"**📅 Date:** {now_str}\n"
                    f"**⏳ Audio Duration:** {format_time(total_podcast_duration if 'total_podcast_duration' in locals() else 0)} | "
                    f"**⏱️ Total Processing Time:** Custom {format_time(file_end_time - file_start_time)}\n\n"
                    f"- **⚙️ Hardware Mode:** {hardware_mode}\n"
                    f"- **👥 Speakers Found:** {len(final_speakers)} (Main: {normal_speakers}, Inserts: {insert_speakers_count})\n"
                    f"- **🔉 Conversion to WAV:** {format_time(t_convert_end - t_convert_start)}\n"
                    f"- **🗣️ Diarization** (`{chosen_model}`): {format_time(t_diarization_end - t_diarization_start)}\n"
                    f"- **📝 Transcription** (`{whisper_kwargs.get('path_or_hf_repo', 'Unknown')}`): {format_time(t_transcribe_end - t_transcribe_start)}\n"
                    f"- **🤖 Post-processing** (`{global_config.get('post_processing', {}).get('model', 'None')}`): {format_time(t_llm_end - t_llm_start)}\n"
                    "---\n\n"
                )

                with open(report_path, "a", encoding="utf-8") as rf:
                    rf.write(report_content)
                
                progress.console.print(f"   📊 [dim]Report appended to {report_path}[/dim]")

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