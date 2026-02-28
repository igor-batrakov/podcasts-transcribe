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

load_dotenv()

def process_podcasts(time_limit=None):
    # Find active audio extensions in the input folder
    supported_extensions = ("*.mp3", "*.m4a", "*.wav", "*.flac")
    audio_files = []
    
    input_dir = "input"
    output_dir = "output"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    for ext in supported_extensions:
        audio_files.extend(glob.glob(os.path.join(input_dir, ext)))
        
    if not audio_files:
        print(f"No audio files found in '{input_dir}' directory.")
        return

    print(f"Audio files found to process: {len(audio_files)}")
    
    # Load master configuration
    global_config = load_global_config()
    whisper_kwargs = global_config["transcription"]
    diarization_cfg = global_config["diarization"]
    cache_cfg = global_config["cache"]
    
    similarity_threshold = diarization_cfg.get("similarity_threshold", 0.35)
    ema_alpha = diarization_cfg.get("ema_alpha", 0.1)
    auto_merge_duplicates = diarization_cfg.get("auto_merge_duplicates", True)
    
    print("Settings loaded from config.yaml")

    # Load speaker recognition model
    print("Loading speaker diarization model...")
    try:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        diarization_pipeline = Pipeline.from_pretrained(
            diarization_cfg.get("model", "pyannote/speaker-diarization-3.1"),
            token=os.environ.get("HF_TOKEN")
        )
        diarization_pipeline.to(device)
    except Exception as e:
        print(f"Error loading diarization model: {e}")
        print("💡 Make sure you have accepted the pyannote model terms:")
        print("   1. https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("   2. https://huggingface.co/pyannote/segmentation-3.0")
        print("   And verify that your HF_TOKEN is specified in the .env file.")
        return

    for audio_path in audio_files:
        # Determine podcast series from filename
        series_name = get_series_name(os.path.basename(audio_path))
        
        # Get base name without extension and path
        base_name, _ = os.path.splitext(os.path.basename(audio_path))
        output_txt = os.path.join(output_dir, f"{base_name}.txt")
        
        if os.path.exists(output_txt):
            # Interactive menu if file exists
            print(f"\n[WARNING] Output file '{output_txt}' already exists.")
            while True:
                choice = input("Action? [O]verwrite, [R]ename, [S]kip file, [Q]uit script: ").strip().upper()
                if choice == 'O':
                    print("File will be overwritten.")
                    break
                elif choice == 'R':
                    output_txt = get_unique_filename(output_txt)
                    print(f"New filename: '{output_txt}'.")
                    break
                elif choice == 'S':
                    print("Skipping this file.")
                    break
                elif choice == 'Q':
                    print("Script stopped by user.")
                    return
                else:
                    print("Invalid choice. Please enter O, R, S, or Q.")
            if choice == 'S':
                continue
            
        print(f"\n>>> Processing: {audio_path} (Series: {series_name})")
        
        # Load saved speaker profiles for this specific series
        config_db = load_series_config(series_name)
        embeddings_db = load_series_embeddings(series_name)
        
        # Automated duplicate speaker merging
        if auto_merge_duplicates:
            if merge_duplicate_speakers(series_name, config_db, embeddings_db):
                save_series_config(series_name, config_db)
                save_series_embeddings(series_name, embeddings_db)
                
        try:
            print("1. Diarization (who speaks when)...")
            temp_wav = convert_to_wav(audio_path, time_limit, cache_cfg=cache_cfg)
            
            # File is processed directly from the persistent cache
            diarization = diarization_pipeline(temp_wav)
            
            print("   Analyzing and matching voices with local database...")
            speaker_mapping = get_global_speaker_mapping(
                diarization, series_name, config_db, embeddings_db, 
                similarity_threshold, ema_alpha
            )
            
            # Save the actively updated speaker profiles back to disk immediately
            save_series_config(series_name, config_db)
            save_series_embeddings(series_name, embeddings_db)

            print("   Unloading Pyannote neural weights from RAM before Whisper...")
            
            print("2. Transcription (what they are saying)...")
            
            # Whisper parses the 16kHz cached WAV (faster than MP3 inference)
            transcribe_source = temp_wav
            
            # Transcription via MLX on Apple Silicon GPU
            result = mlx_whisper.transcribe(
                transcribe_source,
                verbose=False, # Hardcoded false to prevent CLI bloat
                **whisper_kwargs
            )

            print("3. Merging text chunks with global speakers...")
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
            
            print(f"   [DONE] Result saved to {output_txt}")
            
        except Exception as e:
            print(f"   [ERROR] Failed to process file {audio_path}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Podcast Transcribe. Converts audio to text partitioned by global speakers.")
    parser.add_argument("--test", type=int, help="Test run mode. Process only N seconds of the audio. Example: --test 60")
    args = parser.add_argument("--test-run", action='store_true', help="Hidden test flag wrapper") # For IDE silence
    
    args, unknown = parser.parse_known_args()
    
    print("--- STARTING PODCAST TRANSCRIBER ---")
    process_podcasts(time_limit=args.test)
    print("--- ALL FILES PROCESSED ---")