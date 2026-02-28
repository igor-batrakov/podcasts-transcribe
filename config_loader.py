import os
import yaml

def load_global_config():
    """Loads global settings from config.yaml"""
    config_path = "config.yaml"
    default_config = {
        "transcription": {"path_or_hf_repo": "mlx-whisper-large-v3-ru-podlodka", "word_timestamps": True},
        "diarization": {"similarity_threshold": 0.35, "ema_alpha": 0.1, "model": "pyannote/speaker-diarization-3.1"},
        "cache": {"max_size_mb": 5000, "max_age_days": 7},
        "paths": {"input_dir": "input", "output_dir": "output", "speakers_dir": "speakers", "cache_dir": ".cache/audio"}
    }
    
    if not os.path.exists(config_path):
        print(f"[WARNING] Settings file {config_path} not found! Using default parameters.")
        return default_config
        
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
            
        if not yaml_config:
            return default_config
            
        # Extract Whisper parameters (all root keys except specific sections)
        whisper_kwargs = {}
        for key, value in yaml_config.items():
            if key not in ["diarization", "cache", "post_processing", "paths"]:
                whisper_kwargs[key] = value
                        
        return {
            "transcription": whisper_kwargs or default_config["transcription"],
            "diarization": yaml_config.get("diarization", default_config["diarization"]),
            "cache": yaml_config.get("cache", default_config["cache"]),
            "post_processing": yaml_config.get("post_processing", {}),
            "paths": yaml_config.get("paths", default_config["paths"])
        }
    except Exception as e:
        print(f"[ERROR] Failed to load {config_path}: {e}")
        return default_config
