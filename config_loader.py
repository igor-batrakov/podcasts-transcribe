import os
import yaml
from schema import GlobalConfig

def load_global_config() -> GlobalConfig:
    """Loads global settings from config.yaml and returns a GlobalConfig object."""
    config_path = "config.yaml"
    
    # Default fallback data if file is missing or broken
    default_data = {
        "transcription": {"path_or_hf_repo": "models/mlx-whisper-large-v3-ru-podlodka", "language": "ru"},
        "diarization": {"model": "pyannote/speaker-diarization-3.1"},
        "cache": {"max_size_mb": 2000, "max_age_days": 2},
        "paths": {"input_dir": "input", "output_dir": "output", "speakers_dir": "speakers", "cache_dir": ".cache/audio"},
        "performance": {"num_workers": 4, "batch_size": 8},
        "processing": {"skip_noise_and_music": True},
        "post_processing": {"enabled": True}
    }
    
    if not os.path.exists(config_path):
        print(f"[WARNING] Settings file {config_path} not found! Using default parameters.")
        return GlobalConfig.from_dict(default_data)
        
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
            
        if not yaml_config:
            return GlobalConfig.from_dict(default_data)
            
        # Extract Whisper parameters (all root keys except specific sections)
        whisper_kwargs = {}
        special_sections = ["diarization", "cache", "post_processing", "paths", "performance", "processing", "speaker_identification"]
        for key, value in yaml_config.items():
            if key not in special_sections:
                whisper_kwargs[key] = value
        
        # Merge into a structured dictionary
        config_dict = {
            "transcription": whisper_kwargs,
            "diarization": yaml_config.get("diarization", {}),
            "cache": yaml_config.get("cache", {}),
            "post_processing": yaml_config.get("post_processing", {}),
            "paths": yaml_config.get("paths", {}),
            "processing": yaml_config.get("processing", {}),
            "performance": yaml_config.get("performance", {})
        }
        
        return GlobalConfig.from_dict(config_dict)
        
    except Exception as e:
        print(f"[ERROR] Failed to load {config_path}: {e}")
        return GlobalConfig.from_dict(default_data)
