import pytest
import os
import yaml
from config_loader import load_global_config

def test_load_global_config_separates_kwargs(tmp_path, mocker):
    """Test that config_loader correctly separates transcription kwargs from other sections."""
    # Create a dummy config
    dummy_config = {
        "path_or_hf_repo": "models/fake-model",
        "language": "ru",
        "word_timestamps": True,
        "diarization": {"similarity_threshold": 0.5},
        "cache": {"max_age_days": 2},
        "post_processing": {"enabled": True, "provider": "ollama"}
    }
    
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(dummy_config, f)
        
    # Mock os.path.exists and open to use our dummy config
    # We'll just temporarily change directory to tmp_path
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        config = load_global_config()
        
        # Verify transcription kwargs only contain root keys
        assert config["transcription"]["path_or_hf_repo"] == "models/fake-model"
        assert config["transcription"]["language"] == "ru"
        assert "enabled" not in config["transcription"]
        assert "similarity_threshold" not in config["transcription"]
        
        # Verify other sections
        assert config["post_processing"]["enabled"] is True
        assert config["diarization"]["similarity_threshold"] == 0.5
        
    finally:
        os.chdir(original_cwd)
