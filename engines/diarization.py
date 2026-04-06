import os
import torch
import time
import gc
from pyannote.audio import Pipeline
from schema import DiarizationConfig

class DiarizationEngine:
    """Wrapper for Pyannote Audio speaker diarization."""
    
    def __init__(self, config: DiarizationConfig, device=None):
        self.config = config
        self.device = device or (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
        self.current_model_path = None
        self.pipeline = None
        
    def load_pipeline(self, model_path: str):
        """Loads the diarization pipeline if not already loaded."""
        if self.pipeline and self.current_model_path == model_path:
            return self.pipeline
            
        print(f"   [dim]Loading diarization pipeline: {model_path}[/dim]")
        
        if model_path == "pyannote/speaker-diarization-2.1":
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
            pipeline = Pipeline.from_pretrained(model_path, token=os.environ.get("HF_TOKEN"))
            
        pipeline.to(self.device)
        self.pipeline = pipeline
        self.current_model_path = model_path
        return self.pipeline
        
    def diarize(self, audio_path: str, model_path: str):
        """Performs speaker diarization on the given audio file."""
        pipeline = self.load_pipeline(model_path)
        return pipeline(audio_path)
        
    def unload(self):
        """Explicitly unloads the pipeline and frees VRAM/RAM."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            self.current_model_path = None
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            time.sleep(1)
            print("   [dim]Diarization pipeline unloaded.[/dim]")
