import mlx_whisper
import os
import sys
from dataclasses import asdict
from schema import WhisperConfig

class TranscriptionEngine:
    """Wrapper for MLX Whisper transcription."""
    
    def __init__(self, config: WhisperConfig):
        self.config = config
        
    def transcribe(self, audio_path: str) -> dict:
        """Transcribes the given audio file using MLX Whisper."""
        # Redirect stderr to /dev/null to suppress MLX's internal TQDM output
        _original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        try:
            whisper_kwargs = asdict(self.config)
            result = mlx_whisper.transcribe(
                audio_path,
                verbose=False,
                **whisper_kwargs
            )
            return result
        finally:
            sys.stderr.close()
            sys.stderr = _original_stderr
            
    def cleanup(self):
        """Clears MLX/Metal cache to free VRAM."""
        import mlx.core as mx
        mx.metal.clear_cache()
