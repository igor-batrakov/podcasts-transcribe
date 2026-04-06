from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class WhisperConfig:
    path_or_hf_repo: str
    language: str = "ru"
    word_timestamps: bool = True
    condition_on_previous_text: bool = False
    compression_ratio_threshold: float = 2.4
    temperature: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    no_speech_threshold: float = 0.6
    logprob_threshold: float = -1.0

@dataclass
class DiarizationConfig:
    model: str = "pyannote/speaker-diarization-3.1"
    similarity_threshold: float = 0.35
    ema_alpha: float = 0.1
    auto_merge_duplicates: bool = True
    auto_naming: bool = True

@dataclass
class PerformanceConfig:
    vad_enabled: bool = False
    batch_size: int = 8
    num_workers: int = 4

@dataclass
class CacheConfig:
    max_size_mb: int = 2000
    max_age_days: int = 2

@dataclass
class ProcessingConfig:
    skip_noise_and_music: bool = True

@dataclass
class PostProcessingConfig:
    enabled: bool = True
    provider: str = "ollama"
    model: str = "qwen2.5:3b"
    prompt_multi_speaker: str = ""
    prompt_single_speaker: str = ""
    chunk_size_lines: int = 200
    overlap_lines: int = 10

@dataclass
class PathsConfig:
    input_dir: str = "input"
    output_dir: str = "output"
    speakers_dir: str = "speakers"
    cache_dir: str = ".cache/audio"

@dataclass
class GlobalConfig:
    transcription: WhisperConfig
    diarization: DiarizationConfig
    performance: PerformanceConfig
    cache: CacheConfig
    processing: ProcessingConfig
    post_processing: PostProcessingConfig
    paths: PathsConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GlobalConfig":
        return cls(
            transcription=WhisperConfig(**data.get("transcription", {})),
            diarization=DiarizationConfig(**data.get("diarization", {})),
            performance=PerformanceConfig(**data.get("performance", {})),
            cache=CacheConfig(**data.get("cache", {})),
            processing=ProcessingConfig(**data.get("processing", {})),
            post_processing=PostProcessingConfig(**data.get("post_processing", {})),
            paths=PathsConfig(**data.get("paths", {})),
        )
