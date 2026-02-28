# Podcasts Transcribe

A script for automatic podcast transcription with speaker diarization. Integrated with Whisper (MLX) and Pyannote Audio models, heavily optimized for Apple Silicon (MPS).

## Key Features

*   **⚡️ MLX Whisper Transcription:** Utilizes the powerful `mlx-whisper-large-v3-ru-podlodka` model running directly on Apple Silicon (MPS) for extremely fast processing.
*   **🗣 Speaker Recognition (Diarization):** Integrates the latest open-source `pyannote/speaker-diarization-3.1` model. The script intelligently merges Whisper's word-level timestamps with Pyannote's speaker segments to produce outputs like: `[00:01:23] SPEAKER_NAME: text...`.
*   **🧠 Cross-Episode Voice Memory:**
    *   **Auto-Series:** The script parses series prefixes from filenames (e.g., `rt_podcast_10.mp3` and `rt_podcast_11.mp3` will share the `rt_podcast` speaker database).
    *   **Memorization:** Digital voice embeddings are saved in a local database. The program automatically recognizes hosts in new episodes.
    *   **Voice Evolution (EMA):** The script blends 10% of the new voice into the reference database upon each match. This maintains accuracy even if a speaker changes their microphone, voice ages, or has a cold.
    *   **Auto-Merge Duplicates:** If the neural network makes a mistake and creates a duplicate profile, simply assign it the same human name in the local podcast config. On the next run, the script will automatically average their voices and delete the duplicate!
*   **💾 Audio Caching:** Built-in LRU cache for converted `wav` files prevents redundant `ffmpeg` conversions on repeated runs.

## Project Structure

```text
podcasts_transcribe/
├── input/                  <-- Put your MP3, M4A, FLAC, WAV here
├── output/                 <-- Get your finished TXT transcripts here
├── models/                 <-- Downloaded ML models folder
├── speakers/               <-- Voice databases (vectors + YAML configs)
├── .cache/                 <-- Audio cache folder
├── config.yaml             <-- Global settings (thresholds, cache size)
└── transcribe.py           <-- Main launcher script
```

## Installation

1. **System Dependencies:** Ensure the `ffmpeg` utility is installed. On Mac: `brew install ffmpeg`.
2. **Python:** Create a virtual environment (Python 3.9+) and install packages from `requirements.txt`:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Pyannote Models Access Token:**
    * Pyannote Audio models are open-source but require accepting terms on HuggingFace.
    * Visit [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and click "Agree".
    * Visit [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and click "Agree".
    * Create a User Access Token in your HuggingFace account settings.
4. Create a `.env` file in the project root and add the token:
    ```env
    # .env
    HF_TOKEN="your_hf_token_here"
    ```

## Usage (Quick Start)

1. Drop any media files into the `input/` folder.
2. Run the script:
   ```bash
   python transcribe.py
   ```
3. Pick up the ready text document from the `output/` folder!

*(Optional) For Testing:* To process only the first 2 minutes of the audio:
```bash
python transcribe.py --test 120
```

## How to rename `GLOBAL_SPEAKER` to a real name?

1. Once the `transcribe.py` run for a podcast finishes, go to the `speakers/your_series_name/` folder.
2. Open the `config.yaml` file.
3. Assign the real human names:
    ```yaml
    GLOBAL_SPEAKER_1: John Doe
    GLOBAL_SPEAKER_2: Alice
    ```
4. In future podcast episodes, the program will automatically substitute "John Doe" and "Alice".
5. *Recognition Errors?* If a new podcast introduces `GLOBAL_SPEAKER_5`, and it was actually John with a muffled microphone — just rename `GLOBAL_SPEAKER_5: John Doe`. The program will automatically merge their voices into a single database!

## Global Settings (config.yaml)

The root folder contains `config.yaml` with detailed English comments. In it, you can tweak:
- Whisper hallucination defenses.
- Pyannote voice matching sensitivity (0.0 — 1.0).
- Automatic duplicate merging.
- Maximum megabyte limits and TTL for the audio cache folder.

