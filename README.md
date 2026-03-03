# Podcasts Transcribe

*[English Version Below](#english-version)*

Скрипт для автоматической расшифровки подкастов (транскрибации) с разделением по спикерам (диаризацией). Интегрирован с моделями Whisper (MLX), Pyannote Audio и локальными LLM (Ollama), глубоко оптимизирован для Apple Silicon.

## Установка

1. **Зависимости:** Установите утилиты: `brew install ffmpeg ollama`
2. **Python:** Создайте окружение (Python 3.9+) и установите пакеты:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Ключи доступа Pyannote:**
    * Модели Pyannote Audio требуют согласия на HuggingFace.
    * Перейдите на [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) и [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1), нажмите "Agree".
    * Для быстрых моделей и VAD: также согласитесь с `speaker-diarization-2.1`, `segmentation` и `voice-activity-detection`.
4. Создайте токен в настройках HuggingFace и добавьте его в `.env`:
    ```env
    # .env
    HF_TOKEN="твой_hf_токен_здесь"
    ```

## Быстрый старт

1. Откройте терминал и скачайте LLM модель для постобработки: `ollama run qwen2.5:3b` (затем нажмите Ctrl+D для выхода).
2. Закиньте медиа-файлы (MP3/M4A/WAV) в папку `input/`.
3. Запустите скрипт: `python transcribe.py` (для тестов используйте `python transcribe.py --test 120` для первых 2 минут).
4. Готовые файлы `.txt` и форматированные `.md` появятся в папке `output/`.

---

## Ключевые фичи и Настройки (config.yaml)

*   **⚡️ MLX Whisper Транскрибация:** Использует мощную модель `mlx-whisper-large-v3-ru-podlodka`, работающую напрямую на GPU Mac для экстремально быстрой работы. Для других языков: замените модель в `config.yaml` на официальную `mlx-community/whisper-large-v3-mlx`.
*   **🗣 Кросс-эпизодная память (Pyannote):** Скрипт учится узнавать ведущих. Имена спикеров собираются в `speakers/`. Вручную переименуйте `GLOBAL_SPEAKER_1`, и программа назовет его так же в следующих выпусках (и сольет дубликаты голосов).
*   **🧠 Умная Постобработка (LLM):** 
    * Автоматическая пунктуация и абзацы через локальную Ollama (Опционально: OpenAI/Anthropic). Вы можете поменять используемую модель (по умолчанию `qwen2.5:3b`) в `config.yaml` -> `post_processing.model`.
    * **Умный Заголовок:** Нейросеть сама анализирует начало разговора, доставая название шоу, тему и реальные имена участников, собирая красивый Markdown-заголовок.
    * **Определение муз. вставок:** Скрипт по длительности и контексту автоматически вычисляет "джинглы", скрывая их из основной базы спикеров и помечая курсивом.
*   **🏎️ Экстремальная оптимизация памяти:**
    * **Авто-лимиты (Mac 16GB):** Программа автоматически отключает мультипроцессинг и снижает пакетную нагрузку, если у вас 16 ГБ ОЗУ или меньше. Ошибок нехватки памяти (OOM) не будет!
    * **Анализ тишины (VAD):** Экономит колоссальное количество времени, вырезая тишину до запуска тяжелой диаризации.
    * **Глобальный конфигуратор:** При старте программа спрашивает, какие настройки применить (Fast/Accurate/Skip), и затем автономно обрабатывает всю папку.
*   **📊 Прозрачность и Кэширование:** Для каждого аудиофайла дописывается отчет с таймингами обработки в файл `output/processing_report.md`. Вырезанные `.wav` файлы кэшируются для ускорения перезапусков.

## Лицензия

Распространяется по лицензии **MIT**. Подробности в файле [LICENSE](LICENSE).

---
---

<a name="english-version"></a>

# Podcasts Transcribe (English)

A heavily-optimized pipeline for automatic podcast transcription, Pyannote-based speaker diarization, and LLM formatting. Built flawlessly for Apple Silicon (MPS).

## Installation

1. **System Dependencies:** Install required binaries: `brew install ffmpeg ollama`
2. **Python:** Setup your environment (Python 3.9+) and install packages:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Pyannote Models Access Token:**
    * Visit [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) on HuggingFace and click "Agree" on their terms.
    * (Optional) For fast models and VAD, click agree on `speaker-diarization-2.1`, `segmentation`, and `voice-activity-detection`.
4. Create a User Access Token in HF settings and save it in a `.env` file:
    ```env
    # .env
    HF_TOKEN="your_hf_token_here"
    ```

## Usage (Quick Start)

1. Open your terminal and pull the local LLM model for post-processing: `ollama run qwen2.5:3b` (then press Ctrl+D to exit).
2. Drop media files into the `input/` folder.
3. Run the script: `python transcribe.py` (append `--test 120` to transcribe only the first 2 minutes).
4. Pick up your generated transcripts and `.md` reports from the `output/` folder!

---

## Key Features & Settings (`config.yaml`)

*   **⚡️ MLX Whisper Transcription:** Utilizes the Russian-optimized `mlx-whisper-large-v3-ru-podlodka` running directly on Apple Silicon GPUs. Switch to `mlx-community/whisper-large-v3-mlx` in `config.yaml` for 99+ languages global support.
*   **🗣 Cross-Episode Voice Memory:** Remembers host voices across episodes. Rename `GLOBAL_SPEAKER_1` to a real name in the `speakers/` folder, and it will cascade to all future runs and auto-merge duplicate voices.
*   **🧠 Intelligent Post-Processing (LLM):** 
    * Punctuation and paragraph formatting via local Ollama (OpenAI/Anthropic also supported). You can change the default model (`qwen2.5:3b`) in `config.yaml` -> `post_processing.model`.
    * **Smart Markdown Headers:** The LLM auto-extracts podcast names, episode numbers, topics, and speaker introductions to build a beautiful Markdown header document snippet.
    * **Jingle Detection:** Short audio inserts and musical jingles are detected by length and context, formatted beautifully, and kept cleanly out of the global speaker database.
*   **🏎️ Adaptive Hardware Optimization:**
    * **Low-Memory Safety (16GB Macs):** Automatically disables heavy multiprocessing and lowers batch sizes on Macs with 16GB RAM or less to prevent OOM freezes.
    * **Voice Activity Detection (VAD):** Cuts out entire chunks of silence before intensive processing stages to save massive amounts of GPU time.
    * **Interactive Upfront CLI:** The script asks you once how you want to handle diarization for new files (Fast / Accurate / Skip), then autonomously batch processes everything.
*   **📊 Transparent Metrics:** Automatically tracks step-by-step execution durations and appends them to an efficiency log (`output/processing_report.md`). Extracted `.wav` audio is smartly LRU cached.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more information.
---
---

<a name="english-version"></a>

# Podcasts Transcribe (English)

A script for automatic podcast transcription with speaker diarization. Integrated with Whisper (MLX) and Pyannote Audio models, heavily optimized for Apple Silicon (MPS).

## Key Features

*   **⚡️ MLX Whisper Transcription:** Utilizes the powerful `mlx-whisper-large-v3-ru-podlodka` model running directly on Apple Silicon (MPS) for extremely fast processing.
*   **🗣 Speaker Recognition (Diarization):** Integrates the latest open-source `pyannote/speaker-diarization-3.1` model. The script intelligently merges Whisper's word-level timestamps with Pyannote's speaker segments to produce outputs like: `[00:01:23] SPEAKER_NAME: text...`.
*   **🧠 Cross-Episode Voice Memory:**
    *   **Auto-Series:** The script parses series prefixes from filenames (e.g., `rt_podcast_10.mp3` and `rt_podcast_11.mp3` will share the `rt_podcast` speaker database).
    *   **Memorization:** Digital voice embeddings are saved in a local database. The program automatically recognizes hosts in new episodes.
    *   **Voice Evolution (EMA):** The script blends 10% of the new voice into the reference database upon each match. This maintains accuracy even if a speaker changes their microphone, voice ages, or has a cold.
    *   **Auto-Merge Duplicates:** If the neural network makes a mistake and creates a duplicate profile, simply assign it the same human name in the local podcast config. On the next run, the script will automatically average their voices and delete the duplicate!
    *   **🪄 Smart Auto-Naming (LLM):** If enabled, the script parses the transcription context of the first 15 minutes of audio, sending it to an LLM (Ollama, OpenAI, or Anthropic) to extract the real names from the introductions. It seamlessly substitutes `GLOBAL_SPEAKER_1` with "John" automatically in the output file and saves it back to the database.
*   **🏎️ Extreme Optimization (Performance):**
    *   **Upfront Setup (Interactive CLI):** The script scans all media upfront and asks which Diarization model to use for new series (Skip, Fast, Accurate) _before_ starting heavy processing. Then it runs 100% autonomously.
    *   **Voice Activity Detection (VAD):** Enable `vad_enabled` in `config.yaml` to run audio through a lightweight speech-detector first, completely clipping out silence before the heavy Diarization model kicks in. Saves massive GPU time.
    *   **Hardware Acceleration (Batching):** Process audio in parallel batches. Tune `batch_size` and `num_workers` in `config.yaml` to max out your multi-core CPU and GPU.
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
    * If using the Fast Model, visit [speaker-diarization-2.1](https://huggingface.co/pyannote/speaker-diarization-2.1) and [segmentation](https://huggingface.co/pyannote/segmentation) to agree.
    * (Optional) For VAD: visit [voice-activity-detection](https://huggingface.co/pyannote/voice-activity-detection) and click "Agree".
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

## Supported Languages & Models

By default, the script downloads and uses the `mlx-whisper-large-v3-ru-podlodka` model, which is fine-tuned specifically for podcasts. This model:
*   **Perfectly recognizes Russian,** including heavy slang, swearing, anglicisms, and complex interruptions.
*   **Excellently recognizes English.**

**How to transcribe other languages (Spanish, German, Japanese, etc.)?**
The underlying Whisper architecture supports **99 languages**. To enable auto-detection for any language:
1. Open `config.yaml`.
2. Delete the specific Russian model and replace it with the official multi-language community model:
   ```yaml
   path_or_hf_repo: mlx-community/whisper-large-v3-mlx
   ```
3. Change the hardcoded language setting to `null` (auto-detect):
   ```yaml
   language: null
   ```

## Global Settings (config.yaml)

The root folder contains `config.yaml` with detailed English comments. In it, you can tweak:
- **Transcription (`language` & `path_or_hf_repo`)**: Switch models and languages (default is strictly `ru` for speed).
- **Hallucination Defenses**: Adjust `condition_on_previous_text` or `compression_ratio_threshold` to stop Whisper from looping on weird audio parts.
- **Pyannote Voice Matching**: Change `similarity_threshold` (0.0 — 1.0) to make the script more strict or lenient when identifying returning guests.
- **Voice Evolution (`ema_alpha`)**: Controls how fast the script adapts to a speaker's voice aging.
- **Automatic Duplicate Merging**: Enabled by default (`auto_merge_duplicates: true`).
- **Cache Limits (`max_size_mb` / `max_age_days`)**: Set the maximum megabyte limits and TTL (Time-To-Live) for the temporary audio cache folder to save disk space.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more information.
