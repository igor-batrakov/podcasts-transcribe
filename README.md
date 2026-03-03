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

