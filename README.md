# Podcasts Transcribe

*[English Version Below](#english-version)*

Скрипт для автоматической расшифровки подкастов (транскрибации) с разделением по спикерам (диаризацией). Интегрирован с моделями Whisper (MLX) и Pyannote Audio, оптимизирован для работы на Apple Silicon.

## Основные функции

*   **⚡️ MLX Whisper Транскрибация:** Использует мощную модель `mlx-whisper-large-v3-ru-podlodka`, работающую напрямую на графическом ускорителе Apple Silicon (MPS) для экстремально быстрой работы.
*   **🗣 Распознавание спикеров (Diarization):** Интегрирована новейшая open-source модель `pyannote/speaker-diarization-3.1`. Скрипт интеллектуально склеивает пословные таймкоды Whisper с таймкодами Pyannote, чтобы выдавать текст вида: `[00:01:23] ИМЯ_СПИКЕРА: текст...`.
*   **🧠 Кросс-эпизодная память на голоса:**
    *   **Авто-серии:** Скрипт парсит префиксы из имен файлов (например, в файлах `rt_podcast_10.mp3` и `rt_podcast_11.mp3` будет выделена общая база спикеров серии `rt_podcast`).
    *   **Запоминание:** Цифровые слепки голосов (embeddings) записываются в локальную базу данных. Программа сама узнает ведущих в новых выпусках.
    *   **Эволюция голоса (EMA):** Скрипт подмешивает 10% нового голоса в эталонную базу при каждом совпадении. Это спасает качество при смене микрофона или взрослении диктора.
    *   **Авто-слияние:** Если нейросеть ошиблась и создала дубликат профиля, просто задайте ему такое же человеческое имя в локальном конфиге подкаста. При следующем запуске скрипт автоматически усреднит их голоса и удалит дубликат!
*   **🏎️ Экстремальная оптимизация (Performance):**
    *   **Предварительный опрос (Interactive CLI):** Скрипт анализирует все медиафайлы в папке и _до начала тяжелой обработки_ спрашивает у вас какую модель Диаризации использовать для новой серии (Skip, Fast, Accurate). После этого он уходит в полностью автономный режим.
    *   **Voice Activity Detection (VAD):** Можно включить `vad_enabled` в `config.yaml`, чтобы скрипт прогнал аудио через легкую нейросеть-детектор речи и полностью вырезал тишину до того, как скормить аудио тяжелому алгоритму Диаризации. Это экономит колоссальное количество GPU.
    *   **Аппаратное ускорение (Batching):** Поддерживается обработка аудио батчами (пакетами). Вы можете настроить `batch_size` и `num_workers` в `config.yaml` для максимальной загрузки многоядерных процессоров и видеокарт.
*   **💾 Кеширование аудио:** Встроенный LRU-кэш для сконвертированных `wav` файлов спасает от повторных прогонов `ffmpeg`.

## Структура проекта

```text
podcasts_transcribe/
├── input/                  <-- Кладите сюда MP3, M4A, FLAC, WAV
├── output/                 <-- Забирайте отсюда готовые TXT
├── models/                 <-- Папка скачанных нейросетей
├── speakers/               <-- Базы голосов (вектора + YAML-файлы имен)
├── .cache/                 <-- Папка кэша аудио
├── config.yaml             <-- Главные настройки (пороги, размеры кэша)
└── transcribe.py           <-- Скрипт запуска
```

## Установка

1. **Системные зависимости:** Убедитесь, что установлена утилита `ffmpeg`. На Mac: `brew install ffmpeg`.
2. **Python:** Создайте виртуальное окружение (Python 3.9+) и установите пакеты из `requirements.txt`:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Ключ доступа к моделям Pyannote:**
    * Модели Pyannote Audio являются открытыми, но требуют согласия с правилами на HuggingFace.
    * Перейдите на [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) и нажмите "Agree".
    * Перейдите на [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) и нажмите "Agree".
    * Создайте HF Token в настройках аккаунта HuggingFace.
4. Создайте файл `.env` в корне проекта и впишите туда токен:
    ```env
    # .env
    HF_TOKEN="твой_hf_токен_здесь"
    ```

## Использование (Быстрый старт)

1. Закиньте любые медиа-файлы в папку `input/`.
2. Запустите скрипт:
   ```bash
   python transcribe.py
   ```
3. Заберите готовый текстовый документ из папки `output/`!

*(Опционально) Для тестов:* Чтобы прогнать только первые 2 минуты аудио:
```bash
python transcribe.py --test 120
```

## Как переименовать `GLOBAL_SPEAKER` в реальное имя?

1. Когда прогон `transcribe.py` для подкаста завершится, зайдите в папку `speakers/название_вашей_серии/`.
2. Откройте файл `config.yaml`.
3. Присвойте реальные имена:
    ```yaml
    GLOBAL_SPEAKER_1: Игорь
    GLOBAL_SPEAKER_2: Umputun
    ```
4. В следующих подкастах программа автоматически будет подставлять "Игорь" и "Umputun". 
5. *Ошибки распознавания?* Если в новом подкасте появился `GLOBAL_SPEAKER_5`, и это на самом деле был Игорь с глухим микрофоном — просто переименуйте `GLOBAL_SPEAKER_5: Игорь`. Программа автоматически сольет их голоса в единую базу!

## Поддерживаемые языки и модели

По умолчанию скрипт скачивает и использует модель `mlx-whisper-large-v3-ru-podlodka`, которая специально дообучена под подкасты. Эта модель:
*   **Идеально распознает русский язык,** включая сильный мат, сленг, англицизмы и сложные перебивания.
*   **Отлично распознает английский язык.**

**Как транскрибировать другие языки (Испанский, Японский и тд.)?**
Базовая архитектура Whisper поддерживает **99 языков**. Чтобы включить автоопределение для любого языка:
1. Откройте `config.yaml`.
2. Удалите специальную русскую модель и впишите официальную мультиязычную от сообщества MLX:
   ```yaml
   path_or_hf_repo: mlx-community/whisper-large-v3-mlx
   ```
3. Измените жестко заданный язык на `null` (автоопределение):
   ```yaml
   language: null
   ```

## Глобальные настройки (config.yaml)

В корне лежит `config.yaml` с комментариями на английском языке. Там можно настроить:
- **Транскрибацию (`language` & `path_or_hf_repo`)**: Переключать модели и языки (по умолчанию строго `ru` для скорости).
- **Защиту от галлюцинаций**: Меняйте `condition_on_previous_text` или `compression_ratio_threshold`, чтобы Whisper не зацикливался на шумах.
- **Чувствительность распознавания голосов Pyannote**: Меняйте `similarity_threshold` (0.0 — 1.0), чтобы скрипт строже или мягче узнавал гостей из прошлых выпусков.
- **Эволюцию голоса (`ema_alpha`)**: Насколько быстро скрипт адаптируется к изменениям голоса спикера.
- **Автоматическое слияние дубликатов**: Включено по умолчанию (`auto_merge_duplicates: true`). 
- **Лимиты кэша (`max_size_mb` / `max_age_days`)**: Задайте сколько мегабайт и дней могут жить сконвертированные аудиофайлы во временной папке, чтобы не забивать диск.

## Лицензия

Этот проект распространяется по лицензии **MIT**. Подробности в файле [LICENSE](LICENSE).
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
