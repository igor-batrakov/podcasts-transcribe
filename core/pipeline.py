import os
import time
import datetime
import gc
import torch
import re
from typing import List, Tuple, Dict, Optional, Set
from schema import GlobalConfig
from audio_converter import convert_to_wav
from speaker_manager import (
    load_series_config, 
    load_series_embeddings, 
    save_series_config, 
    save_series_embeddings, 
    get_global_speaker_mapping, 
    get_speaker, 
    merge_duplicate_speakers
)
from post_processing import run_post_processing, extract_podcast_metadata_with_llm, unload_ollama_model
from engines.transcription import TranscriptionEngine
from engines.diarization import DiarizationEngine

class TranscriptionPipeline:
    """Orchestrates the entire transcription and diarization process."""
    
    def __init__(self, config: GlobalConfig, progress_callback=None):
        self.config = config
        self.progress_callback = progress_callback
        self.transcription_engine = TranscriptionEngine(config.transcription)
        self.diarization_engine = DiarizationEngine(config.diarization)
        
    def process_file(self, audio_path: str, series_name: str, chosen_model: str, base_name: str, output_txt: str, time_limit: Optional[int] = None, executor=None) -> bool:
        """Processes a single audio file. If executor is provided, LLM part runs in background."""
        try:
            # --- PART 1: HEAVY GPU/CPU PROCESSING (Sequential) ---
            # 1. Setup & Initial Cleanup
            config_db = load_series_config(series_name)
            embeddings_db = load_series_embeddings(series_name)
            
            if self.config.diarization.auto_merge_duplicates:
                if merge_duplicate_speakers(series_name, config_db, embeddings_db):
                    save_series_config(series_name, config_db)
                    save_series_embeddings(series_name, embeddings_db)
            
            file_start_time = time.time()
            
            # 2. Audio Conversion
            t_convert_start = time.time()
            if self.progress_callback: self.progress_callback(f"{base_name}: Converting to WAV")
            temp_wav = convert_to_wav(audio_path, time_limit, cache_cfg=self.config.cache)
            t_convert_end = time.time()
            
            # 3. Diarization
            t_diarization_start = time.time()
            diarization = None
            speaker_mapping = {"SPEAKER_00": "GLOBAL_SPEAKER_1"}
            
            if chosen_model != "skip":
                if self.progress_callback: self.progress_callback(f"{base_name}: Diarization")
                
                # VAD Pre-pass
                active_speech = None
                if self.config.performance.vad_enabled:
                    if self.progress_callback: self.progress_callback(f"{base_name}: VAD")
                    vad_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", token=os.environ.get("HF_TOKEN"))
                    vad_pipeline.to(self.diarization_engine.device)
                    active_speech = vad_pipeline(temp_wav)
                    del vad_pipeline
                    gc.collect()
                    if torch.backends.mps.is_available(): torch.mps.empty_cache()

                # Diarization Pass
                diarization = self.diarization_engine.diarize(temp_wav, chosen_model)
                
                speaker_mapping = get_global_speaker_mapping(
                    diarization, series_name, config_db, embeddings_db, 
                    self.config.diarization.similarity_threshold, 
                    self.config.diarization.ema_alpha
                )
                
                save_series_config(series_name, config_db)
                save_series_embeddings(series_name, embeddings_db)
                
            t_diarization_end = time.time()
            
            # 4. Transcription
            t_transcribe_start = time.time()
            if self.progress_callback: self.progress_callback(f"{base_name}: Transcription")
            result = self.transcription_engine.transcribe(temp_wav)
            self.transcription_engine.cleanup()
            t_transcribe_end = time.time()
            
            # 5. Text Merging & Heuristics
            if self.progress_callback: self.progress_callback(f"{base_name}: Merging text")
            
            total_podcast_duration, insert_speakers = self._analyze_inserts(result, diarization, speaker_mapping, config_db, embeddings_db, series_name)
            raw_text = self._merge_segments(result, diarization, speaker_mapping, insert_speakers)
            
            with open(output_txt, "w", encoding="utf-8") as f:
                f.write(raw_text)

            # --- PART 2: LLM POST-PROCESSING (Potentially Background) ---
            gpu_timings = (file_start_time, t_convert_start, t_convert_end, t_diarization_start, t_diarization_end, t_transcribe_start, t_transcribe_end)
            
            if executor:
                if self.progress_callback: self.progress_callback(f"{base_name}: LLM task queued")
                executor.submit(self._run_llm_and_report, raw_text, config_db, series_name, base_name, total_podcast_duration, gpu_timings, chosen_model, output_txt, insert_speakers)
            else:
                self._run_llm_and_report(raw_text, config_db, series_name, base_name, total_podcast_duration, gpu_timings, chosen_model, output_txt, insert_speakers)
                
            return True
            
        except Exception as e:
            print(f"   ❌ [Pipeline Error] {e}")
            import traceback
            traceback.print_exc()
            return False

    def _run_llm_and_report(self, raw_text, config_db, series_name, base_name, total_podcast_duration, gpu_timings, chosen_model, output_txt, insert_speakers):
        """Runs LLM processing and writes the final report. Can be run in background."""
        try:
            (file_start_time, t_convert_start, t_convert_end, t_diarization_start, t_diarization_end, t_transcribe_start, t_transcribe_end) = gpu_timings
            t_llm_start = time.time()
            
            # Auto-naming
            podcast_metadata = {}
            if self.config.diarization.auto_naming and self.config.post_processing.enabled:
                lines_arr = raw_text.split("\n")
                context_chunk = "\n".join(lines_arr[:200])
                extracted_data = extract_podcast_metadata_with_llm(context_chunk, self.config)
                
                if extracted_data:
                    extracted_speakers = extracted_data.get("speakers", {})
                    podcast_metadata = extracted_data.get("metadata", {})
                    updated_count = self._apply_auto_names(extracted_speakers, config_db, series_name)
                    if updated_count > 0:
                         for spk_key, human_name in config_db.items():
                             if spk_key.startswith("GLOBAL_SPEAKER_") and human_name != spk_key:
                                  raw_text = raw_text.replace(spk_key, human_name)
            
            # Post-processing
            is_single_speaker = self._is_single_speaker(raw_text, chosen_model)
            if self.config.post_processing.enabled:
                formatted_text = run_post_processing(raw_text, self.config, is_single_speaker)
                header_str = self._generate_markdown_header(podcast_metadata, config_db)
                final_text = header_str + "\n\n" + formatted_text
                
                final_output_path = os.path.join(self.config.paths.output_dir, f"{base_name}_formatted.md")
                with open(final_output_path, "w", encoding="utf-8") as f:
                    f.write(final_text)
                
                if self.config.post_processing.provider.lower() == "ollama":
                    unload_ollama_model(self.config.post_processing.model)
            
            t_llm_end = time.time()
            file_end_time = time.time()
            
            self._write_report(base_name, total_podcast_duration, file_end_time - file_start_time, 
                               t_convert_end - t_convert_start, t_diarization_end - t_diarization_start, 
                               t_transcribe_end - t_transcribe_start, t_llm_end - t_llm_start, 
                               chosen_model, output_txt, insert_speakers)
        except Exception as e:
            print(f"   ❌ [LLM Background Error] {base_name}: {e}")

    def _analyze_inserts(self, result, diarization, speaker_mapping, config_db, embeddings_db, series_name) -> Tuple[float, Set[str]]:
        total_podcast_duration = 0.0
        speaker_durations = {}
        
        if diarization is not None:
            for segment in result["segments"]:
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"].strip()
                
                if self.config.processing.skip_noise_and_music:
                    if bool(re.match(r'^[\(\[].*?[\)\]]$', text)): continue
                        
                total_podcast_duration = max(total_podcast_duration, end_time)
                speaker = get_speaker(diarization, start_time, end_time, speaker_mapping)
                if speaker != "UNKNOWN" and speaker != "[UNKNOWN]":
                    speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + (end_time - start_time)
                    
        insert_speakers = set()
        if total_podcast_duration > 0:
            for spk, duration in speaker_durations.items():
                if (duration / total_podcast_duration) <= 0.01:
                    insert_speakers.add(spk)
        return total_podcast_duration, insert_speakers

    def _merge_segments(self, result, diarization, speaker_mapping, insert_speakers) -> str:
        lines = []
        current_speaker = ""
        current_start_time = 0.0
        current_text = []
    
        for segment in result["segments"]:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"].strip()
        
            if "DimaTorzok" in text or "Продолжение следует" in text or "Подпишитесь на канал" in text:
                continue
        
            if diarization is None:
                speaker = "GLOBAL_SPEAKER_1"
            else:
                speaker = get_speaker(diarization, start_time, end_time, speaker_mapping)
                if speaker in insert_speakers: speaker = "[Вставка]"
        
            if self.config.processing.skip_noise_and_music:
                is_pure_noise = bool(re.match(r'^[\(\[].*?[\)\]]$', text))
                if is_pure_noise or speaker == "UNKNOWN" or speaker == "[UNKNOWN]":
                    continue
        
            if not current_speaker:
                current_speaker, current_start_time, current_text = speaker, start_time, [text]
            elif speaker == current_speaker:
                current_text.append(text)
            else:
                lines.append(self._format_line(current_start_time, current_speaker, " ".join(current_text)))
                current_speaker, current_start_time, current_text = speaker, start_time, [text]
                            
        if current_speaker:
            lines.append(self._format_line(current_start_time, current_speaker, " ".join(current_text)))
            
        return "\n".join(lines) + "\n"

    def _format_line(self, start_time, speaker, text) -> str:
        h, rem = divmod(start_time, 3600)
        m, s = divmod(rem, 60)
        return f"[{int(h):02d}.{int(m):02d}.{int(s):02d}] {speaker}: {text}"

    def _apply_auto_names(self, extracted_speakers, config_db, series_name) -> int:
        updated_count = 0
        for global_spk, data in extracted_speakers.items():
            if data.get("confidence", 0) >= 80:
                real_name = data.get("name", "")
                if global_spk in config_db:
                    config_db[global_spk] = real_name
                    updated_count += 1
        if updated_count > 0:
            save_series_config(series_name, config_db)
        return updated_count

    def _is_single_speaker(self, raw_text: str, chosen_model: str) -> bool:
        lines_list = raw_text.strip().split("\n")
        speaker_counts = {}
        total_speech_lines = 0
        for line in lines_list:
            match = re.match(r'\[\d{2}\.\d{2}\.\d{2}\] (.*?):', line)
            if match:
                spk = match.group(1).strip()
                speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
                total_speech_lines += 1
        
        if total_speech_lines > 0:
            max_spk_lines = max(speaker_counts.values())
            if (max_spk_lines / total_speech_lines) >= 0.90 or chosen_model == "skip":
                 return True
        return False

    def _generate_markdown_header(self, metadata, config_db) -> str:
        header_lines = []
        show_name = metadata.get("show_name") or "Podcast"
        ep_num = metadata.get("episode_number")
        header_lines.append(f"# 🎙️ {show_name} - Выпуск {ep_num}" if ep_num else f"# 🎙️ {show_name}")
        
        for key, icon in [("date", "📅 Дата"), ("topic", "💡 Тема")]:
            val = metadata.get(key)
            if val: header_lines.append(f"**{icon}:** {val}")
                        
        guests = [human_name for spk, human_name in config_db.items() if spk.startswith("GLOBAL_SPEAKER_") and human_name != spk]
        if guests: header_lines.append(f"**👥 Участники:** {', '.join(guests)}")
                        
        header_lines.append("\n---")
        return "\n".join(header_lines)

    def _write_report(self, base_name, duration, total_time, t_conv, t_diar, t_trans, t_llm, model, output_txt, insert_speakers):
        def fmt(s):
            m, s = divmod(int(s), 60)
            h, m = divmod(m, 60)
            return f"{h}h {m}m {s}s" if h > 0 else f"{m}m {s}s" if m > 0 else f"{s}s"

        final_speakers = set()
        with open(output_txt, "r", encoding="utf-8") as f:
            for line in f:
                m = re.match(r'\[\d{2}\.\d{2}\.\d{2}\] (.*?):', line)
                if m: final_speakers.add(m.group(1).strip())
        
        report_path = os.path.join(self.config.paths.output_dir, "processing_report.md")
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        hw_mode = "Hardware Accelerated" if self.config.performance.num_workers > 0 else "Low Memory Mode"
        
        report_content = (
            f"### 🎙️ {base_name}\n"
            f"**📅 Date:** {now_str}\n"
            f"**⏳ Audio Duration:** {fmt(duration)} | **⏱️ Total Time:** {fmt(total_time)}\n\n"
            f"- **⚙️ Hardware Mode:** {hw_mode}\n"
            f"- **👥 Speakers Found:** {len(final_speakers)} (Main: {len([s for s in final_speakers if s != '[Вставка]'])}, Inserts: {len(insert_speakers)})\n"
            f"- **🔉 Conversion:** {fmt(t_conv)} | **🗣️ Diarization:** {fmt(t_diar)}\n"
            f"- **📝 Transcription:** {fmt(t_trans)} | **🤖 Post-processing:** {fmt(t_llm)}\n"
            "---\n\n"
        )
        with open(report_path, "a", encoding="utf-8") as rf:
            rf.write(report_content)
