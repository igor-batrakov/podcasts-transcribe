# Backlog of Future Enhancements

- [ ] **LLM Integration**: Add a post-processing step to format text, add punctuation, generate show summaries, and automatically infer speaker names from the transcript context.
- [x] ~~**Subtitles Export**: Automatically generate `.srt` or `.vtt` subtitle files for video platforms, or beautifully formatted `.md`/PDF documents.~~ *(Отменено)*
- [ ] **Web UI (Streamlit)**: Build a graphical interface using Streamlit for non-programmers to drag-and-drop audio files and rename speakers without manually editing YAML files.
- [x] **CLI UI Improvements**: Add terminal UI elements like progress bars for batch processing and colored logs using the `rich` library.

## Backlog for Tomorrow

- [x] **Smart Speaker Auto-Naming**: Automatically identify and rename speakers (e.g., `SPEAKER_00` -> `Игорь`) by analyzing the transcription text to see if they introduce themselves or are addressed by others.
- [x] **Transcript Polish**: Perform a general "cleanup" and formatting polish on the final output transcript file.
