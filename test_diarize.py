import torch
import os
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from transcribe import get_speaker

load_dotenv()
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=os.environ.get("HF_TOKEN")).to(device)

print("Running pipeline on first 5 seconds...")
# use first 5 seconds for fast test
import wave
with wave.open("rt_podcast1001.mp3.temp_diarization.wav", "rb") as fin:
    with wave.open("short.wav", "wb") as fout:
        fout.setnchannels(fin.getnchannels())
        fout.setsampwidth(fin.getsampwidth())
        fout.setframerate(fin.getframerate())
        fout.writeframes(fin.readframes(fin.getframerate() * 5))

diarization = pipeline("short.wav")
speaker = get_speaker(diarization, 0.0, 5.0)
print(f"Successfully got speaker: {speaker}")
