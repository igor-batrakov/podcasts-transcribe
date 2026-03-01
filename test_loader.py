import os
import yaml
from huggingface_hub import hf_hub_download
from pyannote.audio import Pipeline

token = os.environ.get('HF_TOKEN')
path = hf_hub_download(repo_id='pyannote/speaker-diarization', filename='config.yaml', revision='2.1', token=token)
with open(path) as f:
    config = yaml.safe_load(f)

# The magic fix: replace "@" with a dict config
config["pipeline"]["params"]["segmentation"] = {
    "checkpoint": "pyannote/segmentation",
    "revision": "2022.07"
}

try:
    pipeline = Pipeline.from_pretrained(config, cache_dir=None, token=token)
    print("SUCCESS DICT CONFIG!")
except Exception as e:
    print(f"FAIL DICT CONFIG: {e}")
