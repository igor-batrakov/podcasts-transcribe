import os
import json
import yaml
import numpy as np
from scipy.spatial.distance import cosine

from config_loader import load_global_config

def get_series_dir(series_name):
    """Returns the path to the series settings directory"""
    paths_cfg = load_global_config().get("paths", {})
    speakers_dir = paths_cfg.get("speakers_dir", "speakers")
    dir_path = os.path.join(speakers_dir, series_name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def load_series_config(series_name):
    """Loads the human-readable configuration for the series (YAML)"""
    config_path = os.path.join(get_series_dir(series_name), "config.yaml")
    
    # On-the-fly migration from old JSON config to YAML
    old_json_path = os.path.join(get_series_dir(series_name), "config.json")
    if os.path.exists(old_json_path) and not os.path.exists(config_path):
        with open(old_json_path, "r", encoding="utf-8") as f:
            old_data = json.load(f)
        save_series_config(series_name, old_data)
        os.remove(old_json_path)
            
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

def save_series_config(series_name, config):
    """Saves the human-readable configuration for the series (YAML)"""
    config_path = os.path.join(get_series_dir(series_name), "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

def load_series_embeddings(series_name):
    """Loads the hidden voice embeddings for the series"""
    emb_path = os.path.join(get_series_dir(series_name), "embeddings.json")
    if os.path.exists(emb_path):
        with open(emb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Convert back to numpy arrays
            return {k: np.array(v) for k, v in data.items()}
    return {}

def save_series_embeddings(series_name, embeddings_db):
    """Saves the hidden voice embeddings for the series"""
    emb_path = os.path.join(get_series_dir(series_name), "embeddings.json")
    data_to_save = {k: v.tolist() for k, v in embeddings_db.items()}
    with open(emb_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f)

def merge_duplicate_speakers(series_name, config_db, embeddings_db):
    """
    Finds speakers (GLOBAL_SPEAKER_X) assigned the same human name.
    Averages their embeddings, keeps only one GLOBAL ID, and deletes the rest.
    Returns True if changes were made (requiring a save).
    """
    changed = False
    
    # Reverse dictionary: Human_name -> [list of GLOBAL_IDs]
    name_to_ids = {}
    for global_id, human_name in config_db.items():
        if human_name not in name_to_ids:
            name_to_ids[human_name] = []
        name_to_ids[human_name].append(global_id)
        
    for human_name, ids in name_to_ids.items():
        if len(ids) > 1:
            print(f"      [Merge] Found {len(ids)} profiles named '{human_name}': {', '.join(ids)}")
            
            # Collect all valid embeddings for this person
            valid_embs = [embeddings_db[gid] for gid in ids if gid in embeddings_db and np.sum(np.abs(embeddings_db[gid])) > 0]
            
            if len(valid_embs) > 0:
                # Average the vectors (arithmetic mean across dimensions)
                merged_emb = np.mean(valid_embs, axis=0)
                
                # Keep the first ID as primary
                main_id = ids[0]
                embeddings_db[main_id] = merged_emb
                
                # Delete the other "duplicate" IDs from config and embeddings DB
                for idx in range(1, len(ids)):
                    duplicate_id = ids[idx]
                    if duplicate_id in embeddings_db:
                        del embeddings_db[duplicate_id]
                    if duplicate_id in config_db:
                        del config_db[duplicate_id]
                
                print(f"      [Merge ✓] Profiles merged under ID '{main_id}'.")
                changed = True
                
    return changed

def get_global_speaker_mapping(diarization, series_name, config_db, embeddings_db, similarity_threshold, ema_alpha):
    """
    Maps local speakers to the series database.
    Updates embeddings using Exponential Moving Average (EMA).
    Returns dictionary {local_speaker: global_speaker}.
    """
    mapping = {}
    
    embeddings = getattr(diarization, "speaker_embeddings", None)
    if embeddings is None:
        return mapping
        
    annotation = getattr(diarization, "speaker_diarization", diarization)
    labels = list(annotation.labels()) if annotation else []
    
    for i, local_label in enumerate(labels):
        if i >= len(embeddings):
            break
            
        speaker_emb = embeddings[i]
        
        # Prevent division by zero if embedding is completely empty
        if np.sum(np.abs(speaker_emb)) == 0:
            mapping[local_label] = "UNKNOWN"
            continue
        
        best_match = None
        min_distance = float('inf')
        
        for global_id, saved_emb in embeddings_db.items():
            if np.sum(np.abs(saved_emb)) == 0:
                continue
            dist = cosine(speaker_emb, saved_emb)
            if dist < min_distance:
                min_distance = dist
                best_match = global_id
                    
        if best_match is not None and min_distance <= similarity_threshold:
            if min_distance < 0.4:
                embeddings_db[best_match] = (
                    (1.0 - ema_alpha) * embeddings_db[best_match] + 
                    ema_alpha * speaker_emb
                )
            
            display_name = config_db.get(best_match, best_match)
            mapping[local_label] = display_name
            print(f"      [✓] {local_label} recognized as {display_name} (distance: {min_distance:.3f})")
        else:
            new_id = f"GLOBAL_SPEAKER_{len(embeddings_db) + 1}"
            embeddings_db[new_id] = speaker_emb
            config_db[new_id] = new_id
            
            mapping[local_label] = new_id
            dist_str = f" (closest dist: {min_distance:.3f})" if min_distance != float('inf') else ""
            print(f"      [+] Added new host: {new_id} -> {local_label}{dist_str}")
            
    return mapping

def get_speaker(diarization, start, end, speaker_mapping=None):
    """Finds the speaker who talks the most during a specific timeframe"""
    speaker_durations: dict[str, float] = {}
    
    annotation = getattr(diarization, "speaker_diarization", diarization)
    if annotation is None:
        return "UNKNOWN"
    
    for turn, _, local_speaker in annotation.itertracks(yield_label=True):
        overlap_start = max(start, turn.start)
        overlap_end = min(end, turn.end)
        overlap = float(max(0, overlap_end - overlap_start))
        
        if overlap > 0:
            speaker_durations[local_speaker] = speaker_durations.get(local_speaker, 0.0) + overlap
            
    if not speaker_durations:
        return "UNKNOWN"
    
    best_local_speaker = max(speaker_durations.keys(), key=lambda k: speaker_durations[k])
    
    if speaker_mapping and best_local_speaker in speaker_mapping:
        return speaker_mapping[best_local_speaker]
    return best_local_speaker
