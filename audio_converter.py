import os
import subprocess
import time
import hashlib

def cleanup_cache(cache_dir, max_size_mb, max_age_days):
    """Cleans up the cache from old files and limits max size (LRU algorithm)"""
    if not os.path.exists(cache_dir):
        return
        
    max_age_seconds = max_age_days * 24 * 60 * 60
    current_time = time.time()
    
    # 1. Delete everything older than max_age_days
    for filename in os.listdir(cache_dir):
        filepath = os.path.join(cache_dir, filename)
        if os.path.isfile(filepath):
            file_stat = os.stat(filepath)
            # Check last access time (atime)
            if current_time - file_stat.st_atime > max_age_seconds:
                try:
                    os.remove(filepath)
                except:
                    continue
                    
    # 2. Limit total directory size
    files: list[tuple[str, float, int]] = []
    total_size: int = 0
    for filename in os.listdir(cache_dir):
        filepath = os.path.join(cache_dir, filename)
        if os.path.isfile(filepath):
            size = int(os.path.getsize(filepath))
            files.append((filepath, os.stat(filepath).st_atime, size))
            # pyre-ignore[58]
            total_size += size
            
    max_size_bytes = max_size_mb * 1024 * 1024
    
    if total_size > max_size_bytes:
        # Sort files by access time (oldest first)
        files.sort(key=lambda x: x[1])
        for filepath, _, size in files:
            try:
                os.remove(filepath)
                # pyre-ignore[58]
                total_size -= size
                if total_size <= max_size_bytes:
                    break
            except:
                continue

def generate_cache_key(audio_path, time_limit):
    """Generates a unique cache key based on absolute path, size, and modification date."""
    abs_path = os.path.abspath(audio_path)
    file_stat = os.stat(abs_path)
    hash_str = f"{abs_path}_{file_stat.st_mtime}_{file_stat.st_size}_{time_limit}"
    return hashlib.md5(hash_str.encode('utf-8')).hexdigest()

from config_loader import load_global_config

def convert_to_wav(audio_path, time_limit=None, cache_cfg=None):
    """Converts media to a temporary WAV file with caching"""
    if cache_cfg is None:
        cache_cfg = load_global_config().get("cache", {})
        
    cache_dir = os.path.join(".cache", "audio")
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_key = generate_cache_key(audio_path, time_limit)
    cached_wav = os.path.join(cache_dir, f"{cache_key}.wav")
    
    if os.path.exists(cached_wav):
        print(f"   [CACHED] Reusing decoded WAV from previous runs...")
        try:
            # Update file access time to prevent LRU collection
            os.utime(cached_wav, None)
        except:
            pass
        return cached_wav

    print(f"   Converting {audio_path} to wav for speaker recognition...")
    ffmpeg_cmd = ["ffmpeg", "-y"]
    if time_limit is not None:
        print(f"   (TRUNCATED: Test run for {time_limit} sec.)")
        ffmpeg_cmd.extend(["-t", str(time_limit)])
    
    # Decode directly into the cache folder
    ffmpeg_cmd.extend(["-i", audio_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", cached_wav])
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Clean up the cache after adding the new large file
    cleanup_cache(cache_dir, cache_cfg["max_size_mb"], cache_cfg["max_age_days"])
    
    return cached_wav
