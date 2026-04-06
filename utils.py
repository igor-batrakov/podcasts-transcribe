import os
import re

def get_series_name(filename):
    """
    Extracts the podcast series name from the filename.
    Removes numbers and separators from the beginning and end.
    Example: rt_podcast1001.mp3 -> rt_podcast
    """
    name, _ = os.path.splitext(filename)
    
    # Try cleaning the name
    cleaned = re.sub(r'\d+$', '', name)
    cleaned = re.sub(r'[-_]+$', '', cleaned)
    cleaned = re.sub(r'^\d+', '', cleaned)
    cleaned = re.sub(r'^[-_]+', '', cleaned)
    
    # If cleaning results in non-empty string, use it
    if cleaned.strip():
        return cleaned.strip()
        
    # If cleaning made it empty (e.g. "123.mp3"), return original name
    return name if name else "unknown_series"

def get_unique_filename(base_path):
    """Returns a unique filename by adding an increment if the file already exists."""
    if not os.path.exists(base_path):
        return base_path
    
    name, ext = os.path.splitext(base_path)
    counter = 1
    while os.path.exists(f"{name}_v{counter}{ext}"):
        counter += 1
    return f"{name}_v{counter}{ext}"
