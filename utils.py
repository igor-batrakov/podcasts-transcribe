import os
import re

def get_series_name(filename):
    """
    Extracts the podcast series name from the filename.
    Removes numbers at the beginning or end of the string.
    Example: rt_podcast1001.mp3 -> rt_podcast
    """
    name, _ = os.path.splitext(filename)
    name = re.sub(r'\d+$', '', name)
    name = re.sub(r'[-_]+$', '', name)
    name = re.sub(r'^\d+', '', name)
    name = re.sub(r'^[-_]+', '', name)
    
    if not name:
        return "unknown_series"
    return name

def get_unique_filename(base_path):
    """Returns a unique filename by adding an increment if the file already exists."""
    if not os.path.exists(base_path):
        return base_path
    
    name, ext = os.path.splitext(base_path)
    counter = 1
    while os.path.exists(f"{name}_v{counter}{ext}"):
        counter += 1
    return f"{name}_v{counter}{ext}"
