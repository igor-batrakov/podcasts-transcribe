import os
import pytest
from utils import get_series_name, get_unique_filename

def test_get_series_name():
    """Test extracting the series name from various filename patterns."""
    assert get_series_name("rt_podcast_10.mp3") == "rt_podcast"
    assert get_series_name("rt_podcast1001.wav") == "rt_podcast"
    assert get_series_name("interview_elon_musk.m4a") == "interview_elon_musk"
    assert get_series_name("myshow 05.flac") == "myshow" # Assumes trailing numbers are stripped (current logic splits on digit)

def test_get_unique_filename(tmp_path):
    """Test generating a unique filename when collisions exist."""
    # Create a dummy file in the temporary test directory
    test_file = tmp_path / "test_output.txt"
    test_file.touch()
    
    # Check that a unique numeric suffix is appended
    unique_name = get_unique_filename(str(test_file))
    expected_name = str(tmp_path / "test_output_v1.txt")
    
    assert unique_name == expected_name
    
    # Create the _v1 file and test again
    test_file_1 = tmp_path / "test_output_v1.txt"
    test_file_1.touch()
    
    unique_name_2 = get_unique_filename(str(test_file))
    expected_name_2 = str(tmp_path / "test_output_v2.txt")
    
    assert unique_name_2 == expected_name_2
