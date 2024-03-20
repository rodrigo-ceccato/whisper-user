import subprocess
from pydub import AudioSegment

def convert_audio_to_wav(source_path, target_path):
    """
    Converts an audio file to WAV format using pydub.
    
    Parameters:
        source_path (str): The path to the source audio file.
        target_path (str): The path to save the converted WAV file.
    """
    audio = AudioSegment.from_file(source_path)
    audio.export(target_path, format="wav")

def is_supported_format(file_path):
    """
    Checks if the file format is supported by Whisper natively.
    
    Parameters:
        file_path (str): The path to the audio file.
    
    Returns:
        bool: True if the format is supported, False otherwise.
    """
    supported_formats = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    return any(file_path.endswith(ext) for ext in supported_formats)

