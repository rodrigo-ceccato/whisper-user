import subprocess
from pydub import AudioSegment
from whisper import Whisper


def convert_audio_to_wav(source_path: str, target_path: str) -> None:
    """
    Converts an audio file to WAV format using pydub.

    Parameters:
        source_path (str): The path to the source audio file.
        target_path (str): The path to save the converted WAV file.
    """
    audio = AudioSegment.from_file(source_path)
    audio.export(target_path, format="wav")


def is_supported_format(file_path: str) -> bool:
    """
    Checks if the file format is supported by Whisper natively.

    Parameters:
        file_path (str): The path to the audio file.

    Returns:
        bool: True if the format is supported, False otherwise.
    """
    supported_formats = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    return any(file_path.endswith(ext) for ext in supported_formats)


def transcribe_audio(file_path: str, model: Whisper) -> str:
    """
    Transcribes an audio file using the Whisper model.

    Parameters:
        file_path (str): The path to the audio file.
        model(Whisper): The loaded Whisper model.

    Returns:
        str: The transcribed text.
    """
    result = model.transcribe(file_path, verbose=False)
    return result["text"]
