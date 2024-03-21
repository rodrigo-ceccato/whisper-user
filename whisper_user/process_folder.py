import os
import subprocess

import torch
from whisper import load_model, Whisper

from whisper_user.utils.audio_helpers import convert_audio_to_wav, is_supported_format, transcribe_audio
from whisper_user.utils.gpt_helper import gpt_prompt_correction

def process_audio_files(files: list, root: str, model: Whisper) -> list:
    transcriptions = []
    for file in sorted(files):
        file_path = os.path.join(root, file)
        if not is_supported_format(file_path):
            return

        print(f"Processing {file_path}...")
        if not file_path.endswith(".wav"):
            temp_path = "./tmp-audio/temp.wav"
            # create a tmp folder if it doesn't exist
            subprocess.run(["mkdir", "-p", "./tmp-audio"])
            convert_audio_to_wav(file_path, temp_path)
            file_path = temp_path

        print(f"Transcribing {file_path}...")
        transcription = transcribe_audio(file_path, model)
        transcriptions.append(transcription)

        # clean up if converted
        if file_path == "./tmp-audio/temp.wav":
            os.remove(file_path)

    return transcriptions

def process_folder(input_folder: str, output_file: str) -> None:
    """
    Processes each audio file in the input folder, transcribes it, and writes the outputs to a text file.

    Parameters:
        input_folder (str): The path to the folder containing audio files.
        output_file (str): The path to the output text file.
    """

    # device is GPU if we have more than 10GB of VRAM, otherwise CPU
    MODEL = "small"
    memory_required_per_model = {
        "tiny": 1e9,
        "base": 1e9,
        "small": 2e9,
        "medium": 5e9,
        "large-v3": 10e9,
    }
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).total_memory
        > memory_required_per_model[MODEL]
        else "cpu"
    )
    print(f"Using device: {DEVICE} for model: {MODEL}")
    model = load_model(MODEL, device=DEVICE)
    transcriptions = []

    for root, _, files in os.walk(input_folder):
        transcriptions = transcriptions + process_audio_files(
            files,
            root,
            model
        )

    transcription_w_prompt = "\n\n".join(transcriptions)

    # add prompt at beginning of transcription
    transcription_w_prompt = gpt_prompt_correction() + transcription_w_prompt

    print("Transcription process completed.\n\n\n")

    if output_file is None:
        print(transcription_w_prompt)
        return
    
    with open(output_file, "w") as output:
        output.write(transcription_w_prompt)
        return
