import argparse
import os
from whisper import load_model
from audio_helpers import convert_audio_to_wav, is_supported_format
import torch
import subprocess

def transcribe_audio(file_path, model):
    """
    Transcribes an audio file using the Whisper model.
    
    Parameters:
        file_path (str): The path to the audio file.
        model: The loaded Whisper model.
    
    Returns:
        str: The transcribed text.
    """
    result = model.transcribe(file_path, verbose=False)
    return result['text']


def gpt_prompt_correction():
    """
    Appends a GPT-4 prompt to the transcription to correct common mistakes.
    
    Parameters:
        transcription (str): The original transcription text.
    
    Returns:
        str: The transcription text with a GPT-4 prompt appended.
    """

    # get prompt text from file prompt.txt
    with open("prompt.txt", "r") as prompt_file:
        gpt_prompt = prompt_file.read()

    return gpt_prompt

def process_folder(input_folder, output_file):
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
    DEVICE = "cuda" if torch.cuda.is_available() and torch.cuda.get_device_properties(
        0).total_memory > memory_required_per_model[MODEL] else "cpu"
    print(f"Using device: {DEVICE} for model: {MODEL}")
    model = load_model(MODEL, device=DEVICE)
    transcriptions = []

    for root, _, files in os.walk(input_folder):
        for file in sorted(files):
            file_path = os.path.join(root, file)
            if is_supported_format(file_path):
                print(f"Processing {file_path}...")
                if not file_path.endswith('.wav'):
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

    transcription_w_prompt = "\n\n".join(transcriptions)

    # add prompt at beginning of transcription
    transcription_w_prompt = gpt_prompt_correction() + transcription_w_prompt

    print("Transcription process completed.\n\n\n")

    if (output_file is not None):
        with open(output_file, "w") as output:
         output.write(transcription_w_prompt)
    else:
        print(transcription_w_prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio files in a folder using Whisper and correct them with a GPT-4 prompt.")
    parser.add_argument("input_folder", type=str, help="The path to the input folder containing audio files.")
    # if no output file argument, default to None
    parser.add_argument("-o", "--output_file", type=str,
                        help="The path to the output text file for concatenated transcriptions.", default=None, required=False)

    args = parser.parse_args()
    process_folder(args.input_folder, args.output_file)

