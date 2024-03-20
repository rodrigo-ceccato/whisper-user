import argparse
import os
from whisper import load_model
from audio_helpers import convert_audio_to_wav, is_supported_format
import torch

def transcribe_audio(file_path, model):
    """
    Transcribes an audio file using the Whisper model.
    
    Parameters:
        file_path (str): The path to the audio file.
        model: The loaded Whisper model.
    
    Returns:
        str: The transcribed text.
    """
    result = model.transcribe(file_path)
    return result['text']

def append_gpt_prompt_correction(transcription):
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

    return transcription + gpt_prompt

def process_folder(input_folder, output_file):
    """
    Processes each audio file in the input folder, transcribes it, and writes the outputs to a text file.
    
    Parameters:
        input_folder (str): The path to the folder containing audio files.
        output_file (str): The path to the output text file.
    """

    # device is GPU if we have more than 10GB of VRAM, otherwise CPU
    DEVICE = "cuda" if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 10e9 else "cpu"
    print(f"Using device: {DEVICE}")
    model = load_model("large-v3", device=DEVICE)
    transcriptions = []

    for root, _, files in os.walk(input_folder):
        for file in sorted(files):
            file_path = os.path.join(root, file)
            if is_supported_format(file_path):
                print(f"Processing {file_path}...")
                if not file_path.endswith('.wav'):
                    temp_path = file_path + ".wav"
                    convert_audio_to_wav(file_path, temp_path)
                    file_path = temp_path

                print(f"Transcribing {file_path}...")
                transcription = transcribe_audio(file_path, model)
                transcriptions.append(transcription)

                if file_path != os.path.join(root, file):  # Cleanup if converted
                    os.remove(file_path)

    transcription_w_prompt = append_gpt_prompt_correction(
        "\n".join(transcriptions))

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

    print(f"Transcription process completed.")

