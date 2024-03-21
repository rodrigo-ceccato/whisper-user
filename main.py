import argparse

from whisper_user.process_folder import process_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe audio files in a folder using Whisper and correct them with a GPT-4 prompt."
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="The path to the input folder containing audio files."
    )
    # if no output file argument, default to None
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="The path to the output text file for concatenated transcriptions.", default=None, required=False
    )
    args = parser.parse_args()
    process_folder(args.input_folder, args.output_file)
