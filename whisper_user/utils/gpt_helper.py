from os import getenv

PROMPT_FILE = getenv("PROMPT_ENV", "prompt.txt")


def gpt_prompt_correction():
    """
    Appends a GPT-4 prompt to the transcription to correct common mistakes.

    Parameters:
        transcription (str): The original transcription text.

    Returns:
        str: The transcription text with a GPT-4 prompt appended.
    """

    # get prompt text from file prompt.txt
    with open(PROMPT_FILE, "r") as prompt_file:
        gpt_prompt = prompt_file.read()

    return gpt_prompt
