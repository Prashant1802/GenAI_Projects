
import re
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "extracted_text"
    / "extracted_text.txt"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "extracted_text"
    / "cleaned_text.txt"
)


def clean_text(text):

    # Normalize different bullet characters
    text = text.replace("•", "-")
    text = text.replace("▪", "-")
    text = text.replace("●", "-")

    # Remove unwanted control characters
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)

    # Remove trailing spaces from each line
    lines = [line.strip() for line in text.splitlines()]

    # Collapse more than two consecutive blank lines
    cleaned_lines = []
    blank_count = 0

    for line in lines:

        if not line:
            blank_count += 1

            if blank_count <= 2:
                cleaned_lines.append("")

        else:
            blank_count = 0
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    return text.strip()


if __name__ == "__main__":

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    original_text = INPUT_FILE.read_text(encoding="utf-8")

    cleaned_text = clean_text(original_text)

    OUTPUT_FILE.write_text(
        cleaned_text,
        encoding="utf-8"
    )

    print("Cleaning completed!")
    print("Original characters:", len(original_text))
    print("Cleaned characters:", len(cleaned_text))
    print("Saved at:", OUTPUT_FILE)
