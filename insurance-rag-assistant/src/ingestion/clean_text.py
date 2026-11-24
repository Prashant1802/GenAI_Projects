
import re
import math
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

def split_into_pages(text):
    """Separate extracted text into individual pages."""

    pattern = r"--- Page (\d+) ---"
    matches = list(re.finditer(pattern, text))

    pages = []

    for i, match in enumerate(matches):
        page_number = int(match.group(1))
        start = match.end()

        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(text)
        )

        pages.append({
            "page_number": page_number,
            "text": text[start:end]
        })

    return pages

def remove_repeated_headers_footers(pages):
    """Remove lines repeated in page header/footer regions."""

    header_footer_lines = []

    for page in pages:
        lines = page["text"].splitlines()

        # Inspect only the first and last 8 lines.
        candidates = lines[:8] + lines[-8:]

        normalized = {
            re.sub(r"\s+", " ", line).strip()
            for line in candidates
            if line.strip()
        }

        header_footer_lines.extend(normalized)

    line_counts = Counter(header_footer_lines)

    minimum_pages = max(3, math.ceil(len(pages) * 0.3))

    repeated_lines = {
        line
        for line, count in line_counts.items()
        if count >= minimum_pages
    }

    cleaned_pages = []

    for page in pages:
        lines = page["text"].splitlines()

        # Identify lines appearing in header/footer positions.
        header_footer_positions = (
            set(range(min(8, len(lines))))
            | set(range(max(0, len(lines) - 8), len(lines)))
        )

        cleaned_lines = []

        for i, line in enumerate(lines):
            normalized = re.sub(r"\s+", " ", line).strip()

            if (
                i in header_footer_positions
                and normalized in repeated_lines
            ):
                continue

            cleaned_lines.append(line)

        cleaned_pages.append({
            "page_number": page["page_number"],
            "text": "\n".join(cleaned_lines)
        })

    return cleaned_pages, repeated_lines

def clean_text(text):

    # Normalize different bullet characters
    text = text.replace("•", "-")
    text = text.replace("▪", "-")
    text = text.replace("●", "-")

    # Remove unwanted control characters
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)

    pages = split_into_pages(text)

    cleaned_pages, removed_lines = remove_repeated_headers_footers(
        pages
    )

    output_parts = []

    for page in cleaned_pages:
        page_text = page["text"]

        lines = [line.strip() for line in page_text.splitlines()]

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

        output_parts.append(
            f"--- Page {page['page_number']} ---\n\n"
            + "\n".join(cleaned_lines).strip()
        )

    return "\n\n".join(output_parts).strip(), removed_lines


if __name__ == "__main__":

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    original_text = INPUT_FILE.read_text(encoding="utf-8")

    cleaned_text, removed_lines = clean_text(original_text)

    OUTPUT_FILE.write_text(
        cleaned_text,
        encoding="utf-8"
    )

    print("Cleaning completed!")
    print("Original characters:", len(original_text))
    print("Cleaned characters:", len(cleaned_text))
    print("Repeated header/footer lines removed:", len(removed_lines))

    print("\nRemoved lines:")

    for line in sorted(removed_lines):
        print("-", line)

    print("\nSaved at:", OUTPUT_FILE)
