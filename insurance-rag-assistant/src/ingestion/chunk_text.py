
import re
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "extracted_text"
    / "cleaned_text.txt"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "extracted_text"
    / "chunks.json"
)

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400


def find_split_position(text, start, end):
    """
    Find a natural boundary near the end of the chunk.
    Priority: paragraph, newline, sentence, space.
    """

    minimum_position = start + int(CHUNK_SIZE * 0.7)

    separators = ["\n\n", "\n", ". ", " "]

    for separator in separators:
        # rfind() searches backwards for a separator.
        # For example, if our target chunk ends in the middle of a paragraph, we try to find an earlier paragraph or sentence boundary.
        position = text.rfind(
            separator,
            minimum_position,
            end
        )

        if position != -1:
            return position + len(separator)

    return end


def chunk_text(text):

    chunks = []
    start = 0
    chunk_number = 1

    while start < len(text):

        end = min(start + CHUNK_SIZE, len(text))

        # Try to split at a natural text boundary
        if end < len(text):
            end = find_split_position(text, start, end)

        chunk_content = text[start:end].strip()

        if chunk_content:

            # Find page numbers mentioned in this chunk
            page_numbers = re.findall(
                r"--- Page (\d+) ---",
                chunk_content
            )
            # metadata
            chunks.append({
                "chunk_id": chunk_number,
                "text": chunk_content,
                "page_numbers": sorted(
                    set(int(page) for page in page_numbers)
                )
            })

            chunk_number += 1
        
        # Stop after processing the final chunk 
        # This prevents the loop from repeatedly processing the last overlapping section of the document.
        if end == len(text):
            break

        # Move forward while retaining overlap
        next_start = end - CHUNK_OVERLAP

        # Prevent an infinite loop
        start = max(next_start, start + 1)

    return chunks


if __name__ == "__main__":

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    text = INPUT_FILE.read_text(encoding="utf-8")

    chunks = chunk_text(text)

    OUTPUT_FILE.write_text(
        json.dumps(chunks, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("Chunking completed!")
    print("Total chunks:", len(chunks))
    print("Chunk size:", CHUNK_SIZE)
    print("Chunk overlap:", CHUNK_OVERLAP)
    print("Saved at:", OUTPUT_FILE)

    print("\nFirst chunk preview:")
    print(chunks[0]["text"][:500])
