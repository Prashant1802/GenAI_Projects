
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

def extract_pages(text):
    """It identifies markers such as:
        --- Page 1 ---
        --- Page 2 ---
        --- Page 3 ---

    And separates the document into individual page texts."""

    page_pattern = re.compile(r"--- Page (\d+) ---")
    matches = list(page_pattern.finditer(text))

    pages = []

    for i, match in enumerate(matches):
        page_number = int(match.group(1))

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        page_text = text[start:end].strip()

        if page_text:
            pages.append({
                "page_number": page_number,
                "text": page_text
            })

    return pages

def find_split_position(text, start, end):
    """
    Find a natural boundary near the end of the chunk.
    Priority: paragraph, newline, sentence, space.
    """

    minimum_position = start + int((end - start) * 0.7)

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

def find_next_start(text, desired_start):
    """Instead of blindly starting the next chunk at an arbitrary character position, it moves the start to a nearby boundary such as a
    newline, sentence ending, or space. This reduces the chances of chunks starting in the middle of a word."""

    separators = ["\n\n", "\n", ". ", " "]
    positions = []

    for separator in separators:
        position = text.find(separator, desired_start)

        if position != -1:
            positions.append(position + len(separator))

    return min(positions) if positions else desired_start

def chunk_page(page_text, page_number, starting_chunk_id):
    """It splits each page into chunks of approximately 2,000 characters, with approximately 400 characters of overlap.
        It also directly assigns:
        "page_numbers": [page_number]
        So every chunk knows which PDF page it came from."""

    chunks = []

    start = 0
    chunk_id = starting_chunk_id

    while start < len(page_text):

        end = min(start + CHUNK_SIZE, len(page_text))

        if end < len(page_text):
            end = find_split_position(page_text, start, end)

        chunk_content = page_text[start:end].strip()

        if chunk_content:
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_content,
                "page_numbers": [page_number]
            })

            chunk_id += 1

        if end >= len(page_text):
            break

        desired_start = max(start + 1, end - CHUNK_OVERLAP)

        start = find_next_start(page_text, desired_start)

    return chunks, chunk_id

def create_chunks(text):
    """Extract pages and create page-aware chunks."""

    pages = extract_pages(text)

    all_chunks = []
    next_chunk_id = 1

    for page in pages:

        page_chunks, next_chunk_id = chunk_page(
            page_text=page["text"],
            page_number=page["page_number"],
            starting_chunk_id=next_chunk_id
        )

        all_chunks.extend(page_chunks)

    return all_chunks

if __name__ == "__main__":

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    text = INPUT_FILE.read_text(encoding="utf-8")

    chunks = create_chunks(text)

    if not chunks:
        raise ValueError("No chunks were generated. Check page markers.")

    OUTPUT_FILE.write_text(
        json.dumps(chunks, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("Page-aware chunking completed!")
    print("Total chunks:", len(chunks))
    print("Chunk size:", CHUNK_SIZE)
    print("Chunk overlap:", CHUNK_OVERLAP)
    print("Saved at:", OUTPUT_FILE)

    print("\nFirst chunk preview:")
    print(json.dumps(chunks[0], indent=4, ensure_ascii=False))