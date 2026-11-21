
import json
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer


# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input and output paths
CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "extracted_text"
    / "chunks.json"
)

VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

EMBEDDINGS_FILE = VECTOR_STORE_DIR / "chunk_embeddings.npy"
METADATA_FILE = VECTOR_STORE_DIR / "chunk_metadata.json"

# Embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def generate_embeddings():

    # Create output directory
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    # Load chunks
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("No chunks found in chunks.json")

    # Extract text from each chunk
    texts = [chunk["text"] for chunk in chunks]

    print("Total chunks:", len(texts))

    # Load embedding model
    print("Loading embedding model...")

    # This loads all-MiniLM-L6-v2 into memory.
    model = SentenceTransformer(MODEL_NAME)

    # Generate embeddings
    print("Generating embeddings...")

    # normalize_embeddings=True ensures that the embeddings are normalized to unit length, 
    # which is often useful for similarity calculations.
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # This saves the numerical vectors in NumPy's .npy format.
    # We also save the original chunk text and page metadata in JSON. This is important because a 
    # vector alone isn't enough to display the actual retrieved policy content
    np.save(EMBEDDINGS_FILE, embeddings)

    # Save chunk metadata
    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=4, ensure_ascii=False)

    # Display results
    print("\nEmbedding generation completed!")

    print("Number of chunks:", len(texts))
    # ex: output = (45,384): 45 rows = 45 document chunks, 384 columns = 384 numerical dimensions per chunk.
    print("Embedding shape:", embeddings.shape)
    print("Vector dimensions:", embeddings.shape[1])

    print("Embeddings saved at:", EMBEDDINGS_FILE)
    print("Metadata saved at:", METADATA_FILE)


if __name__ == "__main__":

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}"
        )

    generate_embeddings()
