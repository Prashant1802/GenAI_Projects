
import json
import faiss
import numpy as np

from pathlib import Path


# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input files
EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "vector_store"
    / "chunk_embeddings.npy"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "vector_store"
    / "chunk_metadata.json"
)

# Output file
FAISS_INDEX_FILE = (
    PROJECT_ROOT
    / "vector_store"
    / "insurance_policy.index"
)


def build_faiss_index():

    # Check input files
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(EMBEDDINGS_FILE)

    if not METADATA_FILE.exists():
        raise FileNotFoundError(METADATA_FILE)

    # Load embeddings
    embeddings = np.load(EMBEDDINGS_FILE)

    # Load metadata
    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    # Validate input
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")

    if len(embeddings) != len(metadata):
        raise ValueError(
            "Number of embeddings and metadata records do not match"
        )

    # FAISS expects float32 vectors
    embeddings = np.ascontiguousarray(
        embeddings,
        dtype=np.float32
    )

    # Get embedding dimensions
    vector_dimensions = embeddings.shape[1]

    print("Total vectors:", len(embeddings))
    print("Vector dimensions:", vector_dimensions)

    # IndexFlatIP creates an exact inner-product search index.
    # vector_dimensions tells FAISS the size of each vector.
    index = faiss.IndexFlatIP(vector_dimensions)

    # Add embeddings to the index
    index.add(embeddings)

    # Save FAISS index to disk
    faiss.write_index(index, str(FAISS_INDEX_FILE))

    print("\nFAISS index created successfully!")
    print("Vectors stored:", index.ntotal)
    print("Index dimensions:", index.d)
    print("Index saved at:", FAISS_INDEX_FILE)


if __name__ == "__main__":
    build_faiss_index()
