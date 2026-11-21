
import faiss
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_FILE = (
    PROJECT_ROOT
    / "vector_store"
    / "insurance_policy.index"
)

# Load saved index
index = faiss.read_index(str(INDEX_FILE))

print("FAISS index loaded successfully!")

print("Total vectors:", index.ntotal)
print("Vector dimensions:", index.d)
