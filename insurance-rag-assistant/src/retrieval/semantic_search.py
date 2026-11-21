
import json
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_FILE = PROJECT_ROOT / "vector_store" / "insurance_policy.index"
METADATA_FILE = PROJECT_ROOT / "vector_store" / "chunk_metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3


def load_resources():
    """Load FAISS index, chunk metadata and embedding model."""

    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_FILE}")

    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")

    # Loads the FAISS index we created earlier
    index = faiss.read_index(str(INDEX_FILE))

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        # Loads the original chunk text and page numbers.
        metadata = json.load(file)

    # Loads the same embedding model we used to generate document embeddings.
    model = SentenceTransformer(MODEL_NAME)

    return index, metadata, model


def semantic_search(query, index, metadata, model, top_k=TOP_K):
    """Find the most relevant document chunks for a query."""

    # Step 1: Convert the question into an embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Step 2: Convert embedding to FAISS-compatible format
    query_embedding = query_embedding.astype("float32")

    # Step 3: Search the FAISS index
    top_k = min(top_k, index.ntotal)
    # scores contains similarity scores.
    # indices contains the positions of matching vectors in FAISS.
    #top_k=3 means retrieve the three closest matches.
    scores, indices = index.search(query_embedding, top_k)

    # Step 4: Display matching chunks
    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        # The returned index position maps to the corresponding metadata record because we 
        # inserted the embeddings and metadata in the same order.
        chunk = metadata[idx]

        result = {
            "chunk_id": chunk["chunk_id"],
            "similarity_score": float(score),
            "page_numbers": chunk["page_numbers"],
            "text": chunk["text"]
        }

        results.append(result)

    return results


if __name__ == "__main__":

    print("Loading search resources...")
    index, metadata, model = load_resources()

    print("\nSemantic Search Ready!")
    print("Type 'exit' to stop.")

    while True:
        query = input("\nEnter your question: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Please enter a question.")
            continue

        results = semantic_search(
            query=query,
            index=index,
            metadata=metadata,
            model=model,
            top_k=TOP_K
        )

        print("\nTop matching chunks:")
        print("=" * 70)

        for rank, result in enumerate(results, start=1):

            print(f"\nRank: {rank}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Similarity Score: {result['similarity_score']:.4f}")
            print(f"Page Numbers: {result['page_numbers']}")
            print("\nChunk Text:")
            print(result["text"][:1000])
            print("-" * 70)
