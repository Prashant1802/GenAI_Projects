
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


def merge_overlapping_text(existing_text, next_text):
    """Merge two chunks while removing exact overlapping text."""
    """ identifies matching text between two chunks. """
    """It checks for exact overlap, starting with the longest possible match (up to 1,000 characters).
        If it finds at least 50 matching characters, it removes the duplicate portion.
        If no overlap is found, it joins both chunks with a separator."""
    
    max_overlap = min(len(existing_text), len(next_text), 1000)
    min_overlap = 50

    for overlap in range(max_overlap, min_overlap - 1, -1):
        if existing_text[-overlap:] == next_text[:overlap]:
            return existing_text + next_text[overlap:]

    # If no exact overlap is found, keep both chunks.
    return existing_text + "\n\n" + next_text


def expand_chunk_context(metadata, matched_index, window=1):
    """Retrieve neighboring chunks and merge overlapping text."""
    """uses above merging function instead of directly joining chunks."""

    matched_chunk = metadata[matched_index]
    matched_pages = set(matched_chunk["page_numbers"])

    start = max(0, matched_index - window)
    end = min(len(metadata), matched_index + window + 1)

    context_text = ""
    context_chunk_ids = []

    for idx in range(start, end):

        chunk = metadata[idx]

        # Only retrieve chunks from the same page.
        if not matched_pages.intersection(chunk["page_numbers"]):
            continue

        if not context_text:
            context_text = chunk["text"]
        else:
            context_text = merge_overlapping_text(
                context_text,
                chunk["text"]
            )

        context_chunk_ids.append(chunk["chunk_id"])

    return {
        "context_text": context_text,
        "context_chunk_ids": context_chunk_ids
    }


def semantic_search(query, index, metadata, model, top_k=TOP_K):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype("float32")

    top_k = min(top_k, index.ntotal)

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        chunk = metadata[idx]

        expanded_context = expand_chunk_context(
            metadata,
            matched_index=int(idx),
            window=1
        )

        results.append({
            "chunk_id": chunk["chunk_id"],
            "similarity_score": float(score),
            "page_numbers": chunk["page_numbers"],
            "context_chunk_ids": expanded_context["context_chunk_ids"],
            "text": expanded_context["context_text"]
        })

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
            print(f"Matched Chunk ID: {result['chunk_id']}")
            print(f"Context Chunk IDs: {result['context_chunk_ids']}")
            print(f"Similarity Score: {result['similarity_score']:.4f}")
            print(f"Page Numbers: {result['page_numbers']}")
            print("\nChunk Text:")
            print(result["text"])
            print("-" * 70)
