Cosine Similarity = A.B/||A|| ||B||

Where:
- A = document embedding.
- B = query embedding.
- A.B = dot product.
- ||A|| and ||B|| = vector magnitudes.
The score generally ranges from -1 to 1.
- Close to 1: vectors point in similar directions.
- Close to 0: vectors are approximately orthogonal.
- Close to -1: vectors point in opposite directions.

FAISS = Facebook AI Similarity Search

It's a library developed by Meta for efficient similarity search and clustering of dense vectors.

FAISS supports different index types. For now, we'll use:
FAISS expects vectors in a compatible numerical format
IndexFlatIP — Flat Inner Product Index

Why?
- It compares the query against every stored vector.
- It gives exact similarity results.
- It's simple to understand and implement.
- It's suitable for our current small insurance policy dataset.

Index	        Search approach
IndexFlatIP	    Exact inner-product search
IndexFlatL2	    Exact Euclidean distance search
IndexIVFFlat	Approximate search using clusters
IndexHNSWFlat	Graph-based approximate search

Does FAISS store our original text?
No. In our implementation, FAISS stores the numerical vectors.
We maintain a separate metadata file containing the original text and page numbers.

Here's how the relationship works:
Example mapping
FAISS position
Metadata record


0
Chunk 1
Page 1


1
Chunk 2
Page 2


2
Chunk 3
Page 3


3
Chunk 4
Page 4

For example, suppose FAISS returns position 2.

We can retrieve:
metadata[2]

That gives us the corresponding chunk's original text and page information.
Important: This mapping works because we added embeddings and metadata in exactly the same order. We must preserve that order whenever rebuilding or updating the index.

a high similarity score doesn't guarantee that the chunk contains the complete answer.