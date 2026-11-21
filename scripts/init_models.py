"""Download the local models (embeddings, reranker) so the first run has no surprises."""
import os

from sentence_transformers import CrossEncoder, SentenceTransformer

EMBEDDING = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER = "BAAI/bge-reranker-v2-m3"

if __name__ == "__main__":
    print(f"Downloading: {EMBEDDING}")
    SentenceTransformer(EMBEDDING)
    print(f"Downloading: {RERANKER}")
    CrossEncoder(RERANKER)
    print("Models ready.")
