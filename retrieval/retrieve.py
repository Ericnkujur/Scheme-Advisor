"""
Retrieve the top-k most relevant chunks for a free-text query.

Uses the same embedding model as embed.py — must match, since query and
document embeddings need to live in the same vector space.

Usage (CLI smoke test):
    python retrieve.py --query "scholarship for OBC student in Haryana" --db-path data/chroma_db
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass

import chromadb
from sentence_transformers import SentenceTransformer

try:
    from embed import MODEL_NAME, COLLECTION_NAME
except ImportError:
    from retrieval.embed import MODEL_NAME, COLLECTION_NAME

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    scheme_name: str
    scheme_slug: str
    source_document: str
    distance: float


def retrieve(query: str, db_path: str, top_k: int = 5) -> list[RetrievedChunk]:
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(COLLECTION_NAME)

    model = _get_model()
    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        retrieved.append(
            RetrievedChunk(
                chunk_id=results["ids"][0][i],
                text=results["documents"][0][i],
                scheme_name=meta["scheme_name"],
                scheme_slug=meta["scheme_slug"],
                source_document=meta["source_document"],
                distance=results["distances"][0][i],
            )
        )
    return retrieved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--db-path", default="data/chroma_db")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    results = retrieve(args.query, args.db_path, args.top_k)
    for r in results:
        print(f"[{r.distance:.4f}] {r.scheme_name} ({r.source_document})")
        print(f"  {r.text[:200]}...")
        print()


if __name__ == "__main__":
    main()
