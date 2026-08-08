"""
Embed chunk JSON files and store them in a local (on-disk) Chroma collection.

Uses BAAI/bge-small-en-v1.5 — small, fast, free, runs on CPU. Good enough
for v1's English-only, ~15-20 document scale. Swap in a bigger model or
BGE-M3 (multilingual) later without touching any other file, since
retrieve.py only depends on the collection, not on how it was built.

Usage:
    python embed.py --chunks data/processed/scheme_001_chunks.json data/processed/scheme_002_chunks.json \
                     --db-path data/chroma_db
"""
from __future__ import annotations
import argparse
import json
from glob import glob
import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "jansahayak_chunks"


def load_chunks(paths: list[str]) -> list[dict]:
    chunks = []

    for path in paths:
        matched_paths = glob(path)

        if not matched_paths:
            print(f"Warning: no files matched: {path}")
            continue

        for p in matched_paths:
            print(f"Loading: {p}")

            with open(p, "r", encoding="utf-8") as f:
                chunks.extend(json.load(f))

    return chunks


def build_index(chunk_paths: list[str], db_path: str) -> None:
    chunks = load_chunks(chunk_paths)
    if not chunks:
        print("No chunks found — nothing to embed.")
        return

    print(f"Loading embedding model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    client = chromadb.PersistentClient(path=db_path)
    # Fresh collection each run for v1 simplicity — re-embed everything when
    # source documents change, rather than doing incremental upserts.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[
            {
                "scheme_name": c["scheme_name"],
                "scheme_slug": c["scheme_slug"],
                "source_document": c["source_document"],
                "chunk_index": c["chunk_index"],
            }
            for c in chunks
        ],
    )
    print(f"Indexed {len(chunks)} chunks into '{COLLECTION_NAME}' at {db_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", nargs="+", required=True, help="One or more chunk JSON files")
    parser.add_argument("--db-path", default="data/chroma_db")
    args = parser.parse_args()

    build_index(args.chunks, args.db_path)


if __name__ == "__main__":
    main()
