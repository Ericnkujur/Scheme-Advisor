"""
Split extracted scheme text into overlapping chunks for embedding.

Simple paragraph-aware splitter — no need for anything fancier at v1 scale
(15-20 documents). Each chunk carries metadata so retrieval results can be
traced back to a scheme and a rough location in the source document, which
is what the citation step in generation/answer.py needs.

Usage:
    python chunk.py --input data/processed/scheme_001.txt \
                     --scheme-name "Haryana Post-Matric Scholarship" \
                     --source-doc "Guidelines_Haryana_PostMatric_2026.pdf" \
                     --out data/processed/scheme_001_chunks.json
"""
from __future__ import annotations
import argparse
import json
import re
from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    chunk_id: str
    text: str
    scheme_name: str
    scheme_slug: str  
    source_document: str
    chunk_index: int


def split_into_chunks(
    text: str,
    max_chars: int = 1500,
    overlap_chars: int = 200,
) -> list[str]:
    """
    Paragraph-first splitting: join paragraphs into chunks up to max_chars,
    falling back to a sliding window with overlap if a single paragraph is
    longer than max_chars on its own.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > max_chars:
            # flush current, then window this long paragraph on its own
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(para):
                end = start + max_chars
                chunks.append(para[start:end])
                start = end - overlap_chars
            continue

        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > max_chars:
            chunks.append(current)
            current = para
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def chunk_document(text: str, scheme_name: str, scheme_slug: str, source_document: str) -> list[Chunk]:
    raw_chunks = split_into_chunks(text)
    return [
        Chunk(
            chunk_id=f"{source_document}::{i}",
            text=raw,
            scheme_name=scheme_name,
            scheme_slug=scheme_slug,
            source_document=source_document,
            chunk_index=i,
        )
        for i, raw in enumerate(raw_chunks)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--scheme-name", required=True)
    parser.add_argument("--source-doc", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_document(text, args.scheme_name, args.source_doc)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, indent=2, ensure_ascii=False)

    print(f"{len(chunks)} chunks -> {args.out}")


if __name__ == "__main__":
    main()
