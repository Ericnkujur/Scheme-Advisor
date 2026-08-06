"""
Extract clean text from a scheme guideline PDF.

Usage:
    python extract_pdf.py --input data/raw_pdfs/scheme_001.pdf \
                           --out data/processed/scheme_001.txt
"""
from __future__ import annotations
import argparse
import fitz  # PyMuPDF


def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    text = extract_text(args.input)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Extracted {len(text)} chars -> {args.out}")


if __name__ == "__main__":
    main()
