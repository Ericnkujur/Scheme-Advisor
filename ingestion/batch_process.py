"""
Run the full per-scheme pipeline (extract text -> extract rules -> chunk)
over everything already downloaded via download_manifest.py, using
data/collection_status.json to know which PDFs belong to which scheme.

For each scheme:
  1. Extract text from guideline PDF (and FAQ PDF, if present)
  2. Call the LLM once to extract structured rules from the COMBINED
     guideline + FAQ text (richer signal -> better extraction)
  3. Chunk the guideline and FAQ text SEPARATELY for retrieval, so
     citations can point to the right document

Skips schemes that already have a rules JSON, unless --force is passed.
Continues past individual failures (rather than aborting the whole batch)
and prints a summary at the end — with 33 PDFs, some LLM extractions may
need a retry or manual fix, and one bad PDF shouldn't block the other 32.

Usage:
    python ingestion/batch_process.py
    python ingestion/batch_process.py --force
    python ingestion/batch_process.py --limit 5   # test on a few first
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rules"))

from ingestion.extract_pdf import extract_text
from ingestion.chunk import chunk_document
from rules.extract_rules import extract_rule

RAW_PDF_DIR = Path("data/raw_pdfs")
PROCESSED_DIR = Path("data/processed")
RULES_DIR = Path("data/rules")
STATUS_FILE = Path("data/collection_status.json")


def load_status() -> dict:
    with open(STATUS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_rule_with_retry(combined_text: str, source_document: str, max_retries: int = 3):
    """Wraps extract_rule with backoff on transient errors (rate limit,
    server overload). Daily quota exhaustion is NOT retried — retrying just
    burns more of an already-exhausted daily allowance for nothing; instead
    it fails immediately so you can stop the batch and resume tomorrow."""
    import datetime
    delay = 15
    last_error = None
    for attempt in range(max_retries):
        try:
            return extract_rule(
                raw_text=combined_text,
                source_document=source_document,
                last_verified_date=datetime.date.today().isoformat(),
            )
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "413" in err_str or "too large" in err_str.lower():
                # shrink further and retry immediately, no need to wait
                text = text[: len(text) // 2]
                print(f"    request too large, truncating further and retrying ({len(text)} chars)...")
                continue
            if "rate_limit" in err_str.lower() or "429" in err_str or "503" in err_str or "overloaded" in err_str.lower():
                print(f"    transient error (rate limit or server overload), waiting {delay}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(delay)
                delay *= 2
            else:
                raise
    raise last_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Reprocess schemes that already have rules")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N schemes (for testing)")
    parser.add_argument("--delay", type=float, default=6.0, help="Seconds to wait between schemes (paces requests to stay under free-tier RPM limits)")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RULES_DIR.mkdir(parents=True, exist_ok=True)

    status = load_status()
    schemes = list(status.items())
    if args.limit:
        schemes = schemes[: args.limit]

    results = {"ok": [], "skipped": [], "failed": []}

    for scheme_name, entry in schemes:
        if entry.get("status") not in ("ok", "partial_or_failed"):
            continue

        docs = entry.get("documents", {})
        guideline_info = docs.get("guideline")
        if not guideline_info or guideline_info.get("result") != "ok":
            results["failed"].append((scheme_name, "no successfully downloaded guideline PDF"))
            print(f"[FAIL] {scheme_name}: no guideline PDF available")
            continue

        import re
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", scheme_name.strip()).strip("_")[:80]

        rules_out_path = RULES_DIR / f"{slug}.json"
        chunks_out_path = PROCESSED_DIR / f"{slug}_chunks.json"

        need_rules = args.force or not rules_out_path.exists()
        need_chunks = args.force or not chunks_out_path.exists()

        if not need_rules and not need_chunks:
            results["skipped"].append(scheme_name)
            print(f"[SKIP] {scheme_name} (already fully processed)")
            continue

        try:
            # --- 1. Extract text ---
            guideline_pdf = RAW_PDF_DIR / guideline_info["filename"]
            guideline_text = extract_text(str(guideline_pdf))
            guideline_txt_path = PROCESSED_DIR / f"{slug}__guideline.txt"
            guideline_txt_path.write_text(guideline_text, encoding="utf-8")

            combined_text = guideline_text
            faq_text = None
            faq_info = docs.get("faq")
            if faq_info and faq_info.get("result") == "ok":
                faq_pdf = RAW_PDF_DIR / faq_info["filename"]
                faq_text = extract_text(str(faq_pdf))
                faq_txt_path = PROCESSED_DIR / f"{slug}__faq.txt"
                faq_txt_path.write_text(faq_text, encoding="utf-8")
                combined_text = guideline_text + "\n\n--- FAQ ---\n\n" + faq_text

            # --- 2. Extract structured rules (LLM, combined text for richer signal) ---
            if need_rules:
                rule = _extract_rule_with_retry(combined_text, guideline_info["filename"])
                rules_out_path.write_text(rule.model_dump_json(indent=2), encoding="utf-8")
            else:
                from rules.schema import EligibilityRule
                rule = EligibilityRule.model_validate_json(rules_out_path.read_text(encoding="utf-8"))

            # --- 3. Chunk separately for retrieval ---
            all_chunks = chunk_document(guideline_text, scheme_name, slug, guideline_info["filename"])
            if faq_text:
                all_chunks += chunk_document(faq_text, scheme_name, slug, faq_info["filename"])

            from dataclasses import asdict
            chunks_out_path = PROCESSED_DIR / f"{slug}_chunks.json"
            chunks_out_path.write_text(
                json.dumps([asdict(c) for c in all_chunks], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            results["ok"].append(scheme_name)
            print(f"[OK]   {scheme_name} -> {rules_out_path.name}, {len(all_chunks)} chunks")

        except Exception as e:
            results["failed"].append((scheme_name, str(e)))
            print(f"[FAIL] {scheme_name}: {e}")
            if "PerDay" in str(e) or "GenerateRequestsPerDay" in str(e):
                print("\nDaily free-tier quota exhausted — stopping here. "
                      "Rerun this script after the quota resets (~24h from your first call today); "
                      "already-completed schemes will be skipped automatically.")
                break

        time.sleep(args.delay)

    print("\n--- Summary ---")
    print(f"OK: {len(results['ok'])}  Skipped: {len(results['skipped'])}  Failed: {len(results['failed'])}")
    if results["failed"]:
        print("\nFailed schemes (fix and rerun with --force, or process individually):")
        for name, err in results["failed"]:
            print(f"  - {name}: {err}")


if __name__ == "__main__":
    main()
