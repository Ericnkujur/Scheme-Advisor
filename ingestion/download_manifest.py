"""
Download scheme guideline (and optional FAQ) PDFs listed in a manifest,
naming them consistently and tracking what's been collected so far.

Why manifest-driven: scholarships.gov.in's scheme list is rendered
client-side (JS), so it isn't reliably scrapable with requests/BeautifulSoup.
The "Guidelines" and "FAQs" links themselves are direct PDF URLs though —
so the practical v1 approach is: browse the filter page yourself,
right-click -> copy link on each one you want, paste them into
data/scheme_manifest.json, then run this script to fetch + track them all
in one place.

FAQs are worth collecting alongside guidelines: they often state eligibility
in plainer, more concrete terms (exact amounts, quotas, renewal conditions)
than the formal guideline document, which helps both rule extraction and
retrieval quality.

data/scheme_manifest.json format:
[
  {
    "scheme_name": "Prime Minister's Scholarship Scheme For CAPF And Assam Rifles",
    "state": null,
    "guideline_url": "https://scholarships.gov.in/.../guidelines.pdf",
    "faq_url": "https://scholarships.gov.in/public/FAQ/FAQonPMSSS.pdf",
    "application_deadline": "2026-10-31"
  },
  ...
]

"faq_url" is optional — omit it or leave null if a scheme has no separate FAQ doc.

Usage:
    python ingestion/download_manifest.py --manifest data/scheme_manifest.json
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import requests

RAW_PDF_DIR = Path("data/raw_pdfs")
STATUS_FILE = Path("data/collection_status.json")


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_")
    return slug[:80]


def load_manifest(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_status() -> dict:
    if STATUS_FILE.exists():
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_status(status: dict) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)


def download_file(url: str, out_path: Path) -> str:
    """Returns 'ok' or an error message."""
    try:
        resp = requests.get(url, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (compatible; JanSahayakCollector/1.0)"
        })
        resp.raise_for_status()
        if "pdf" not in resp.headers.get("Content-Type", "").lower() and not resp.content.startswith(b"%PDF"):
            return f"error: response doesn't look like a PDF (content-type: {resp.headers.get('Content-Type')})"
        out_path.write_bytes(resp.content)
        return "ok"
    except Exception as e:
        return f"error: {e}"


def download_entry(entry: dict, out_dir: Path) -> dict:
    """Downloads guideline (required) and FAQ (optional) for one scheme.
    Returns a dict of doc_type -> {filename, result}."""
    slug = slugify(entry["scheme_name"])
    results = {}

    guideline_filename = f"{slug}__guideline.pdf"
    guideline_result = download_file(entry["guideline_url"], out_dir / guideline_filename)
    results["guideline"] = {"filename": guideline_filename, "result": guideline_result}

    faq_url = entry.get("faq_url")
    if faq_url:
        faq_filename = f"{slug}__faq.pdf"
        faq_result = download_file(faq_url, out_dir / faq_filename)
        results["faq"] = {"filename": faq_filename, "result": faq_result}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", default=str(RAW_PDF_DIR))
    parser.add_argument("--force", action="store_true", help="Re-download even if already marked ok")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    status = load_status()

    for entry in manifest:
        name = entry["scheme_name"]
        if not args.force and status.get(name, {}).get("status") == "ok":
            print(f"skip (already collected): {name}")
            continue

        doc_results = download_entry(entry, out_dir)

        all_ok = all(d["result"] == "ok" for d in doc_results.values())
        status[name] = {
            "status": "ok" if all_ok else "partial_or_failed",
            "documents": doc_results,
            "state": entry.get("state"),
            "guideline_url": entry.get("guideline_url"),
            "faq_url": entry.get("faq_url"),
            "application_deadline": entry.get("application_deadline"),
        }

        for doc_type, d in doc_results.items():
            marker = "OK " if d["result"] == "ok" else "FAIL"
            print(f"[{marker}] {name} ({doc_type}) -> {d['filename']} ({d['result']})")

    save_status(status)
    ok_count = sum(1 for v in status.values() if v["status"] == "ok")
    print(f"\n{ok_count}/{len(manifest)} schemes fully collected. Status file: {STATUS_FILE}")


if __name__ == "__main__":
    main()

