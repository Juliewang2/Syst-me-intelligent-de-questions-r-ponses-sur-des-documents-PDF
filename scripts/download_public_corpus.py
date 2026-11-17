"""
scripts/download_public_corpus.py
-----------------------------------
Downloads the public documents listed in corpus/public_manifest.json
into data/corpus/public/ (PUBLIC_CORPUS_DIR), and writes each one's
metadata sidecar (<doc_id>.json) for scripts/ingest_corpus.py.

The manifest is the single source of truth for *which* public texts the
corpus contains and where each official copy lives; the files
themselves are never committed (UN / ICJ copyright, and size). Anyone
can rebuild the same corpus from the manifest.

Rules, so a run never silently ingests something unverified:
  - format other than "pdf"  -> skipped with a warning (only PDFs are
                                ingested; save HTML/TXT pages as PDF)
  - empty url                -> skipped with a warning
  - status other than "verified" -> skipped unless --include-unverified
  - a response that is not a PDF (%PDF- header) is rejected, not saved
Existing files are kept unless --force. Every sidecar is marked
source=public, classification=PUBLIC.

Usage (project root):
    venv\\Scripts\\python.exe scripts\\download_public_corpus.py
    venv\\Scripts\\python.exe scripts\\download_public_corpus.py --include-unverified --force
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import get_settings  # noqa: E402
from services.document_meta import DocumentMeta  # noqa: E402

settings = get_settings()

MANIFEST = ROOT / "corpus" / "public_manifest.json"
TIMEOUT_SECONDS = 60
# Generic client name: no personal details are sent to the remote sites.
USER_AGENT = "Mozilla/5.0 (compatible; corpus-downloader/1.0)"


def load_manifest(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["documents"]


def skip_reason(entry: dict, include_unverified: bool) -> Optional[str]:
    if entry.get("format", "pdf") != "pdf":
        return f"format {entry.get('format')!r} is not ingested (save it as PDF)"
    if not entry.get("url"):
        return "no url yet"
    if entry.get("status") != "verified" and not include_unverified:
        return f"status {entry.get('status')!r} (use --include-unverified once checked)"
    return None


def sidecar_for(entry: dict) -> DocumentMeta:
    """Manifest entry -> validated metadata, always public."""
    return DocumentMeta(
        doc_id=entry["doc_id"],
        title=entry.get("title"),
        doc_type=entry.get("doc_type"),
        source="public",
        classification="PUBLIC",
        language=entry.get("language"),
        date=entry.get("date"),
        un_symbol=entry.get("un_symbol"),
    )


def fetch_pdf(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
        data = response.read()
    if not data.startswith(b"%PDF-"):
        raise ValueError("response is not a PDF (an HTML page or an error?)")
    return data


def write_sidecar(pdf_path: Path, meta: DocumentMeta) -> None:
    data = meta.model_dump(mode="json", exclude_none=True)
    pdf_path.with_suffix(".json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def process(entry: dict, out_dir: Path, include_unverified: bool, force: bool) -> str:
    """Download one entry; returns a one-line status for the log."""
    name = entry.get("doc_id", "?")
    reason = skip_reason(entry, include_unverified)
    if reason:
        return f"SKIP  {name}: {reason}"
    try:
        meta = sidecar_for(entry)
    except ValueError as exc:
        return f"FAIL  {name}: invalid metadata - {exc}"

    pdf_path = out_dir / f"{meta.doc_id}.pdf"
    if pdf_path.exists() and not force:
        write_sidecar(pdf_path, meta)
        return f"KEEP  {name}: already downloaded (sidecar refreshed)"
    try:
        pdf_path.write_bytes(fetch_pdf(entry["url"]))
    except Exception as exc:  # noqa: BLE001 - one bad URL shouldn't stop the rest
        return f"FAIL  {name}: {exc}"
    write_sidecar(pdf_path, meta)
    return f"OK    {name}: {pdf_path.stat().st_size // 1024} KB"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the public corpus listed in the manifest.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--out", type=Path, default=settings.public_corpus_dir)
    parser.add_argument("--include-unverified", action="store_true", help="Also fetch status != verified")
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    lines = [process(e, args.out, args.include_unverified, args.force) for e in load_manifest(args.manifest)]
    for line in lines:
        print(line)
    failures = sum(line.startswith("FAIL") for line in lines)
    skipped = sum(line.startswith("SKIP") for line in lines)
    print(f"\n{len(lines)} entries: {failures} failed, {skipped} skipped. Files in {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
