"""
ClaimPilot ingestion: extract text from all PDFs in data/policies/
and save structured JSON per document.
"""

import json
import re
from pathlib import Path
from pypdf import PdfReader

ROOT = Path(__file__).parent
POLICIES_DIR = ROOT / "policies"
EXTRACTED_DIR = ROOT / "extracted"
EXTRACTED_DIR.mkdir(exist_ok=True)


def parse_filename(stem: str) -> dict:
    """
    Extract plan_name and doc_type from a filename like
    'my-optima-secure__wordings' or 'my-health-suraksha__brochure'.
    """
    parts = stem.split("__")
    plan_slug = parts[0]
    doc_type = parts[1] if len(parts) > 1 else "unknown"

    # 'my-optima-secure' -> 'My Optima Secure'
    plan_name = " ".join(word.capitalize() for word in plan_slug.split("-"))

    return {
        "plan_name": plan_name,
        "plan_slug": plan_slug,
        "doc_type": doc_type,  # 'wordings' or 'brochure'
    }


def clean_text(text: str) -> str:
    """Normalize whitespace, kill page-number artifacts."""
    if not text:
        return ""
    # collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    # strip lines that are only page numbers (e.g. "42", "Page 42", "- 42 -")
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if re.fullmatch(r"(?:page\s*)?[-–]?\s*\d{1,3}\s*[-–]?", stripped, re.IGNORECASE):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def extract_pdf(path: Path) -> dict:
    """Extract a single PDF into a structured dict."""
    reader = PdfReader(str(path))
    meta = parse_filename(path.stem)

    pages = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        if cleaned:
            pages.append({"page": i, "text": cleaned})

    return {
        "source_pdf": path.name,
        "plan_name": meta["plan_name"],
        "plan_slug": meta["plan_slug"],
        "doc_type": meta["doc_type"],
        "total_pages": len(reader.pages),
        "pages_with_text": len(pages),
        "pages": pages,
    }


def main():
    pdfs = sorted(POLICIES_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {POLICIES_DIR}")
        return

    total_chars = 0
    for pdf in pdfs:
        print(f"Processing {pdf.name}...")
        doc = extract_pdf(pdf)
        out_path = EXTRACTED_DIR / f"{pdf.stem}.json"
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

        doc_chars = sum(len(p["text"]) for p in doc["pages"])
        total_chars += doc_chars
        print(f"  -> {doc['pages_with_text']}/{doc['total_pages']} pages, {doc_chars:,} chars")

    print()
    print(f"Done. {len(pdfs)} documents ingested.")
    print(f"Total text: {total_chars:,} chars (~{total_chars // 4:,} tokens)")


if __name__ == "__main__":
    main()