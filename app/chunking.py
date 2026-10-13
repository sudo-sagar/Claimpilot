"""
ClaimPilot chunking.

Reads extracted PDF JSONs from data/extracted/ and produces
section-aware chunks saved to data/chunks.jsonl (one JSON per line).
"""

import json
import re
from pathlib import Path
from collections import Counter

# ---- Paths ----
ROOT = Path(__file__).parent.parent
EXTRACTED_DIR = ROOT / "data" / "extracted"
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"

# ---- Chunking params ----
TARGET_CHUNK_CHARS = 2400
OVERLAP_CHARS = 400
MIN_CHUNK_CHARS = 200

# ---- Section heading detector ----
SECTION_PATTERN = re.compile(
    r"^\s*(?:\d+[\.\)]?\s*)?"
    r"(DEFINITIONS?|COVERAGE|EXCLUSIONS?|WAITING\s+PERIOD|"
    r"CLAIM\s+PROCEDURE|PRE[-\s]?EXISTING(?:\s+DISEASES?)?|"
    r"BENEFITS?|SCOPE\s+OF\s+COVERAGE|"
    r"TERMS\s+AND\s+CONDITIONS|GENERAL\s+CONDITIONS|"
    r"EXCLUSIONS?\s+AND\s+LIMITATIONS|"
    r"SUM\s+INSURED|ELIGIBILITY|RENEWAL|CANCELLATION)"
    r"\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def split_into_sections(text):
    """Split text into (section_name, section_text) tuples."""
    matches = list(SECTION_PATTERN.finditer(text))

    if not matches:
        return [("Unstructured", text)]

    sections = []

    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("Preamble", preamble))

    for i, m in enumerate(matches):
        section_name = m.group(1).strip().upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((section_name, body))

    return sections


def chunk_text(text):
    """Split a section into overlapping chunks."""
    text = text.strip()
    if len(text) <= TARGET_CHUNK_CHARS:
        return [text] if len(text) >= MIN_CHUNK_CHARS else []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + TARGET_CHUNK_CHARS, len(text))

        if end < len(text):
            window = text[max(0, end - 200): end]
            for sep in ["\n\n", ". ", "\n"]:
                pos = window.rfind(sep)
                if pos != -1:
                    end = max(0, end - 200) + pos + len(sep)
                    break

        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(0, end - OVERLAP_CHARS)

    return chunks


def process_document(doc):
    """Turn one extracted document into a list of chunk records."""
    source = doc["source_pdf"]
    plan = doc["plan_name"]
    doc_type = doc["doc_type"]

    chunks = []
    chunk_idx = 0

    for page in doc["pages"]:
        page_num = page["page"]
        page_text = page["text"]

        for section_name, section_text in split_into_sections(page_text):
            for sub_chunk in chunk_text(section_text):
                chunks.append(
                    {
                        "chunk_id": source + "::p" + str(page_num) + "::c" + str(chunk_idx),
                        "text": sub_chunk,
                        "source_pdf": source,
                        "plan_name": plan,
                        "doc_type": doc_type,
                        "page": page_num,
                        "section": section_name,
                        "char_count": len(sub_chunk),
                    }
                )
                chunk_idx += 1

    return chunks


def main():
    if not EXTRACTED_DIR.exists():
        print("ERROR: " + str(EXTRACTED_DIR) + " not found. Run data/ingest.py first.")
        return

    docs = sorted(EXTRACTED_DIR.glob("*.json"))
    if not docs:
        print("ERROR: no JSON files in " + str(EXTRACTED_DIR))
        return

    all_chunks = []

    for doc_path in docs:
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        chunks = process_document(doc)
        all_chunks.extend(chunks)
        print(doc_path.name + " -> " + str(len(chunks)) + " chunks")

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    total_chars = sum(c["char_count"] for c in all_chunks)
    avg_chars = total_chars // len(all_chunks) if all_chunks else 0

    print("")
    print("=" * 68)
    print("Total documents:  " + str(len(docs)))
    print("Total chunks:     " + str(len(all_chunks)))
    print("Total chars:      " + str(total_chars))
    print("Avg chunk size:   " + str(avg_chars) + " chars")
    print("Output file:      " + str(CHUNKS_PATH))

    sections = Counter(c["section"] for c in all_chunks)
    print("")
    print("Top sections:")
    for section, count in sections.most_common(10):
        print("  " + section.ljust(40) + " " + str(count))

    doc_types = Counter(c["doc_type"] for c in all_chunks)
    print("")
    print("By doc type:")
    for dt, count in doc_types.items():
        print("  " + dt.ljust(40) + " " + str(count))


if __name__ == "__main__":
    main()