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

# ---- Boilerplate patterns (headers/footers/legal/addresses) ----
BOILERPLATE_PATTERNS = [
    # Legal / regulatory
    re.compile(r"IRDAI\s+Reg", re.IGNORECASE),
    re.compile(r"CIN\s*:", re.IGNORECASE),
    re.compile(r"Reg\.?\s+No\.?\s*\d", re.IGNORECASE),
    re.compile(r"UIN\s*:", re.IGNORECASE),
    re.compile(r"HDFHL[A-Z0-9]+", re.IGNORECASE),

    # Contact
    re.compile(r"Toll\s+Free\s+[\d\-]+", re.IGNORECASE),
    re.compile(r"www\.[a-z0-9\-\.]+", re.IGNORECASE),
    re.compile(r"customerservice@", re.IGNORECASE),

    # Company names in headers
    re.compile(r"^HDFC\s+ERGO\s+General\s+Insurance", re.IGNORECASE),
    re.compile(r"^Apollo\s+Munich", re.IGNORECASE),
    re.compile(r"^Star\s+Health", re.IGNORECASE),
    re.compile(r"Registered\s+&\s+Corporate\s+Office", re.IGNORECASE),

    # Bare page numbers
    re.compile(r"^\s*\d+\s*$"),

    # ---- NEW: address & corporate boilerplate ----
    re.compile(r"Andheri[-\s]?Kurla", re.IGNORECASE),
    re.compile(r"Leela\s+Business\s+Park", re.IGNORECASE),
    re.compile(r"Backbay\s+Reclamation", re.IGNORECASE),
    re.compile(r"H\.?\s*T\.?\s*Parekh\s+Marg", re.IGNORECASE),
    re.compile(r"Mumbai\s*[–\-]\s*\d{6}", re.IGNORECASE),
    re.compile(r"\b\d{6}\b\s*$"),                                # trailing pincode
    re.compile(r"Trade\s+Logo\s+displayed", re.IGNORECASE),
    re.compile(r"Prospectus\s+my\s*:", re.IGNORECASE),
    re.compile(r"Policy\s+Wording\s+my\s*:", re.IGNORECASE),
    re.compile(r"Policy\s+Wording\s+Optima", re.IGNORECASE),
    re.compile(r"Policy\s+Wording\s+HDFC", re.IGNORECASE),
    re.compile(r"please\s+read\s+the\s+policy\s+document", re.IGNORECASE),
    re.compile(r"concluding\s+a\s+sale", re.IGNORECASE),
    re.compile(r"Floor\s*,", re.IGNORECASE),
    re.compile(r"^\s*(?:Road|Street|Marg|Lane)\s*,", re.IGNORECASE),
]


def strip_boilerplate(text: str) -> str:
    """Remove header/footer/contact/address lines that pollute retrieval."""
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if len(stripped) < 5:
            continue
        if any(p.search(stripped) for p in BOILERPLATE_PATTERNS):
            continue
        lines.append(line)
    return "\n".join(lines)


# ---- Section heading detector ----
# Loosened: matches headings inline, not just on their own line.
SECTION_PATTERN = re.compile(
    r"(?:\b\d+[\.\)]?\s*)?"
    r"\b("
    r"DEFINITIONS?"
    r"|COVERAGE"
    r"|EXCLUSIONS?(?:\s+AND\s+LIMITATIONS)?"
    r"|WAITING\s+PERIODS?"
    r"|CLAIM\s+PROCEDURE"
    r"|PRE[-\s]?EXISTING(?:\s+DISEASES?)?"
    r"|BENEFITS?"
    r"|SCOPE\s+OF\s+COVERAGE"
    r"|TERMS\s+AND\s+CONDITIONS"
    r"|GENERAL\s+CONDITIONS"
    r"|SUM\s+INSURED"
    r"|ELIGIBILITY"
    r"|RENEWAL"
    r"|CANCELLATION"
    r")\b",
    re.IGNORECASE,
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
    """Split a section into overlapping chunks, respecting sentence boundaries."""
    text = text.strip()
    if len(text) <= TARGET_CHUNK_CHARS:
        return [text] if len(text) >= MIN_CHUNK_CHARS else []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + TARGET_CHUNK_CHARS, len(text))

        if end < len(text):
            # Look backwards for a sentence boundary
            window = text[max(0, end - 300): end]
            for sep in [". ", ".\n", "\n\n", "\n"]:
                pos = window.rfind(sep)
                if pos != -1:
                    end = max(0, end - 300) + pos + len(sep)
                    break

        chunk = text[start:end].strip()

        # Snap chunk start forward to first sentence boundary
        # (only if we're not at the very beginning of the section)
        if start > 0:
            first_period = chunk.find(". ")
            if 0 < first_period < 200:
                chunk = chunk[first_period + 2:].strip()

        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(0, end - OVERLAP_CHARS)

    return chunks


def is_low_quality_chunk(text: str) -> bool:
    """
    Reject chunks that are mostly headers/numbers/non-sentence text.
    Relaxed thresholds so legitimate short clauses survive.
    """
    if len(text) < 80:                              # was 100
        return True
    # Ratio of alphabetic chars
    alpha = sum(c.isalpha() for c in text)
    if alpha / max(len(text), 1) < 0.35:            # was 0.5
        return True
    # Note: punctuation requirement removed — insurance clauses often
    # are bullet-like without full stops.
    return False


def process_document(doc):
    """Turn one extracted document into a list of chunk records."""
    source = doc["source_pdf"]
    plan = doc["plan_name"]
    doc_type = doc["doc_type"]

    chunks = []
    chunk_idx = 0

    for page in doc["pages"]:
        page_num = page["page"]
        page_text = strip_boilerplate(page["text"])

        for section_name, section_text in split_into_sections(page_text):
            for sub_chunk in chunk_text(section_text):
                if is_low_quality_chunk(sub_chunk):
                    continue
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