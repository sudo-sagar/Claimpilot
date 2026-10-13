"""
Quick stats on the extracted corpus.
"""
import json
from pathlib import Path

EXTRACTED_DIR = Path(__file__).parent / "extracted"

total_chars = 0
print(f"{'Document':<54} {'Chars':>12}")
print("-" * 68)

for p in sorted(EXTRACTED_DIR.glob("*.json")):
    doc = json.loads(p.read_text(encoding="utf-8"))
    chars = sum(len(pg["text"]) for pg in doc["pages"])
    total_chars += chars
    name = doc["source_pdf"][:52]
    print(f"{name:<54} {chars:>12,}")

print("-" * 68)
print(f"{'TOTAL CHARS':<54} {total_chars:>12,}")
print(f"{'APPROX TOKENS (chars / 4)':<54} {total_chars // 4:>12,}")