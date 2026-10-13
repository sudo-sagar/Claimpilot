"""
ClaimPilot embeddings.

Embeds every chunk in data/chunks.jsonl with OpenAI's
text-embedding-3-small and saves to data/embeddings.npz.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
EMBEDDINGS_PATH = ROOT / "data" / "embeddings.npz"

EMBED_MODEL = "nvidia/nemotron-3-embed-1b:free"
EMBED_DIM = 1536
BATCH_SIZE = 100

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",  # <-- The critical line
    api_key=os.environ["API_KEY"]       # Your sk-or-v1-... key
)


def load_chunks():
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def embed_batch(texts):
    """Embed a batch of texts. Retries on rate limit."""
    for attempt in range(3):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            return [item.embedding for item in resp.data]
        except Exception as e:
            if attempt == 2:
                raise
            print("  retry after error: " + str(e)[:80])
            time.sleep(2 ** attempt)
    return []


def build_embeddings(chunks):
    """Embed all chunks in batches, with progress."""
    vectors = []
    total = len(chunks)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = [c["text"] for c in chunks[start:end]]
        vecs = embed_batch(batch)
        vectors.extend(vecs)
        print("  embedded " + str(end) + "/" + str(total))

    return np.array(vectors, dtype=np.float32)


def main():
    if not CHUNKS_PATH.exists():
        print("ERROR: " + str(CHUNKS_PATH) + " not found. Run app/chunking.py first.")
        return

    chunks = load_chunks()
    print("Loaded " + str(len(chunks)) + " chunks")

    # Cache check
    if EMBEDDINGS_PATH.exists():
        data = np.load(EMBEDDINGS_PATH)
        if data["vectors"].shape[0] == len(chunks):
            print("Embeddings cache hit: " + str(EMBEDDINGS_PATH))
            print("  shape: " + str(data["vectors"].shape))
            return
        print("Cache stale (chunk count changed), rebuilding...")

    print("Embedding with " + EMBED_MODEL + "...")
    t0 = time.time()
    vectors = build_embeddings(chunks)
    elapsed = time.time() - t0

    # Normalize for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    np.savez_compressed(EMBEDDINGS_PATH, vectors=vectors)

    print("")
    print("=" * 60)
    print("Chunks embedded:  " + str(len(chunks)))
    print("Vector shape:     " + str(vectors.shape))
    print("Time:             " + str(round(elapsed, 1)) + "s")
    print("Output:           " + str(EMBEDDINGS_PATH))
    print("File size:        " + str(round(EMBEDDINGS_PATH.stat().st_size / 1024 / 1024, 2)) + " MB")


if __name__ == "__main__":
    main()