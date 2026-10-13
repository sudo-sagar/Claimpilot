"""
ClaimPilot hybrid retrieval.

Dense (embeddings) + sparse (BM25) search with:
  - Score fusion (normalized weighted score combination), OR
  - RRF (Reciprocal Rank Fusion)

Normalization for score fusion:
  - "rankdecay" : rank 1 -> 1.0, rank 2 -> 0.7, rank 3 -> 0.5, ...
  - "divmax"    : divide each score by max  -> top hit = 1.0
  - "minmax"    : (s - lo) / (hi - lo)      -> top hit = 1.0, bottom = 0.0

Switch via FUSION_MODE / NORMALIZATION_MODE below.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

load_dotenv()

ROOT = Path(__file__).parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
EMBEDDINGS_PATH = ROOT / "data" / "embeddings.npz"

EMBED_MODEL = "nvidia/nemotron-3-embed-1b:free"

# ---- Fusion configuration ----
FUSION_MODE = "score"            # "score" or "rrf"
NORMALIZATION_MODE = "rankdecay" # "rankdecay" | "divmax" | "minmax"  (score fusion only)

RRF_K = 200                      # used only when FUSION_MODE == "rrf"

# Tuned weights: dense and sparse balanced; rankdecay prevents BM25's
# top-rank generosity from dominating the fusion.
DENSE_WEIGHT = 1.0
SPARSE_WEIGHT = 1.0

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",   # ✅ points at OpenRouter
    api_key=os.environ["API_KEY"],
)

# ---- Domain synonyms for insurance queries ----
SYNONYMS = {
    "ped": ["pre", "existing", "disease"],
    "preexisting": ["pre", "existing"],
    "pre": ["pre"],
    "waiting": ["wait", "waiting"],
    "wait": ["waiting"],
    "maternity": ["maternity", "pregnancy"],
    "copay": ["co", "pay", "copayment"],
    "deductible": ["deductible", "excess"],
    "room": ["room", "rent"],
    "icu": ["icu", "intensive", "care"],
    "ayush": ["ayush", "ayurveda", "homeopathy"],
    "sum": ["sum", "insured"],
    "si": ["sum", "insured"],
    "definition": ["define", "means", "meaning"],
    "define": ["definition", "means"],
    "means": ["definition", "define"],
    "hospital": ["hospital", "inpatient", "in-patient", "institution"],
    "claim": ["claim", "reimburse", "reimbursement"],
    "excluded": ["excluded", "exclusion", "not", "covered"],

}


class Retriever:
    def __init__(self):
        # Load chunks
        with CHUNKS_PATH.open("r", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]

        # Load embeddings
        data = np.load(EMBEDDINGS_PATH)
        self.vectors = data["vectors"]  # already L2-normalized

        # Build BM25 index
        tokenized = [self._tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

        print("Retriever ready: " + str(len(self.chunks)) + " chunks")
        print("Fusion: " + FUSION_MODE
              + "  |  norm: " + NORMALIZATION_MODE
              + "  |  weights dense=" + str(DENSE_WEIGHT)
              + ", sparse=" + str(SPARSE_WEIGHT))

    # ---------- Tokenization ----------

    @staticmethod
    def _tokenize(text):
        """
        Lowercase, normalize hyphens/slashes to spaces, strip punctuation,
        then expand domain synonyms.
        """
        text = text.lower()
        text = re.sub(r"[-/]", " ", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = [t for t in text.split() if len(t) > 1]

        expanded = []
        for t in tokens:
            expanded.append(t)
            if t in SYNONYMS:
                expanded.extend(SYNONYMS[t])
        return expanded

    # ---------- Retrieval primitives ----------

    def _embed_query(self, query):
        resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _dense_search(self, query, top_k):
        qvec = self._embed_query(query)
        scores = self.vectors @ qvec  # cosine (vectors are normalized)
        idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in idx]

    def _sparse_search(self, query, top_k):
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in idx]

    # ---------- Normalization ----------

    @staticmethod
    def _normalize_scores(rank_list):
        """
        Normalize scores in a rank list to [0, 1].
        Mode is controlled by NORMALIZATION_MODE.
        """
        if not rank_list:
            return []

        if NORMALIZATION_MODE == "rankdecay":
            # Rank 1 -> 1.0, rank 2 -> 0.7, rank 3 -> 0.5, ...
            # Penalizes lower ranks much more than divmax.
            decay = [1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03]
            out = []
            for rank, (i, _s) in enumerate(rank_list):
                if rank < len(decay):
                    out.append((i, decay[rank]))
                else:
                    out.append((i, 0.02))
            return out

        elif NORMALIZATION_MODE == "divmax":
            max_score = max(s for _, s in rank_list)
            if max_score <= 0:
                return [(i, 0.0) for i, _ in rank_list]
            return [(i, s / max_score) for i, s in rank_list]

        elif NORMALIZATION_MODE == "minmax":
            scores = [s for _, s in rank_list]
            lo, hi = min(scores), max(scores)
            if hi == lo:
                return [(i, 1.0) for i, _ in rank_list]
            return [(i, (s - lo) / (hi - lo)) for i, s in rank_list]

        else:
            raise ValueError("Unknown NORMALIZATION_MODE: " + str(NORMALIZATION_MODE))

    # ---------- Fusion ----------

    @staticmethod
    def _rrf(rank_lists, weights=None, k=RRF_K):
        """Weighted Reciprocal Rank Fusion (rank-based)."""
        if weights is None:
            weights = [1.0] * len(rank_lists)
        fused = {}
        for weight, rank_list in zip(weights, rank_lists):
            for rank, (idx, _score) in enumerate(rank_list):
                fused[idx] = fused.get(idx, 0.0) + weight * (1.0 / (k + rank + 1))
        return sorted(fused.items(), key=lambda x: -x[1])

    def _score_fusion(self, dense, sparse,
                      weights=(DENSE_WEIGHT, SPARSE_WEIGHT)):
        """
        Weighted score fusion.
        1. Normalize each ranker's scores to [0, 1].
        2. Combine: fused = w_dense * nd + w_sparse * ns
        """
        norm_dense = self._normalize_scores(dense)
        norm_sparse = self._normalize_scores(sparse)

        fused = {}
        for w, rl in zip(weights, [norm_dense, norm_sparse]):
            for idx, score in rl:
                fused[idx] = fused.get(idx, 0.0) + w * score
        return sorted(fused.items(), key=lambda x: -x[1])

    def _fuse(self, dense, sparse):
        """Dispatch to the configured fusion method."""
        if FUSION_MODE == "rrf":
            return self._rrf([dense, sparse],
                             weights=[DENSE_WEIGHT, SPARSE_WEIGHT])
        elif FUSION_MODE == "score":
            return self._score_fusion(dense, sparse)
        else:
            raise ValueError("Unknown FUSION_MODE: " + str(FUSION_MODE))

    # ---------- Filtering ----------

    def _passes_filter(self, chunk, filters):
        if not filters:
            return True
        for key, value in filters.items():
            if value is None:
                continue
            if chunk.get(key) != value:
                return False
        return True

    # ---------- Public API ----------

    def search(self, query, top_k=5, candidate_k=50, filters=None):
        """
        Hybrid search.
        - candidate_k: how many to take from each ranker before fusion
        - top_k: how many to return
        - filters: dict like {"doc_type": "wordings"} (optional)
        """
        dense = self._dense_search(query, candidate_k)
        sparse = self._sparse_search(query, candidate_k)

        if filters:
            dense = [(i, s) for i, s in dense
                     if self._passes_filter(self.chunks[i], filters)]
            sparse = [(i, s) for i, s in sparse
                      if self._passes_filter(self.chunks[i], filters)]

        fused = self._fuse(dense, sparse)[:top_k]

        results = []
        for idx, score in fused:
            c = self.chunks[idx]
            results.append(
                {
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "source_pdf": c["source_pdf"],
                    "plan_name": c["plan_name"],
                    "doc_type": c["doc_type"],
                    "page": c["page"],
                    "section": c["section"],
                    "score": round(score, 6),
                }
            )
        return results

    # ---------- Debug helpers ----------

    def debug_query(self, query, top_k=5):
        """Print dense-only, sparse-only, and fused results side-by-side."""
        print("=" * 78)
        print("QUERY: " + query)
        print("TOKENIZED: " + str(self._tokenize(query)))
        print("=" * 78)

        dense = self._dense_search(query, top_k)
        sparse = self._sparse_search(query, top_k)

        print("\n--- DENSE ONLY (top " + str(top_k) + ") ---")
        for rank, (idx, score) in enumerate(dense, 1):
            c = self.chunks[idx]
            print(str(rank) + ". score=" + str(round(score, 4))
                  + " | [" + c["doc_type"] + "] " + c["plan_name"]
                  + " p" + str(c["page"]))
            print("   " + c["text"][:180].replace("\n", " "))

        print("\n--- SPARSE (BM25) ONLY (top " + str(top_k) + ") ---")
        for rank, (idx, score) in enumerate(sparse, 1):
            c = self.chunks[idx]
            print(str(rank) + ". score=" + str(round(score, 4))
                  + " | [" + c["doc_type"] + "] " + c["plan_name"]
                  + " p" + str(c["page"]))
            print("   " + c["text"][:180].replace("\n", " "))

        print("\n--- FUSED (" + FUSION_MODE
              + " / " + NORMALIZATION_MODE
              + ", weights " + str(DENSE_WEIGHT)
              + " / " + str(SPARSE_WEIGHT) + ") ---")
        for rank, res in enumerate(self.search(query, top_k=top_k), 1):
            print(str(rank) + ". score=" + str(res["score"])
                  + " | [" + res["doc_type"] + "] " + res["plan_name"]
                  + " p" + str(res["page"]) + " (" + res["section"] + ")")
            print("   " + res["text"][:180].replace("\n", " "))
        print("")


_singleton = None


def get_retriever():
    global _singleton
    if _singleton is None:
        _singleton = Retriever()
    return _singleton


if __name__ == "__main__":
    r = get_retriever()
    q = "What is the waiting period for pre-existing diseases?"

    # Debug mode: compare dense vs sparse vs fused
    r.debug_query(q, top_k=5)

    # Normal mode output
    print("=" * 78)
    print("FINAL TOP 5 RESULTS")
    print("=" * 78)
    for i, res in enumerate(r.search(q, top_k=5), 1):
        print(str(i) + ". [" + res["doc_type"] + "] " + res["plan_name"]
              + " p" + str(res["page"]) + " (" + res["section"] + ")")
        print("   " + res["text"][:250].replace("\n", " "))
        print("")