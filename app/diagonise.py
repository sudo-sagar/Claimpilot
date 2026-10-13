"""
Quick diagnostic: compare dense vs sparse retrieval quality.
Run from app/ folder: python diagnostic.py
"""
from retreival import get_retriever

QUERIES = [
    "waiting period for pre-existing diseases",
    "cataract surgery coverage",
    "maternity benefits",
]

def show(label, results, chunks):
    print("--- " + label + " ---")
    for i, (idx, score) in enumerate(results, 1):
        c = chunks[idx]
        snippet = c["text"][:120].replace("\n", " ")
        print(str(i) + ". [" + c["doc_type"] + "] p" + str(c["page"]) +
              " | score=" + str(round(score, 4)))
        print("   " + snippet)
    print("")

def main():
    r = get_retriever()
    for q in QUERIES:
        print("=" * 80)
        print("QUERY: " + q)
        print("=" * 80)
        show("DENSE (embeddings)", r._dense_search(q, 3), r.chunks)
        show("SPARSE (BM25)", r._sparse_search(q, 3), r.chunks)

if __name__ == "__main__":
    main()