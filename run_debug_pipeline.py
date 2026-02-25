"""Quick debug runner: synthesize embeddings and BM25 scores for fixture
patents, apply current ranking pipeline thresholds, and print score
distributions + top-N.

Run: python run_debug_pipeline.py
"""
from pathlib import Path
import json
import numpy as np

from app.config import settings
from models.schemas import PatentRecord
from services import ranking_service
from retrieval.similarity import cosine_similarity_matrix


def load_fixture(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    patents = [PatentRecord(**p) for p in data["patents"]]
    return patents


def synthesize_embeddings(n, dim=128, seed=0):
    rs = np.random.RandomState(seed)
    mat = rs.randn(n, dim)
    # L2-normalise rows
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    return mat


def main():
    fx = Path("tests/fixtures/patentsview_page_full.json")
    patents = load_fixture(fx)
    n = len(patents)

    doc_vecs = synthesize_embeddings(n, dim=128, seed=1)
    # Make query_vec similar to the first patent to get a range of cosines
    query_vec = doc_vecs[0] + 0.2 * np.random.RandomState(2).randn(doc_vecs.shape[1])
    query_vec = query_vec / np.linalg.norm(query_vec)

    # Synthetic BM25 raw scores (unbounded) — higher for patent 0
    bm25_raw = np.array([12.0, 6.0, 3.0, 1.5, 0.5], dtype=float)

    # Use ranking_service helpers to compute
    cosine = cosine_similarity_matrix(query_vec, doc_vecs)
    print("Cosine scores:", np.round(cosine, 4))

    # Apply cosine threshold filter
    all_idx = np.arange(n)
    keep_idx, used_threshold = ranking_service._apply_cosine_threshold(cosine, all_idx)
    print(f"Applied cosine threshold {used_threshold:.2f}: kept {len(keep_idx)}/{n}")

    cosine_f = cosine[keep_idx]
    bm25_raw_f = bm25_raw[keep_idx]

    bm25_norm = ranking_service.normalize(bm25_raw_f)
    hybrid = ranking_service.hybrid_rank(cosine_f, bm25_raw_f)

    print("BM25 raw (filtered):", np.round(bm25_raw_f, 4))
    print("BM25 norm (filtered):", np.round(bm25_norm, 4))
    print("Hybrid (filtered):", np.round(hybrid, 4))

    # Apply hybrid threshold
    h_threshold = getattr(settings, "hybrid_threshold", 0.0)
    if h_threshold and h_threshold > 0.0:
        mask_h = hybrid >= h_threshold
        print(f"Applying hybrid threshold {h_threshold:.2f}: {mask_h.sum()} pass")
        hybrid = hybrid[mask_h]
        cosine_f = cosine_f[mask_h]
        bm25_norm = bm25_norm[mask_h]
        keep_idx = keep_idx[mask_h]

    # Sort and show top-k
    k = settings.top_k_results
    order = np.argsort(hybrid)[::-1][:k]

    print("\nTop results:")
    for rank, i in enumerate(order, start=1):
        idx = keep_idx[i]
        p = patents[idx]
        print(f"{rank}\t{p.patent_id}\t{p.patent_title}\t{p.patent_type}\t" \
              f"cosine={cosine_f[i]:.4f}\tbm25={bm25_norm[i]:.4f}\thybrid={hybrid[i]:.4f}")


if __name__ == "__main__":
    main()
