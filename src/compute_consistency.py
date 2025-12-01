# src/compute_consistency.py
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .model_configs import EMBEDDINGS_DIR, CONSISTENCY_DIR, MODEL_CONFIGS


def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    norms = np.clip(norms, eps, None)
    return x / norms


def compute_pairwise_mean_distances(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute mean cosine distance of each embedding to all others.
    Returns an array of shape (n_samples,).
    """
    n = embeddings.shape[0]
    if n == 1:
        return np.zeros(1, dtype=np.float32)

    X = l2_normalize(embeddings, axis=1)
    sim = X @ X.T  # cosine similarity matrix
    np.fill_diagonal(sim, 1.0)
    dist = 1.0 - sim  # cosine distance
    mean_dist = dist.sum(axis=1) / (n - 1)
    return mean_dist.astype(np.float32)


def compute_mean_style_distance(selected_embeddings: np.ndarray) -> float:
    """
    Compute mean cosine distance of selected embeddings to their centroid.
    """
    X = selected_embeddings.astype(np.float32)
    Xn = l2_normalize(X, axis=1)
    centroid = Xn.mean(axis=0, keepdims=True)
    centroid = l2_normalize(centroid, axis=1)
    sims = (Xn @ centroid.T).reshape(-1)
    dists = 1.0 - sims
    return float(dists.mean())


def process_model(model_key: str, top_k: int = 100):
    model_dir = EMBEDDINGS_DIR / model_key
    if not model_dir.is_dir():
        raise FileNotFoundError(f"No embeddings directory for model {model_key}: {model_dir}")

    rows_all = []
    npz_files = sorted(model_dir.glob("*.npz"))

    for npz_path in tqdm(npz_files, desc=f"consistency-{model_key}"):
        data = np.load(npz_path, allow_pickle=True)
        author_id = str(data["author_id"])
        embeddings = data["embeddings"].astype(np.float32)
        num_reviews_total = embeddings.shape[0]

        if num_reviews_total == 0:
            continue

        if num_reviews_total >= 2:
            mean_dists = compute_pairwise_mean_distances(embeddings)
        else:
            # With a single review, mean distance is undefined; treat as zero for bookkeeping.
            mean_dists = np.zeros(1, dtype=np.float32)

        if num_reviews_total >= 6:
            # Select the 6 reviews with smallest mean distance
            sel_idx = np.argsort(mean_dists)[:6]
            selected_embeddings = embeddings[sel_idx]
            mean_style_distance = compute_mean_style_distance(selected_embeddings)
            num_used = 6
            selected_indices_list = sel_idx.tolist()
        else:
            # Use all reviews for logging; these authors will not enter top-100.
            sel_idx = np.arange(num_reviews_total, dtype=int)
            selected_embeddings = embeddings[sel_idx]
            mean_style_distance = compute_mean_style_distance(selected_embeddings)
            num_used = int(num_reviews_total)
            selected_indices_list = sel_idx.tolist()

        rows_all.append(
            {
                "author_id": author_id,
                "num_reviews_total": int(num_reviews_total),
                "num_reviews_used": num_used,
                "mean_style_distance": mean_style_distance,
                # JSON-encoded list of indices in original embeddings array
                "selected_indices": json.dumps(selected_indices_list),
            }
        )

    df_all = pd.DataFrame(rows_all)
    all_path = CONSISTENCY_DIR / f"{model_key}_all_authors.csv"
    df_all.to_csv(all_path, index=False)

    # Top-100 only among authors with >= 6 reviews
    eligible = df_all[df_all["num_reviews_total"] >= 6].copy()
    eligible = eligible.sort_values("mean_style_distance", ascending=True)
    df_top100 = eligible.head(top_k)
    top_path = CONSISTENCY_DIR / f"{model_key}_top100.csv"
    df_top100.to_csv(top_path, index=False)

    print(f"Wrote {all_path} and {top_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute style consistency and select 6 most consistent reviews per author."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Which model_keys to process; 'all' runs every model in MODEL_CONFIGS.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Number of most consistent authors to keep in *_top100.csv (default 100).",
    )
    args = parser.parse_args()

    if "all" in args.models:
        model_keys = list(MODEL_CONFIGS.keys())
    else:
        model_keys = args.models

    for mk in model_keys:
        process_model(mk, top_k=args.top_k)


if __name__ == "__main__":
    main()