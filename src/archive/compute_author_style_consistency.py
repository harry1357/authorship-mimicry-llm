import os
import glob
from typing import List, Dict

import numpy as np
import pandas as pd


def load_author_embedding_files(emb_dir: str) -> List[str]:
    """
    Find all per-author .npz embedding files in data/style_embeddings.
    Each file is expected to be named <AUTHOR_ID>.npz.
    """
    pattern = os.path.join(emb_dir, "*.npz")
    files = glob.glob(pattern)
    return sorted(files)


def compute_mean_cosine_distance(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Compute mean and std cosine distance of each review to the author's centroid.

    Assumes embeddings are L2-normalised (normalize_embeddings=True in the encoder).

    Cosine similarity between normalised vectors is just their dot product.
    Cosine distance = 1 - cosine_similarity.
    """
    if embeddings.shape[0] < 2:
        # Not enough reviews to define a spread
        return {"mean_dist": np.nan, "std_dist": np.nan}

    # Author centroid in embedding space
    centroid = embeddings.mean(axis=0, keepdims=True)  # shape (1, dim)

    # Cosine similarity: dot product between each embedding and the centroid
    sims = embeddings @ centroid.T  # (n, dim) @ (dim, 1) -> (n, 1)
    sims = sims.squeeze(axis=1)     # (n,)

    # Cosine distances
    dists = 1.0 - sims

    return {
        "mean_dist": float(dists.mean()),
        "std_dist": float(dists.std()),
    }


def main():
    # repo_root = one level above src/
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(this_dir)

    emb_dir = os.path.join(repo_root, "data", "style_embeddings")
    out_path = os.path.join(repo_root, "data", "author_style_consistency.csv")

    if not os.path.isdir(emb_dir):
        raise FileNotFoundError(
            f"Embedding directory not found: {emb_dir}\n"
            "Make sure you ran vectorise_style_embeddings.py first."
        )

    files = load_author_embedding_files(emb_dir)
    if not files:
        raise FileNotFoundError(
            f"No .npz files found in {emb_dir}. "
            "Did the vectorisation step complete successfully?"
        )

    print(f"Found {len(files)} author embedding files.")

    rows = []

    for path in files:
        filename = os.path.basename(path)
        author_id, _ext = os.path.splitext(filename)

        data = np.load(path, allow_pickle=True)
        embeddings = data["embeddings"]  # shape (n_reviews, dim)

        n_reviews = embeddings.shape[0]

        stats = compute_mean_cosine_distance(embeddings)

        rows.append(
            {
                "author_id": author_id,
                "num_reviews": int(n_reviews),
                "mean_style_distance": stats["mean_dist"],
                "std_style_distance": stats["std_dist"],
            }
        )

    df = pd.DataFrame(rows)

    # Sort by mean_style_distance ascending (tighter cluster = more consistent style)
    df_sorted = df.sort_values("mean_style_distance", na_position="last")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_sorted.to_csv(out_path, index=False)

    print(f"Saved author style consistency metrics to {out_path}")
    print(df_sorted.head(10))


if __name__ == "__main__":
    main()