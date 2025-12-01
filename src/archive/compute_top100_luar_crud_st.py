import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import pdist


def compute_author_stats(emb_path: Path):
    data = np.load(emb_path)
    author_id = str(data["author_id"])
    emb = data["embeddings"]  # shape [num_reviews, dim]
    num_reviews = emb.shape[0]

    # We only keep authors with exactly 6 reviews, per Shun's request
    if num_reviews != 6:
        return None

    # Pairwise distances between the 6 review vectors
    # pdist returns condensed distance matrix (n*(n-1)/2)
    dists = pdist(emb, metric="cosine")
    mean_dist = float(dists.mean())
    return {
        "author_id": author_id,
        "num_reviews": num_reviews,
        "mean_style_distance": mean_dist,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emb_dir",
        type=str,
        default="data/style_embeddings_luar_crud_st",
        help="Directory containing *.npz embeddings for each author.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="data/author_style_consistent_top100_luar_crud_st.csv",
        help="Output CSV for the top-100 most consistent authors.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Number of most consistent authors to keep.",
    )
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir)
    paths = sorted(emb_dir.glob("*.npz"))

    rows = []
    for p in tqdm(paths, desc="Authors"):
        stats = compute_author_stats(p)
        if stats is not None:
            rows.append(stats)

    if not rows:
        print("No authors with exactly 6 reviews were found.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values("mean_style_distance", ascending=True)

    top_k = min(args.top_k, len(df))
    df_top = df.head(top_k).reset_index(drop=True)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_top.to_csv(out_path, index=False)
    print(f"Wrote top-{top_k} authors to {out_path}")


if __name__ == "__main__":
    main()