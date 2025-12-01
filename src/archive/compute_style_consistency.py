#!/usr/bin/env python
"""
compute_style_consistency.py

Reads all .npz embedding files in a directory, and for each author:
  - takes their embedding matrix [num_reviews, dim]
  - (optionally) enforces exactly min_reviews
  - computes distances from centroid
  - outputs mean/std/max distances
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def compute_metrics(embeddings: np.ndarray) -> dict:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

    num_reviews = embeddings.shape[0]
    centroid = embeddings.mean(axis=0)
    diffs = embeddings - centroid
    dists = np.linalg.norm(diffs, axis=1)

    return {
        "num_reviews": num_reviews,
        "mean_style_distance": float(dists.mean()),
        "std_style_distance": float(dists.std()),
        "max_style_distance": float(dists.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emb_dir",
        type=str,
        required=True,
        help="Directory containing per-author .npz files with 'embeddings' key.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Output CSV path for the consistency stats.",
    )
    parser.add_argument(
        "--min_reviews",
        type=int,
        default=6,
        help="Minimum number of reviews per author (default: 6).",
    )
    parser.add_argument(
        "--exact_reviews",
        type=int,
        default=6,
        help="If > 0, require exactly this many reviews per author (default: 6).",
    )
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir)
    if not emb_dir.exists():
        raise FileNotFoundError(f"Embedding directory not found: {emb_dir}")

    npz_files = sorted(glob.glob(str(emb_dir / "*.npz")))
    print(f"Found {len(npz_files)} .npz files in {emb_dir}")

    rows = []
    skipped = 0

    for path in npz_files:
        data = np.load(path, allow_pickle=True)
        embeddings = data["embeddings"]
        author_id = str(data["author_id"])

        num_reviews = embeddings.shape[0]
        if num_reviews < args.min_reviews:
            skipped += 1
            continue
        if args.exact_reviews > 0 and num_reviews != args.exact_reviews:
            skipped += 1
            continue

        metrics = compute_metrics(embeddings)
        rows.append(
            {
                "author_id": author_id,
                **metrics,
            }
        )

    df = pd.DataFrame(rows).sort_values("mean_style_distance", ascending=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Wrote stats for {len(df)} authors to {out_csv}")
    print(f"Skipped {skipped} authors due to review count constraints.")


if __name__ == "__main__":
    main()