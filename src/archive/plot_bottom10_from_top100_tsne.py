#!/usr/bin/env python
"""
plot_bottom10_from_top100_tsne.py

For a given model:

  - Load top-100 authors CSV (sorted by mean_style_distance ascending).
  - Take the bottom 10 (worst within the "good" set).
  - Load global t-SNE coords for ALL reviews (from precompute_global_tsne_*).
  - For each of the 10 authors, plot:
      * all other reviews in light grey
      * that author's 6 reviews in orange
    using the SAME global x/y limits per model.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_authors_csv", type=str, required=True)
    parser.add_argument("--tsne_npz", type=str, required=True)
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for per-author plots.",
    )
    parser.add_argument(
        "--bottom_k",
        type=int,
        default=10,
        help="How many worst authors (from top-100) to plot.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load top-100 authors, sorted best -> worst
    df_top = pd.read_csv(args.top_authors_csv)
    df_top = df_top.sort_values("mean_style_distance", ascending=True).reset_index(drop=True)

    if len(df_top) < args.bottom_k:
        print(f"[WARN] Only {len(df_top)} rows in {args.top_authors_csv}, bottom_k={args.bottom_k}")
        bottom = df_top
    else:
        bottom = df_top.tail(args.bottom_k)

    print(f"Plotting bottom {len(bottom)} authors from top-100.")

    # Load global t-SNE
    data = np.load(args.tsne_npz, allow_pickle=True)
    coords = data["coords"]        # [N_docs, 2]
    author_ids = data["author_ids"]  # [N_docs]
    x_min, x_max = float(data["x_min"]), float(data["x_max"])
    y_min, y_max = float(data["y_min"]), float(data["y_max"])

    author_ids = np.array(author_ids)

    for _, row in tqdm(bottom.iterrows(), total=len(bottom), desc="Authors"):
        author_id = row["author_id"]

        mask_this = (author_ids == author_id)
        mask_other = ~mask_this

        if mask_this.sum() != 6:
            print(f"[WARN] Author {author_id} has {mask_this.sum()} points in TSNE, expected 6.")

        plt.figure(figsize=(6, 6))
        plt.scatter(
            coords[mask_other, 0],
            coords[mask_other, 1],
            s=5,
            alpha=0.2,
            label="Other authors",
            color="lightgrey",
        )
        plt.scatter(
            coords[mask_this, 0],
            coords[mask_this, 1],
            s=30,
            alpha=0.9,
            label=f"{author_id}",
        )
        plt.title(f"Author {author_id} (bottom of top-100)")
        plt.xlabel("t-SNE dim 1")
        plt.ylabel("t-SNE dim 2")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()

        out_path = out_dir / f"{author_id}_tsne_bottom.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()