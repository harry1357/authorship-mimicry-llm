#!/usr/bin/env python
"""
plot_top100_vs_all_tsne_luar_crud_st.py

For each of the top-100 most consistent authors (by LUAR-CRUD-ST),
create a plot where:

  - All reviews (all authors) are shown as light-grey points (global t-SNE).
  - This author's reviews (6 points) are highlighted in colour.

All plots use the same global t-SNE coordinates and axis limits.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--top_authors_csv",
        type=str,
        default="data/author_style_consistent_top100_luar_crud_st.csv",
        help="CSV with top-100 authors (must have an 'author_id' column).",
    )
    parser.add_argument(
        "--tsne_npz",
        type=str,
        default="data/global_tsne_coords_luar_crud_st.npz",
        help="NPZ with global t-SNE coords and author_ids.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/distribution_plots_luar_crud_st_tsne_global",
        help="Directory where per-author PNGs will be saved.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load top-100 authors
    df = pd.read_csv(args.top_authors_csv)
    top_author_ids = df["author_id"].astype(str).tolist()
    print(f"Top authors loaded: {len(top_author_ids)}")

    # Load global t-SNE coords + metadata
    data = np.load(args.tsne_npz, allow_pickle=True)
    coords = data["coords"]  # [N_docs, 2]
    all_author_ids = data["author_ids"]  # [N_docs]
    x_min, x_max = float(data["x_min"]), float(data["x_max"])
    y_min, y_max = float(data["y_min"]), float(data["y_max"])

    print(f"Global t-SNE: {coords.shape[0]} docs total")
    print(f"x range: [{x_min:.3f}, {x_max:.3f}], y range: [{y_min:.3f}, {y_max:.3f}]")

    for author_id in tqdm(top_author_ids, desc="Plotting authors"):
        mask_author = (all_author_ids == author_id)
        if not mask_author.any():
            # No docs found for this author in t-SNE coords
            print(f"[WARN] No docs for author {author_id} in t-SNE embedding, skipping.")
            continue

        fig, ax = plt.subplots(figsize=(6, 6))

        # All other reviews in light grey
        ax.scatter(
            coords[~mask_author, 0],
            coords[~mask_author, 1],
            s=5,
            c="lightgrey",
            alpha=0.3,
            label="Other authors",
        )

        # This author's reviews highlighted
        ax.scatter(
            coords[mask_author, 0],
            coords[mask_author, 1],
            s=35,
            c="tab:orange",
            edgecolors="black",
            linewidths=0.5,
            alpha=0.9,
            label=f"Author {author_id}",
        )

        ax.set_title(
            f"Author {author_id} – style distribution vs all others\n"
            f"LUAR-CRUD-ST + global t-SNE",
            fontsize=10,
        )
        ax.set_xlabel("t-SNE dimension 1")
        ax.set_ylabel("t-SNE dimension 2")

        # Use same axes for all plots
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.legend(loc="best", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        out_path = out_dir / f"author_{author_id}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    print(f"Saved per-author t-SNE plots to {out_dir}")


if __name__ == "__main__":
    main()