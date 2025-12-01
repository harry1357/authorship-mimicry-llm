# src/plot_bottom_k.py
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model_configs import (
    EMBEDDINGS_DIR,
    CONSISTENCY_DIR,
    PLOTS_DIR,
    MODEL_CONFIGS,
)


def plot_bottom_k_for_model(model_key: str, k: int = 10):
    tsne_path = EMBEDDINGS_DIR / f"global_tsne_{model_key}.npz"
    top100_path = CONSISTENCY_DIR / f"{model_key}_top100.csv"

    if not tsne_path.is_file():
        raise FileNotFoundError(f"Missing t-SNE file: {tsne_path}")
    if not top100_path.is_file():
        raise FileNotFoundError(f"Missing top-100 file: {top100_path}")

    tsne_data = np.load(tsne_path, allow_pickle=True)
    coords = tsne_data["coords"]
    author_ids = tsne_data["author_ids"]
    doc_indices = tsne_data["doc_indices"]

    df_top = pd.read_csv(top100_path)
    # Sort descending to get "worst" authors among the best 100
    df_top_sorted = df_top.sort_values("mean_style_distance", ascending=False)
    worst_authors = df_top_sorted.head(k).copy()

    # Map author_id -> selected_indices for quick lookup
    sel_map = {}
    for _, row in df_top.iterrows():
        aid = str(row["author_id"])
        try:
            sel = json.loads(row["selected_indices"])
        except Exception:
            sel = []
        sel_map[aid] = [int(x) for x in sel]

    out_dir = PLOTS_DIR / model_key / "bottomK"
    out_dir.mkdir(parents=True, exist_ok=True)

    for rank, (_, row) in enumerate(worst_authors.iterrows(), start=1):
        author_id = str(row["author_id"])
        selected_indices = sel_map.get(author_id, [])

        mask_author = (author_ids == author_id)
        if selected_indices:
            mask_highlight = mask_author & np.isin(doc_indices, selected_indices)
        else:
            # Fallback: highlight all points for this author if selection missing
            mask_highlight = mask_author

        fig, ax = plt.subplots(figsize=(6, 6))
        # Background: all points in light grey
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=5,
            c="lightgrey",
            alpha=0.25,
            edgecolors="none",
        )
        # Highlighted author: 6 selected points
        ax.scatter(
            coords[mask_highlight, 0],
            coords[mask_highlight, 1],
            s=30,
            c="tab:red",
            alpha=0.9,
            edgecolors="black",
            linewidths=0.5,
        )

        ax.set_title(
            f"{model_key} – worst top-100 author {author_id}\n"
            f"rank {rank} by mean_style_distance"
        )
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")

        fig.tight_layout()
        out_path = out_dir / f"{rank:02d}_{author_id}.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot bottom-K authors (worst of top-100) per model using global t-SNE."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Which model_keys to process; 'all' runs every model in MODEL_CONFIGS.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of worst authors among top-100 to plot (default 10).",
    )
    args = parser.parse_args()

    if "all" in args.models:
        model_keys = list(MODEL_CONFIGS.keys())
    else:
        model_keys = args.models

    for mk in model_keys:
        plot_bottom_k_for_model(mk, k=args.k)


if __name__ == "__main__":
    main()