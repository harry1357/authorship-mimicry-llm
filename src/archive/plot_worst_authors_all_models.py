#!/usr/bin/env python

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from style_pipeline_utils import MODEL_CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--consistency_root",
        type=str,
        default="data/consistency",
    )
    parser.add_argument(
        "--tsne_root",
        type=str,
        default="data/plots",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="data/plots",
    )
    parser.add_argument(
        "--bottom_k",
        type=int,
        default=10,
        help="How many 'worst of top-100' authors to plot per model.",
    )
    args = parser.parse_args()

    tsne_root = Path(args.tsne_root)
    cons_root = Path(args.consistency_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for model_key in MODEL_CONFIG.keys():
        cons_csv = cons_root / f"{model_key}_top100.csv"
        tsne_npz = tsne_root / f"{model_key}_global_tsne.npz"

        if not cons_csv.is_file() or not tsne_npz.is_file():
            print(f"[WARN] Missing files for {model_key}, skipping plots.")
            continue

        df = pd.read_csv(cons_csv)
        # bottom_k of the *top-100* (worst of good authors)
        df_bottom = df.tail(args.bottom_k).copy()

        tsne_data = np.load(tsne_npz, allow_pickle=True)
        coords = np.asarray(tsne_data["coords"])
        author_ids = np.asarray(tsne_data["author_ids"])

        for _, row in tqdm(df_bottom.iterrows(), total=len(df_bottom), desc=f"{model_key} plots"):
            author_id = row["author_id"]
            mean_dist = row["mean_centroid_dist"]

            mask_author = (author_ids == author_id)
            coords_author = coords[mask_author]
            coords_others = coords[~mask_author]

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(
                coords_others[:, 0],
                coords_others[:, 1],
                s=4,
                c="lightgray",
                alpha=0.3,
                label="Other authors",
            )
            ax.scatter(
                coords_author[:, 0],
                coords_author[:, 1],
                s=40,
                c="tab:blue",
                alpha=0.9,
                label=author_id,
            )
            ax.set_title(
                f"{model_key} – author {author_id}\n"
                f"(bottom of top-100, mean centroid dist={mean_dist:.4f})"
            )
            ax.set_xlabel("t-SNE dim 1")
            ax.set_ylabel("t-SNE dim 2")
            ax.legend(loc="best")

            out_path = out_root / f"{model_key}_author_{author_id}_bottom_top100.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

        print(f"{model_key}: wrote {len(df_bottom)} bottom-of-top-100 plots to {out_root}")


if __name__ == "__main__":
    main()