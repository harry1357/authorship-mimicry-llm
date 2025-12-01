#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from style_pipeline_utils import MODEL_CONFIG


def load_author_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def compute_centroid_consistency(emb: np.ndarray) -> float:
    """
    Our ranking metric: mean Euclidean distance from
    each review embedding to the author centroid.

    Lower = more consistent.
    """
    if emb.shape[0] == 0:
        return np.inf
    centroid = emb.mean(axis=0, keepdims=True)
    dists = np.linalg.norm(emb - centroid, axis=1)
    return float(dists.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emb_root",
        type=str,
        default="data/embeddings",
        help="Base embedding folder (contains subdirs per model).",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="data/consistency",
        help="Folder for consistency CSVs.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Number of most consistent authors to keep.",
    )
    args = parser.parse_args()

    emb_root = Path(args.emb_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for model_key in MODEL_CONFIG.keys():
        model_dir = emb_root / model_key
        if not model_dir.is_dir():
            print(f"[WARN] Missing embeddings for {model_key} in {model_dir}, skipping.")
            continue

        rows = []
        npz_files = sorted(model_dir.glob("*.npz"))
        print(f"Model {model_key}: found {len(npz_files)} authors.")

        for f in tqdm(npz_files, desc=f"{model_key} authors"):
            data = load_author_npz(f)
            author_id = str(data["author_id"])
            emb = np.asarray(data["embeddings"])
            orig_n = int(data.get("original_num_reviews", emb.shape[0]))

            # Enforce exactly 6 reviews for consistency calc
            if emb.shape[0] == 0:
                # completely missing – give infinite distance
                mean_dist = np.inf
                used_n = 0
            else:
                # adjust to 6 if needed
                if emb.shape[0] < 6:
                    # repeat last row
                    last = emb[-1:]
                    while emb.shape[0] < 6:
                        emb = np.vstack([emb, last])
                elif emb.shape[0] > 6:
                    emb = emb[:6]

                used_n = emb.shape[0]
                mean_dist = compute_centroid_consistency(emb)

            rows.append(
                {
                    "author_id": author_id,
                    "mean_centroid_dist": mean_dist,
                    "original_num_reviews": orig_n,
                    "used_reviews": used_n,
                }
            )

        df = pd.DataFrame(rows)
        # Filter to authors where we actually have 6 used reviews
        df = df[df["used_reviews"] == 6].copy()

        df_sorted = df.sort_values("mean_centroid_dist", ascending=True)
        df_top = df_sorted.head(args.top_k)

        out_csv = out_root / f"{model_key}_top{args.top_k}.csv"
        df_top.to_csv(out_csv, index=False)
        print(f"Model {model_key}: wrote top-{args.top_k} to {out_csv}")


if __name__ == "__main__":
    main()