#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from style_pipeline_utils import MODEL_CONFIG, save_npz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emb_root",
        type=str,
        default="data/embeddings",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="data/plots",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    emb_root = Path(args.emb_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for model_key in MODEL_CONFIG.keys():
        model_dir = emb_root / model_key
        if not model_dir.is_dir():
            print(f"[WARN] Missing embeddings for {model_key}, skipping t-SNE.")
            continue

        print(f"Computing global t-SNE for {model_key}...")

        # Stack all review embeddings
        all_emb = []
        all_author_ids = []
        all_doc_idx = []  # 0..5 within author (after trunc/pad in consistency, but ok here)

        npz_files = sorted(model_dir.glob("*.npz"))
        for f in tqdm(npz_files, desc=f"{model_key} authors"):
            data = np.load(f, allow_pickle=True)
            author_id = str(data["author_id"])
            emb = np.asarray(data["embeddings"])
            n = emb.shape[0]
            if n == 0:
                continue
            # we keep all reviews the model produced, even if != 6
            all_emb.append(emb)
            all_author_ids.extend([author_id] * n)
            all_doc_idx.extend(list(range(n)))

        if not all_emb:
            print(f"[WARN] No embeddings found for {model_key}")
            continue

        X = np.vstack(all_emb).astype(np.float32)
        # Standardise before t-SNE for comparability
        X_scaled = StandardScaler().fit_transform(X)

        tsne = TSNE(
            n_components=2,
            perplexity=args.perplexity,
            learning_rate="auto",
            init="pca",
            metric="euclidean",
            random_state=args.random_state,
        )
        coords = tsne.fit_transform(X_scaled)

        out_npz = out_root / f"{model_key}_global_tsne.npz"
        save_npz(
            out_npz,
            coords=coords,
            author_ids=np.array(all_author_ids, dtype=object),
            doc_idx=np.array(all_doc_idx, dtype=np.int32),
        )
        print(f"{model_key}: saved global t-SNE coords to {out_npz}")


if __name__ == "__main__":
    main()