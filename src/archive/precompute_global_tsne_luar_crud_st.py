#!/usr/bin/env python
"""
precompute_global_tsne_luar_crud_st.py

Goal:
  - Load ALL review embeddings for ALL authors from
      data/style_embeddings_luar_crud_st/*.npz
  - Stack them into a big [N_docs, dim] matrix
  - Run 2D t-SNE once on the whole matrix
  - Save:
      coords (N_docs, 2)
      author_ids (N_docs,)
      review_indices (N_docs,)
      x_min/x_max/y_min/y_max
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emb_dir",
        type=str,
        default="data/style_embeddings_luar_crud_st",
        help="Directory with per-author *.npz embedding files.",
    )
    parser.add_argument(
        "--out_npz",
        type=str,
        default="data/global_tsne_coords_luar_crud_st.npz",
        help="Output NPZ with global 2D t-SNE coords and metadata.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (controls local vs global structure).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=200.0,
        help="t-SNE learning rate.",
    )
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir)
    paths = sorted(emb_dir.glob("*.npz"))
    if not paths:
        print(f"No .npz files found in {emb_dir}")
        return

    all_embs = []
    all_author_ids = []
    all_review_indices = []

    for p in tqdm(paths, desc="Loading embeddings"):
        data = np.load(p, allow_pickle=True)
        author_id = str(data["author_id"])
        emb = data["embeddings"]  # shape [num_reviews, dim]
        n = emb.shape[0]

        all_embs.append(emb)
        all_author_ids.extend([author_id] * n)
        all_review_indices.extend(list(range(n)))

    X = np.vstack(all_embs)  # [N_docs, dim]
    print(f"Total docs: {X.shape[0]}, dim: {X.shape[1]}")

    # t-SNE (using parameters compatible with older sklearn versions)
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        init="pca",
        random_state=42,
        verbose=1,
    )
    coords = tsne.fit_transform(X)  # [N_docs, 2]

    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())

    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        coords=coords.astype(np.float32),
        author_ids=np.array(all_author_ids, dtype=object),
        review_indices=np.array(all_review_indices, dtype=np.int32),
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )

    print(f"Saved global t-SNE coords to {out_path}")
    print(f"x range: [{x_min:.3f}, {x_max:.3f}], y range: [{y_min:.3f}, {y_max:.3f}]")


if __name__ == "__main__":
    main()