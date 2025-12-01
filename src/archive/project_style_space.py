import os
import glob
import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def load_all_embeddings(
    emb_dir: str,
    max_reviews_per_author: int | None = 20,
    max_total_points: int | None = 5000,
    random_seed: int = 42,
):
    """
    Load embeddings from all <AUTHOR_ID>.npz files in emb_dir.

    Returns:
      embeddings_all: (N, dim) numpy array
      author_ids_all: list of length N, author_id per review
    """
    rng = np.random.default_rng(random_seed)

    pattern = os.path.join(emb_dir, "*.npz")
    files = sorted(glob.glob(pattern))

    all_embeddings = []
    all_authors = []

    for path in files:
        filename = os.path.basename(path)
        author_id, _ext = os.path.splitext(filename)

        data = np.load(path, allow_pickle=True)
        emb = data["embeddings"]  # (n_reviews, dim)

        n = emb.shape[0]
        if n == 0:
            continue

        # Optionally limit number of reviews per author
        if max_reviews_per_author is not None and n > max_reviews_per_author:
            idx = rng.choice(n, size=max_reviews_per_author, replace=False)
            emb = emb[idx, :]
            n = emb.shape[0]

        all_embeddings.append(emb)
        all_authors.extend([author_id] * n)

    if not all_embeddings:
        raise ValueError(f"No embeddings loaded from {emb_dir}")

    embeddings_all = np.vstack(all_embeddings)  # (N, dim)
    author_ids_all = np.array(all_authors, dtype=object)

    # Optionally limit total number of points overall
    N = embeddings_all.shape[0]
    if max_total_points is not None and N > max_total_points:
        idx = rng.choice(N, size=max_total_points, replace=False)
        embeddings_all = embeddings_all[idx, :]
        author_ids_all = author_ids_all[idx]

    return embeddings_all, author_ids_all


def run_pca(embeddings: np.ndarray, random_seed: int = 42):
    """
    Run 2D PCA on embeddings.
    """
    pca = PCA(n_components=2, random_state=random_seed)
    coords = pca.fit_transform(embeddings)
    return coords, pca


def run_tsne(embeddings: np.ndarray, random_seed: int = 42):
    """
    Run 2D t-SNE (t-distributed Stochastic Neighbor Embedding) on embeddings.
    """
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=random_seed,
    )
    coords = tsne.fit_transform(embeddings)
    return coords, tsne


def scatter_to_png(xy: np.ndarray, out_path: str, title: str):
    """
    Save a simple 2D scatter plot (all points in the same colour).
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(xy[:, 0], xy[:, 1], s=5, alpha=0.5)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(this_dir)

    parser = argparse.ArgumentParser(
        description="Project style embeddings to 2D using PCA and/or t-SNE."
    )
    parser.add_argument(
        "--emb_dir",
        type=str,
        default=os.path.join(repo_root, "data", "style_embeddings"),
        help="Directory containing per-author <AUTHOR_ID>.npz embedding files.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=os.path.join(repo_root, "data", "style_space"),
        help="Prefix for output CSV/PNG files (no extension).",
    )
    parser.add_argument(
        "--max_reviews_per_author",
        type=int,
        default=20,
        help="Maximum number of reviews per author to include (None for all).",
    )
    parser.add_argument(
        "--max_total_points",
        type=int,
        default=5000,
        help="Maximum total number of review points to include (None for all).",
    )
    parser.add_argument(
        "--no_pca",
        action="store_true",
        help="Skip PCA projection.",
    )
    parser.add_argument(
        "--no_tsne",
        action="store_true",
        help="Skip t-SNE projection.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for subsampling and t-SNE.",
    )

    args = parser.parse_args()

    emb_dir = args.emb_dir
    out_prefix = args.output_prefix

    if not os.path.isdir(emb_dir):
        raise FileNotFoundError(f"Embedding directory not found: {emb_dir}")

    print(f"Loading embeddings from: {emb_dir}")
    embeddings, author_ids = load_all_embeddings(
        emb_dir=emb_dir,
        max_reviews_per_author=args.max_reviews_per_author,
        max_total_points=args.max_total_points,
        random_seed=args.random_seed,
    )
    print(f"Loaded embeddings for {embeddings.shape[0]} reviews "
          f"(dim={embeddings.shape[1]})")

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    # PCA
    if not args.no_pca:
        print("Running PCA...")
        pca_coords, _ = run_pca(embeddings, random_seed=args.random_seed)

        pca_csv = out_prefix + "_pca.csv"
        pca_png = out_prefix + "_pca.png"

        df_pca = pd.DataFrame({
            "x": pca_coords[:, 0],
            "y": pca_coords[:, 1],
            "author_id": author_ids,
        })
        df_pca.to_csv(pca_csv, index=False)
        print(f"Saved PCA coordinates to {pca_csv}")

        scatter_to_png(pca_coords, pca_png, title="Style space (PCA)")
        print(f"Saved PCA plot to {pca_png}")

    # t-SNE
    if not args.no_tsne:
        print("Running t-SNE (this may take a while)...")
        tsne_coords, _ = run_tsne(embeddings, random_seed=args.random_seed)

        tsne_csv = out_prefix + "_tsne.csv"
        tsne_png = out_prefix + "_tsne.png"

        df_tsne = pd.DataFrame({
            "x": tsne_coords[:, 0],
            "y": tsne_coords[:, 1],
            "author_id": author_ids,
        })
        df_tsne.to_csv(tsne_csv, index=False)
        print(f"Saved t-SNE coordinates to {tsne_csv}")

        scatter_to_png(tsne_coords, tsne_png, title="Style space (t-SNE)")
        print(f"Saved t-SNE plot to {tsne_png}")


if __name__ == "__main__":
    main()