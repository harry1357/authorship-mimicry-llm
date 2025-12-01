import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_author_embeddings(emb_dir: str, author_id: str) -> np.ndarray:
    """
    Load the embeddings array for a single author from <emb_dir>/<AUTHOR_ID>.npz.
    Returns a numpy array of shape (n_reviews, dim).
    """
    path = os.path.join(emb_dir, f"{author_id}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Embedding file not found for author {author_id}: {path}")

    data = np.load(path, allow_pickle=True)
    emb = data["embeddings"]  # (n_reviews, dim)
    return emb


def pca_2d(embeddings: np.ndarray, random_seed: int = 42) -> np.ndarray:
    """
    Run a 2D PCA on the given embeddings (n_reviews, dim).
    Returns an array of shape (n_reviews, 2).
    """
    if embeddings.shape[0] < 2:
        return np.zeros((embeddings.shape[0], 2))

    pca = PCA(n_components=2, random_state=random_seed)
    coords = pca.fit_transform(embeddings)
    return coords


def plot_author_distribution(
    coords: np.ndarray,
    author_id: str,
    out_path: str,
    xlim,
    ylim,
):
    """
    Make a small scatter plot for a single author:
      - each point = one review (expected 6)
      - coordinates = 2D PCA within this author's style space
      - x/y limits shared across all authors for comparability
    """
    plt.figure(figsize=(4, 4))

    plt.scatter(coords[:, 0], coords[:, 1], s=40, alpha=0.9, color="C0")

    # Label each review 1..n
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i + 1), fontsize=8, ha="center", va="center", color="black")

    plt.title(f"Author {author_id} – review style distribution")
    plt.xlabel("PC1 (within-author)")
    plt.ylabel("PC2 (within-author)")

    # Use the same limits for all authors
    plt.xlim(xlim)
    plt.ylim(ylim)

    # Make scale equal so distance is visually meaningful
    plt.gca().set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(this_dir)

    parser = argparse.ArgumentParser(
        description="Make per-author 2D PCA plots for the top-20 most consistent authors (exactly 6 reviews), with shared axis ranges."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of most-consistent authors to plot (default: 20).",
    )
    parser.add_argument(
        "--emb_dir",
        type=str,
        default=os.path.join(repo_root, "data", "style_embeddings"),
        help="Directory with <AUTHOR_ID>.npz embedding files.",
    )
    parser.add_argument(
        "--consistency_csv",
        type=str,
        default=os.path.join(repo_root, "data", "author_style_consistent_top100.csv"),
        help="CSV with author_id and consistency metrics, sorted by mean_style_distance.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(repo_root, "data", "distribution_plots"),
        help="Output directory for per-author distribution plots.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for PCA initialisation.",
    )

    args = parser.parse_args()

    emb_dir = args.emb_dir
    consistency_csv = args.consistency_csv
    out_dir = args.out_dir

    if not os.path.isdir(emb_dir):
        raise FileNotFoundError(f"Embedding directory not found: {emb_dir}")
    if not os.path.isfile(consistency_csv):
        raise FileNotFoundError(
            f"Consistency CSV not found: {consistency_csv}. "
            "Run compute_author_style_consistency.py and select_consistent_authors.py first."
        )

    os.makedirs(out_dir, exist_ok=True)

    # Load consistency table and take top_k authors
    df = pd.read_csv(consistency_csv)
    df_topk = df.head(args.top_k).copy()

    print(f"Top {args.top_k} authors from {consistency_csv}:")
    print(df_topk[["author_id", "num_reviews", "mean_style_distance"]])

    # First pass: compute coordinates for all authors and collect global bounds
    coords_dict = {}
    xs = []
    ys = []

    for _, row in df_topk.iterrows():
        author_id = str(row["author_id"])
        num_reviews = int(row["num_reviews"])

        print(f"\nComputing PCA coords for author {author_id} (num_reviews={num_reviews})")

        emb = load_author_embeddings(emb_dir, author_id)

        if emb.shape[0] != 6:
            print(
                f"  WARNING: author {author_id} has {emb.shape[0]} reviews in embeddings, "
                "expected 6."
            )

        coords = pca_2d(emb, random_seed=args.random_seed)
        coords_dict[author_id] = coords

        xs.extend(coords[:, 0].tolist())
        ys.extend(coords[:, 1].tolist())

    # Compute global axis limits with a small margin
    if not xs or not ys:
        raise ValueError("No coordinates collected for any author.")

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Add ~10% padding
    x_range = x_max - x_min
    y_range = y_max - y_min

    pad_x = 0.1 * x_range if x_range > 0 else 0.1
    pad_y = 0.1 * y_range if y_range > 0 else 0.1

    xlim = (x_min - pad_x, x_max + pad_x)
    ylim = (y_min - pad_y, y_max + pad_y)

    print(f"\nGlobal xlim: {xlim}")
    print(f"Global ylim: {ylim}")

    # Second pass: plot each author using shared limits
    for _, row in df_topk.iterrows():
        author_id = str(row["author_id"])
        coords = coords_dict[author_id]

        out_path = os.path.join(
            out_dir,
            f"author_{author_id}_distribution_pca2d_shared_axes.png",
        )
        plot_author_distribution(coords, author_id, out_path, xlim, ylim)


if __name__ == "__main__":
    main()