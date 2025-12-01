import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_top10_labelled(
    proj_csv_path: str,
    top100_csv_path: str,
    out_path: str,
    title: str,
):
    """
    Plot 2D style space for the TOP-10 most consistent authors only.

    Each point = one review (6 per author).
    Each of the 10 authors gets its own colour.
    The author ID is written at the centroid of its points.
    """
    df_proj = pd.read_csv(proj_csv_path)
    df_top = pd.read_csv(top100_csv_path)

    # Ensure author_id is string
    df_proj["author_id"] = df_proj["author_id"].astype(str)
    df_top["author_id"] = df_top["author_id"].astype(str)

    # Take only the first 10 authors from the consistency ranking
    df_top10 = df_top.head(10).copy()
    top10_ids = df_top10["author_id"].tolist()
    top10_set = set(top10_ids)

    print(f"Top-10 author IDs: {top10_ids}")

    # Keep only projection points for these 10 authors
    df_top_points = df_proj[df_proj["author_id"].isin(top10_set)].copy()

    print(f"Total projection points: {len(df_proj)}")
    print(f"Points for top-10 authors: {len(df_top_points)}")

    if df_top_points.empty:
        raise ValueError("No projection points found for top-10 authors.")

    unique_authors = df_top_points["author_id"].unique()
    n_authors = len(unique_authors)
    print(f"Unique top-10 authors in projection: {n_authors}")

    # Set up colours – tab10 is perfect for 10 distinct colours
    cmap = plt.get_cmap("tab10")
    colours = [cmap(i / max(1, n_authors - 1 or 1)) for i in range(n_authors)]

    plt.figure(figsize=(8, 8))

    for colour, author in zip(colours, unique_authors):
        mask = df_top_points["author_id"] == author
        points = df_top_points.loc[mask, ["x", "y"]].values

        # Scatter the reviews
        plt.scatter(
            points[:, 0],
            points[:, 1],
            s=30,
            alpha=0.9,
            color=colour,
            label=author,
        )

        # Centroid for label
        cx, cy = points.mean(axis=0)
        plt.text(
            cx,
            cy,
            author,
            fontsize=7,
            ha="center",
            va="center",
            color="black",
        )

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(fontsize=6, loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved labelled top-10 plot to {out_path}")


def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(this_dir)

    parser = argparse.ArgumentParser(
        description=(
            "Plot style space for the top-10 most consistent authors only, "
            "with each author's reviews coloured and labelled."
        )
    )
    parser.add_argument(
        "--projection",
        type=str,
        choices=["pca", "tsne"],
        default="tsne",
        help="Which projection to use: 'pca' or 'tsne'. Default: tsne.",
    )

    args = parser.parse_args()

    proj_csv = os.path.join(repo_root, f"data/style_space_{args.projection}.csv")
    top100_csv = os.path.join(repo_root, "data/author_style_consistent_top100.csv")
    out_png = os.path.join(
        repo_root, f"data/style_space_{args.projection}_top10_labelled.png"
    )

    if not os.path.isfile(proj_csv):
        raise FileNotFoundError(
            f"{proj_csv} not found. Run project_style_space.py first to create it."
        )
    if not os.path.isfile(top100_csv):
        raise FileNotFoundError(
            f"{top100_csv} not found. Run select_consistent_authors.py first."
        )

    title = f"Style space ({args.projection.upper()}): top-10 authors (6 reviews each)"
    plot_top10_labelled(
        proj_csv_path=proj_csv,
        top100_csv_path=top100_csv,
        out_path=out_png,
        title=title,
    )


if __name__ == "__main__":
    main()