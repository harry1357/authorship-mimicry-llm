import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_top100_vs_others(
    proj_csv_path: str,
    top100_csv_path: str,
    out_path: str,
    title: str,
):
    """
    Plot 2D style space where:
      - reviews from the top-100 most consistent authors are coloured
        (each author gets its own colour),
      - reviews from all other authors are grey.

    Assumes:
      proj_csv has columns: x, y, author_id
      top100_csv has column: author_id
    """
    df_proj = pd.read_csv(proj_csv_path)
    df_top = pd.read_csv(top100_csv_path)

    # Ensure consistent types
    df_proj["author_id"] = df_proj["author_id"].astype(str)
    df_top["author_id"] = df_top["author_id"].astype(str)

    top_ids = df_top["author_id"].tolist()
    top_set = set(top_ids)

    print(f"Projection points: {len(df_proj)}")
    print(f"Top-100 author IDs loaded: {len(top_ids)}")

    # Mask for top-100 vs others
    is_top = df_proj["author_id"].isin(top_set)

    df_top_points = df_proj[is_top].copy()
    df_other_points = df_proj[~is_top].copy()

    print(f"Top-100 points: {len(df_top_points)}")
    print(f"Other points: {len(df_other_points)}")

    plt.figure(figsize=(7, 7))

    # Plot others as grey background
    if not df_other_points.empty:
        plt.scatter(
            df_other_points["x"],
            df_other_points["y"],
            s=5,
            alpha=0.15,
            color="lightgrey",
            label="Other authors",
        )

    # We want distinct colours for up to 100 authors
    unique_top_authors = df_top_points["author_id"].unique()
    n_top = len(unique_top_authors)
    print(f"Unique top-100 authors found in projection: {n_top}")

    # Build a colour map with n_top distinct colours
    cmap = plt.get_cmap("tab20") if n_top <= 20 else plt.get_cmap("hsv")
    colours = [cmap(i / max(1, n_top)) for i in range(n_top)]

    # Plot each top-100 author in its own colour
    for colour, author in zip(colours, unique_top_authors):
        mask = df_top_points["author_id"] == author
        plt.scatter(
            df_top_points.loc[mask, "x"],
            df_top_points.loc[mask, "y"],
            s=10,
            alpha=0.7,
            color=colour,
        )

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")

    # Legend with just two entries: top-100 vs others
    # (avoids a 100-line legend)
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="lightgrey", markersize=6, label="Other authors"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="C0", markersize=6, label="Top-100 authors"),
    ]
    plt.legend(handles=legend_elements, loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")


def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(this_dir)

    parser = argparse.ArgumentParser(
        description="Plot style space with top-100 consistent authors in colour, others in grey."
    )
    parser.add_argument(
        "--projection",
        type=str,
        choices=["pca", "tsne"],
        default="pca",
        help="Which projection to use: 'pca' or 'tsne'.",
    )

    args = parser.parse_args()

    proj_csv = os.path.join(repo_root, f"data/style_space_{args.projection}.csv")
    top100_csv = os.path.join(repo_root, "data/author_style_consistent_top100.csv")
    out_png = os.path.join(
        repo_root, f"data/style_space_{args.projection}_top100_vs_others.png"
    )

    if not os.path.isfile(proj_csv):
        raise FileNotFoundError(
            f"{proj_csv} not found. Run project_style_space.py first to create it."
        )
    if not os.path.isfile(top100_csv):
        raise FileNotFoundError(
            f"{top100_csv} not found. Run compute_author_style_consistency.py and "
            "select_consistent_authors.py first."
        )

    title = f"Style space ({args.projection.upper()}): top-100 authors vs others"
    plot_top100_vs_others(
        proj_csv_path=proj_csv,
        top100_csv_path=top100_csv,
        out_path=out_png,
        title=title,
    )


if __name__ == "__main__":
    main()