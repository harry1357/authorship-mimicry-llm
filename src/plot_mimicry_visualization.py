#!/usr/bin/env python3
"""
Specialized Mimicry Visualization

Creates plots optimized for ASSESSING MIMICRY QUALITY, not author separation.

Unlike the global 157-author plots (which maximize separation), these plots:
1. Show each author individually with their generated texts
2. Use color to indicate HONEST distance to real training documents
3. Display TRUE embedding distances (in the color scale, not the 2D layout)
4. Create per-author subplots or grids for easy comparison

Distances are always:
    cosine distance between each point and ALL training documents for that author,
    averaged across all training docs (HONEST metric, not centroid distance).

Usage:
    # Individual author plots for top 10
    python src/plot_mimicry_visualization.py --model-key style_embedding --full-run 1 --top-n 10

    # Grid view of all top authors
    python src/plot_mimicry_visualization.py --model-key style_embedding --full-run 1 --top-n 10 --grid-view
"""

import argparse
import csv
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from generation_config import (
    EMBEDDINGS_DIR,
    CONSISTENCY_DIR,
    REFERENCE_CONSISTENCY_CSV,
    STYLE_MODEL_KEYS,
)
from model_configs import PLOTS_DIR


def load_selected_indices():
    """Load pre-computed selected indices from consistency CSV."""
    mapping = {}
    if not REFERENCE_CONSISTENCY_CSV.exists():
        return mapping

    with REFERENCE_CONSISTENCY_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            author_id = row["author_id"]
            raw = row.get("selected_indices", "").strip()
            if not raw:
                continue
            try:
                indices = ast.literal_eval(raw)
            except Exception:
                indices = [
                    int(x)
                    for x in raw.replace("[", "").replace("]", "").split(",")
                    if x.strip()
                ]
            indices = [int(i) for i in indices]
            if indices:
                mapping[author_id] = indices
    return mapping


def load_author_embeddings(author_id, model_key, llm_key, full_run, selected_indices_map):
    """
    Load training + generated embeddings with consistent training doc selection.

    Training docs:
      • Prefer selected_indices from REFERENCE_CONSISTENCY_CSV (same as prompts/analysis)
      • Fallback to 'selected_indices' inside the .npz (if present)
      • Otherwise use first 6 embeddings (if available)
    """
    train_path = EMBEDDINGS_DIR / model_key / f"{author_id}.npz"
    if not train_path.exists():
        return None

    train_data = np.load(train_path, allow_pickle=True)
    train_embs = train_data["embeddings"]

    # Choose which training indices to use (robustly)
    if author_id in selected_indices_map:
        raw_idx = selected_indices_map[author_id]
        valid_idx = [i for i in raw_idx if 0 <= i < len(train_embs)]
        if len(valid_idx) >= 6:
            indices_to_use = sorted(valid_idx)[:6]
        elif len(valid_idx) > 0:
            indices_to_use = sorted(valid_idx)
            print(
                f"[WARNING] Author {author_id}: fewer than 6 valid indices in CSV "
                f"({len(valid_idx)} valid, {len(raw_idx)} total); using valid subset."
            )
        else:
            if len(train_embs) >= 6:
                indices_to_use = list(range(6))
                print(
                    f"[WARNING] Author {author_id}: all CSV indices out of range; "
                    f"falling back to first 6 embeddings."
                )
            else:
                indices_to_use = list(range(len(train_embs)))
                print(
                    f"[WARNING] Author {author_id}: not enough embeddings "
                    f"({len(train_embs)}); using all."
                )
        training_embs = train_embs[indices_to_use]
    elif "selected_indices" in train_data:
        internal_idx = train_data["selected_indices"]
        valid_idx = [i for i in internal_idx if 0 <= i < len(train_embs)]
        if not valid_idx:
            if len(train_embs) >= 6:
                training_embs = train_embs[:6]
            else:
                training_embs = train_embs
            print(
                f"[WARNING] Author {author_id}: 'selected_indices' inside npz are "
                f"invalid; falling back to first {len(training_embs)} embeddings."
            )
        else:
            training_embs = train_embs[valid_idx]
    else:
        if len(train_embs) >= 6:
            training_embs = train_embs[:6]
        else:
            training_embs = train_embs

    # Generated embeddings
    simple_path = (
        EMBEDDINGS_DIR
        / "generated"
        / model_key
        / llm_key
        / "simple"
        / f"fullrun{full_run}"
        / f"{author_id}.npz"
    )
    if not simple_path.exists():
        return None
    simple_embs = np.load(simple_path, allow_pickle=True)["embeddings"]

    complex_path = (
        EMBEDDINGS_DIR
        / "generated"
        / model_key
        / llm_key
        / "complex"
        / f"fullrun{full_run}"
        / f"{author_id}.npz"
    )
    if not complex_path.exists():
        return None
    complex_embs = np.load(complex_path, allow_pickle=True)["embeddings"]

    return {
        "training": training_embs,
        "simple": simple_embs,
        "complex": complex_embs,
    }


def _reduce_to_2d(all_embs, method):
    """
    Reduce embeddings to 2D for plotting.

    NOTE: 2D layout is ONLY for visualization. True distances are carried
    in the color values, not the x/y distances.
    """
    n_samples = len(all_embs)

    # For tiny sample sizes, TSNE is unstable / invalid. Fall back to PCA.
    if method == "tsne" and n_samples >= 5:
        # Perplexity must be < n_samples and >= 2
        perplexity = min(5, max(2, n_samples - 1))
        reducer = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            max_iter=1000,
        )
        coords_2d = reducer.fit_transform(all_embs)
        method_name = "t-SNE"
    else:
        reducer = PCA(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(all_embs)
        if method == "pca":
            method_name = "PCA"
        else:
            method_name = "PCA (fallback)"

    return coords_2d, method_name


def plot_single_author_mimicry(
    author_id,
    author_data,
    rank,
    model_key,
    full_run,
    output_dir,
    method="pca",
):
    """
    Create a single-author plot showing mimicry quality.

    • Each point's COLOR encodes the HONEST distance to all training docs
      (mean cosine distance to all training embeddings for that author)
    • The 2D layout (PCA/t-SNE) is just a visualization; do NOT read distance off the axes.
    """
    # Combine all embeddings
    all_embs = np.vstack(
        [
            author_data["training"],
            author_data["simple"],
            author_data["complex"],
        ]
    )

    # HONEST distances: mean distance to ALL training docs (not centroid)
    training_docs = author_data["training"]
    all_dists_to_training = cdist(all_embs, training_docs, metric="cosine")
    all_mean_dists_to_training = np.mean(all_dists_to_training, axis=1)

    # Project to 2D for visualization
    coords_2d, method_name = _reduce_to_2d(all_embs, method)

    # Split coordinates and distances
    n_train = len(author_data["training"])
    n_simple = len(author_data["simple"])
    n_complex = len(author_data["complex"])

    train_coords = coords_2d[:n_train]
    simple_coords = coords_2d[n_train : n_train + n_simple]
    complex_coords = coords_2d[n_train + n_simple :]

    train_dists = all_mean_dists_to_training[:n_train]
    simple_dists = all_mean_dists_to_training[n_train : n_train + n_simple]
    complex_dists = all_mean_dists_to_training[n_train + n_simple :]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Training docs
    ax.scatter(
        train_coords[:, 0],
        train_coords[:, 1],
        c=train_dists,
        cmap="Greens_r",
        vmin=0.05,
        vmax=0.35,
        marker="o",
        s=200,
        alpha=0.7,
        edgecolors="black",
        linewidths=2,
        label="Training",
    )

    # Simple generated
    scatter_simple = ax.scatter(
        simple_coords[:, 0],
        simple_coords[:, 1],
        c=simple_dists,
        cmap="RdYlGn_r",
        vmin=0.05,
        vmax=0.35,
        marker="s",
        s=250,
        alpha=0.9,
        edgecolors="black",
        linewidths=2,
        label="Simple Generated",
    )

    # Complex generated
    ax.scatter(
        complex_coords[:, 0],
        complex_coords[:, 1],
        c=complex_dists,
        cmap="RdYlGn_r",
        vmin=0.05,
        vmax=0.35,
        marker="^",
        s=250,
        alpha=0.9,
        edgecolors="black",
        linewidths=2,
        label="Complex Generated",
    )

    # Colorbar uses the HONEST distance scale
    cbar = plt.colorbar(scatter_simple, ax=ax, pad=0.02)
    cbar.set_label(
        "HONEST Distance: Mean to All Training Docs\n(Not Centroid - More Accurate)",
        fontsize=11,
        fontweight="bold",
    )

    # Annotate training docs with distances
    for x, y, d in zip(train_coords[:, 0], train_coords[:, 1], train_dists):
        ax.annotate(f"{d:.3f}", (x, y), fontsize=8, ha="center", va="bottom")

    # Annotate simple docs
    for i, (x, y, d) in enumerate(
        zip(simple_coords[:, 0], simple_coords[:, 1], simple_dists)
    ):
        ax.annotate(
            f"S{i+1}\n{d:.3f}",
            (x, y),
            fontsize=9,
            ha="center",
            va="bottom",
            fontweight="bold",
            color="darkred",
        )

    # Annotate complex docs
    for i, (x, y, d) in enumerate(
        zip(complex_coords[:, 0], complex_coords[:, 1], complex_dists)
    ):
        ax.annotate(
            f"C{i+1}\n{d:.3f}",
            (x, y),
            fontsize=9,
            ha="center",
            va="bottom",
            fontweight="bold",
            color="darkblue",
        )

    # Mean HONEST distances for quality label
    avg_simple = float(np.mean(simple_dists))
    avg_complex = float(np.mean(complex_dists))
    avg_overall = (avg_simple + avg_complex) / 2.0

    # Same thresholds as true_distance radial plots
    if avg_overall < 0.15:
        quality = "EXCELLENT (Indistinguishable)"
        quality_color = "darkgreen"
    elif avg_overall < 0.25:
        quality = "GOOD (Strong Mimicry)"
        quality_color = "green"
    elif avg_overall < 0.40:
        quality = "FAIR (Moderate Mimicry)"
        quality_color = "orange"
    else:
        quality = "POOR (Weak Mimicry)"
        quality_color = "red"

    ax.set_title(
        (
            f"Mimicry Quality: Rank #{rank} - {author_id} - {quality}\n"
            f"Avg HONEST Distance: Simple={avg_simple:.4f}, "
            f"Complex={avg_complex:.4f}, Overall={avg_overall:.4f}\n"
            f"Green=Closer (Better Mimicry), Red=Further (Worse Mimicry)"
        ),
        fontsize=14,
        fontweight="bold",
        pad=20,
        color=quality_color,
    )

    ax.set_xlabel(f"{method_name} Dimension 1", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{method_name} Dimension 2", fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    textstr = (
        "COLOR encodes TRUE distances in 512D embedding space\n"
        f"NOT the visual distance in this 2D {method_name} projection.\n"
        "Lower distance = Better mimicry (closer to real training docs)."
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir
        / f"mimicry_rank{rank:02d}_{author_id}_{model_key}_run{full_run}.png"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {output_path}")
    return output_path


def create_mimicry_grid(
    all_author_data,
    author_ids_ranked,
    model_key,
    full_run,
    output_dir,
):
    """
    Create a grid of subplots showing mimicry for multiple authors.

    Distances are HONEST mean distances to all training docs; color encodes quality.
    2D layout is PCA-only here for speed; interpret color, not geometric distance.
    """
    n_authors = len(author_ids_ranked)
    n_cols = min(3, n_authors)
    n_rows = (n_authors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 7 * n_rows))
    if n_authors == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, author_id in enumerate(author_ids_ranked):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        author_data = all_author_data[author_id]

        all_embs = np.vstack(
            [
                author_data["training"],
                author_data["simple"],
                author_data["complex"],
            ]
        )
        training_docs = author_data["training"]

        # HONEST distances for all points
        dist_matrix = cdist(all_embs, training_docs, metric="cosine")
        all_dists = dist_matrix.mean(axis=1)

        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(all_embs)

        n_train = len(author_data["training"])
        n_simple = len(author_data["simple"])

        train_coords = coords_2d[:n_train]
        simple_coords = coords_2d[n_train : n_train + n_simple]
        complex_coords = coords_2d[n_train + n_simple :]

        train_dists = all_dists[:n_train]
        simple_dists = all_dists[n_train : n_train + n_simple]
        complex_dists = all_dists[n_train + n_simple :]

        ax.scatter(
            train_coords[:, 0],
            train_coords[:, 1],
            c=train_dists,
            cmap="Greens_r",
            vmin=0.05,
            vmax=0.35,
            marker="o",
            s=100,
            alpha=0.6,
            edgecolors="black",
            linewidths=1,
        )

        ax.scatter(
            simple_coords[:, 0],
            simple_coords[:, 1],
            c=simple_dists,
            cmap="RdYlGn_r",
            vmin=0.05,
            vmax=0.35,
            marker="s",
            s=120,
            alpha=0.9,
            edgecolors="black",
            linewidths=1.5,
        )

        ax.scatter(
            complex_coords[:, 0],
            complex_coords[:, 1],
            c=complex_dists,
            cmap="RdYlGn_r",
            vmin=0.05,
            vmax=0.35,
            marker="^",
            s=120,
            alpha=0.9,
            edgecolors="black",
            linewidths=1.5,
        )

        avg_dist = (float(np.mean(simple_dists)) + float(np.mean(complex_dists))) / 2.0

        ax.set_title(
            f"Rank #{idx+1}: {author_id[:12]}...\nAvg HONEST Dist: {avg_dist:.4f}",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("PCA 1", fontsize=8)
        ax.set_ylabel("PCA 2", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_authors, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    fig.suptitle(
        (
            f"Mimicry Quality Grid: Top {n_authors} Authors\n"
            f"Color: Green=Closer to training style (better mimicry), "
            f"Red=Further from training style (worse mimicry)"
        ),
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()
    output_path = (
        output_dir
        / f"mimicry_grid_top{n_authors}_{model_key}_run{full_run}.png"
    )
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] Grid view: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create mimicry-focused visualizations"
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default="style_embedding",
        choices=STYLE_MODEL_KEYS,
    )
    parser.add_argument("--full-run", type=int, default=1, choices=[1, 2])
    parser.add_argument("--llm-key", type=str, default="gpt-5.1")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--rank-by",
        type=str,
        default="average",
        choices=["simple", "complex", "best", "average"],
    )
    parser.add_argument(
        "--grid-view",
        action="store_true",
        help="Create grid view with all authors",
    )
    parser.add_argument("--method", type=str, default="pca", choices=["pca", "tsne"])

    args = parser.parse_args()

    # Updated CSV path to include LLM key
    csv_path = CONSISTENCY_DIR / f"simple_vs_complex_{args.model_key}_{args.llm_key}_fullrun{args.full_run}.csv"
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        print(f"[HINT] Run: python src/analyse_simple_vs_complex.py --model-key {args.model_key} --llm-key {args.llm_key} --full-run {args.full_run}")
        return

    df = pd.read_csv(csv_path)

    # Use HONEST metrics if available, fallback to legacy
    if "dist_to_training_simple" in df.columns and "dist_to_training_complex" in df.columns:
        simple_col = "dist_to_training_simple"
        complex_col = "dist_to_training_complex"
        print("[INFO] Using HONEST metric: distance to actual training docs")
    else:
        simple_col = "dist_real_centroid_simple"
        complex_col = "dist_real_centroid_complex"
        print(
            "[WARNING] Using LEGACY metric (centroid) - "
            "rerun analyse_simple_vs_complex.py if possible"
        )

    if args.rank_by == "average":
        df["avg_mimicry_dist"] = (df[simple_col] + df[complex_col]) / 2.0
        df_sorted = df.sort_values("avg_mimicry_dist")
    elif args.rank_by == "best":
        df["best_mimicry_dist"] = df[[simple_col, complex_col]].min(axis=1)
        df_sorted = df.sort_values("best_mimicry_dist")
    elif args.rank_by == "simple":
        df_sorted = df.sort_values(simple_col)
    else:
        df_sorted = df.sort_values(complex_col)

    author_ids = df_sorted.head(args.top_n)["author_id"].tolist()

    print(f"\n{'='*80}")
    print(f"Mimicry Visualization: {args.model_key}, Run {args.full_run}")
    print(f"Creating specialized plots for top {args.top_n} authors")
    print(f"{'='*80}\n")

    print("[INFO] Loading selected training indices...")
    selected_indices_map = load_selected_indices()

    print("[INFO] Loading embeddings...")
    all_author_data = {}
    for author_id in tqdm(author_ids, desc="Loading"):
        data = load_author_embeddings(
            author_id, args.model_key, args.llm_key, args.full_run, selected_indices_map
        )
        if data is not None:
            all_author_data[author_id] = data

    if len(all_author_data) == 0:
        print("[ERROR] No data loaded!")
        return

    # Include LLM key in output path to separate GPT and Gemini plots
    output_dir = PLOTS_DIR / args.model_key / args.llm_key / f"fullrun{args.full_run}_mimicry_focus"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[INFO] Creating individual mimicry plots...")
    for rank, author_id in enumerate(author_ids, 1):
        if author_id in all_author_data:
            plot_single_author_mimicry(
                author_id,
                all_author_data[author_id],
                rank,
                args.model_key,
                args.full_run,
                output_dir,
                method=args.method,
            )

    if args.grid_view:
        print("\n[INFO] Creating grid view...")
        create_mimicry_grid(
            all_author_data,
            [aid for aid in author_ids if aid in all_author_data],
            args.model_key,
            args.full_run,
            output_dir,
        )

    print(f"\n[SUCCESS] Mimicry visualizations saved to: {output_dir}\n")


if __name__ == "__main__":
    main()