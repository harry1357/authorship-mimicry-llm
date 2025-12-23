#!/usr/bin/env python3
"""
Generate Global t-SNE / UMAP Plot: All 157 Authors (Training + Simple + Complex)

Creates ONE global visualization showing ALL documents from all authors:
- ~6 training documents per author (most consistent, HONEST selection)
- 2 simple-generated documents per author
- 2 complex-generated documents per author

Total ≈ 10 points per author.

Color-coded by author and marker shape by type:
- Training: circles
- Simple: squares
- Complex: triangles

What this lets you see:
- Whether generated texts cluster with their author's training documents
- Overall separation between authors in style space
- Relative performance of simple vs complex prompts

Supports multiple visualization types:
- t-SNE (2D/3D)
- UMAP (2D/3D)
- Plotly interactive 3D (rotate, zoom, hover over points)

Usage examples:

    # Default: 2D t-SNE on a single model
    python src/plot_author_training_vs_generated_all.py --model-key style_embedding --full-run 1

    # 2D + 3D t-SNE, interactive, top-150 authors only
    python src/plot_author_training_vs_generated_all.py \\
        --model-key style_embedding --full-run 1 --top-n 150 \\
        --plot-3d --interactive

    # 2D/3D UMAP + interactive UMAP
    python src/plot_author_training_vs_generated_all.py \\
        --model-key style_embedding --full-run 1 --viz-type umap --plot-3d --interactive

    # Run for all style models
    python src/plot_author_training_vs_generated_all.py --all-models --full-run 1
"""

import argparse
from pathlib import Path
from typing import List, Dict

import ast
import csv

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

# Optional UMAP
try:
    from umap import UMAP

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Optional Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from generation_config import (
    EMBEDDINGS_DIR,
    STYLE_MODEL_KEYS,
    REFERENCE_MODEL_KEY,
    REFERENCE_CONSISTENCY_CSV,
    CONSISTENCY_DIR,
)
from model_configs import PLOTS_DIR


def load_selected_indices() -> Dict[str, list]:
    """
    Load pre-computed indices of the most consistent reviews for each author.

    These indices are the SAME ones used for:
      - training prompts
      - mimicry analysis
      - per-author plots

    We aim to use 6 training docs per author, but if the CSV has fewer valid
    indices or the .npz has fewer embeddings, we degrade gracefully.
    """
    mapping: Dict[str, list] = {}

    if not REFERENCE_CONSISTENCY_CSV.exists():
        print(
            f"[WARNING] Reference consistency CSV not found: "
            f"{REFERENCE_CONSISTENCY_CSV}"
        )
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

    print(f"[INFO] Loaded selected indices for {len(mapping)} authors")
    return mapping


def load_author_embeddings(
    author_id: str,
    model_key: str,
    llm_key: str,
    full_run: int,
    selected_indices_map: Dict[str, list] | None = None,
) -> Dict:
    """
    Load all embeddings for an author: training + simple + complex.

    Training docs:
        • Prefer selected_indices from REFERENCE_CONSISTENCY_CSV (HONEST)
        • Fallback to 'selected_indices' in the .npz (if present)
        • Fallback to the first 6 embeddings

    Returns:
        Dict with keys: 'training', 'simple', 'complex'
        Each value is a 2D numpy array of embeddings
    """
    train_path = EMBEDDINGS_DIR / model_key / f"{author_id}.npz"
    if not train_path.exists():
        return None

    train_data = np.load(train_path, allow_pickle=True)
    train_embs = train_data["embeddings"]

    # Robust selection of training indices
    if selected_indices_map and author_id in selected_indices_map:
        raw_idx = selected_indices_map[author_id]
        valid_idx = [i for i in raw_idx if 0 <= i < len(train_embs)]

        if len(valid_idx) >= 6:
            indices_to_use = sorted(valid_idx)[:6]
        elif len(valid_idx) > 0:
            indices_to_use = sorted(valid_idx)
            print(
                f"[WARNING] Author {author_id}: fewer than 6 valid indices in CSV "
                f"({len(valid_idx)} valid / {len(raw_idx)} total). Using valid subset."
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
                    f"({len(train_embs)}). Using all."
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
                f"[WARNING] Author {author_id}: invalid internal 'selected_indices'; "
                f"falling back to first {len(training_embs)} embeddings."
            )
        else:
            training_embs = train_embs[valid_idx]
    else:
        if len(train_embs) >= 6:
            training_embs = train_embs[:6]
        else:
            training_embs = train_embs

    # Simple generated embeddings
    simple_path = (
        EMBEDDINGS_DIR
        / "generated"
        / model_key
        / llm_key
        / "simple"
        / f"fullrun{full_run}"
        / f"{author_id}.npz"
    )
    if simple_path.exists():
        simple_data = np.load(simple_path, allow_pickle=True)
        simple_embs = simple_data["embeddings"]
    else:
        simple_embs = None

    # Complex generated embeddings
    complex_path = (
        EMBEDDINGS_DIR
        / "generated"
        / model_key
        / llm_key
        / "complex"
        / f"fullrun{full_run}"
        / f"{author_id}.npz"
    )
    if complex_path.exists():
        complex_data = np.load(complex_path, allow_pickle=True)
        complex_embs = complex_data["embeddings"]
    else:
        complex_embs = None

    if simple_embs is None or complex_embs is None:
        return None

    return {
        "training": training_embs,
        "simple": simple_embs,
        "complex": complex_embs,
    }


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2-normalize embeddings to unit length.
    
    This removes magnitude differences and focuses on directional similarity,
    which is appropriate for cosine-distance-based analysis.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return embeddings / norms


def _safe_tsne(
    all_embeddings: np.ndarray, n_components: int, label: str = "t-SNE"
) -> np.ndarray:
    """Run t-SNE with a perplexity that is always valid for the dataset size."""
    n_docs = len(all_embeddings)
    if n_docs < 3:
        raise ValueError(f"Not enough samples ({n_docs}) for {label}")

    # Normalize embeddings to remove magnitude bias
    print(f"[INFO] Normalizing embeddings to unit length...")
    all_embeddings = _normalize_embeddings(all_embeddings)

    # Perplexity must be < n_docs and at least 2.
    base = max(5, n_docs // 4)
    perplexity = min(50, base, n_docs - 1)
    perplexity = max(2, perplexity)

    print(f"[INFO] Running {label} with perplexity={perplexity}, n_docs={n_docs}")

    tsne = TSNE(
        n_components=n_components,
        random_state=42,
        perplexity=perplexity,
        max_iter=1000,
        verbose=1,
    )
    return tsne.fit_transform(all_embeddings)


def plot_global_tsne(
    all_data: Dict,
    model_key: str,
    full_run: int,
    output_dir: Path,
    top_n: int | None = None,
    author_order: List[str] | None = None,
):
    """
    Create ONE global 2D t-SNE plot with all authors.
    Each author gets a unique color; shapes indicate doc type.
    """
    all_embeddings = []
    all_types = []
    all_authors = []

    # Enforce a stable author ordering (ranked list if provided)
    author_list = author_order if author_order is not None else sorted(all_data.keys())
    author_to_idx = {author: idx for idx, author in enumerate(author_list)}

    for author_id, embs in all_data.items():
        idx = author_to_idx[author_id]

        for emb in embs["training"]:
            all_embeddings.append(emb)
            all_types.append("training")
            all_authors.append(idx)

        for emb in embs["simple"]:
            all_embeddings.append(emb)
            all_types.append("simple")
            all_authors.append(idx)

        for emb in embs["complex"]:
            all_embeddings.append(emb)
            all_types.append("complex")
            all_authors.append(idx)

    all_embeddings = np.array(all_embeddings)
    all_authors = np.array(all_authors)
    types_array = np.array(all_types)

    n_docs = len(all_embeddings)
    n_authors = len(all_data)

    print(f"[INFO] Total documents: {n_docs}")
    print(f"[INFO] Total authors: {n_authors}")

    coords_2d = _safe_tsne(all_embeddings, n_components=2, label="2D t-SNE")

    print("[INFO] Creating 2D visualization...")

    fig, ax = plt.subplots(figsize=(24, 20))

    # Colormap for authors - use qualitative for ≤20, rainbow for many authors
    if n_authors <= 20:
        cmap = cm.get_cmap("tab20")
    else:
        # For many authors, use a perceptually distributed colormap
        cmap = cm.get_cmap("gist_rainbow")
    
    # Generate distinct colors for each author
    colors = [cmap(i / max(1, n_authors - 1)) for i in range(n_authors)]

    training_mask = types_array == "training"
    simple_mask = types_array == "simple"
    complex_mask = types_array == "complex"

    # Training docs
    for author_idx in range(n_authors):
        author_mask = all_authors == author_idx
        combined_mask = training_mask & author_mask
        if combined_mask.sum() == 0:
            continue

        ax.scatter(
            coords_2d[combined_mask, 0],
            coords_2d[combined_mask, 1],
            c=[colors[author_idx]],
            marker="o",
            s=120,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
        )

    # Simple generated
    for author_idx in range(n_authors):
        author_mask = all_authors == author_idx
        combined_mask = simple_mask & author_mask
        if combined_mask.sum() == 0:
            continue

        ax.scatter(
            coords_2d[combined_mask, 0],
            coords_2d[combined_mask, 1],
            c=[colors[author_idx]],
            marker="s",
            s=200,
            alpha=0.9,
            edgecolors="black",
            linewidths=1.5,
        )

    # Complex generated
    for author_idx in range(n_authors):
        author_mask = all_authors == author_idx
        combined_mask = complex_mask & author_mask
        if combined_mask.sum() == 0:
            continue

        ax.scatter(
            coords_2d[combined_mask, 0],
            coords_2d[combined_mask, 1],
            c=[colors[author_idx]],
            marker="^",
            s=200,
            alpha=0.9,
            edgecolors="black",
            linewidths=1.5,
        )

    # Label training centroids
    print("[INFO] Adding author labels at training centroids...")
    for author_idx, author_id in enumerate(author_list):
        author_mask = all_authors == author_idx
        training_author_mask = training_mask & author_mask
        if training_author_mask.sum() == 0:
            continue

        training_centroid = np.mean(coords_2d[training_author_mask], axis=0)
        ax.text(
            training_centroid[0],
            training_centroid[1],
            f"A{author_idx+1}",
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor=colors[author_idx],
                linewidth=2,
                alpha=0.9,
            ),
            zorder=100,
        )

    # Legend (doc types only)
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=12,
            alpha=0.6,
            label="Training (≈6 per author)",
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=14,
            alpha=0.9,
            label="Simple (2 per author)",
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=14,
            alpha=0.9,
            label="Complex (2 per author)",
            linestyle="None",
        ),
    ]

    ax.set_xlabel("t-SNE Dimension 1", fontsize=16, fontweight="bold")
    ax.set_ylabel("t-SNE Dimension 2", fontsize=16, fontweight="bold")

    title_prefix = f"Top {top_n}" if top_n else f"All {n_authors}"
    ax.set_title(
        f"Global 2D t-SNE: {title_prefix} Authors - Training + Simple + Complex\n"
        f"Each author has a unique color. Model: {model_key}, Run: {full_run}",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    ax.legend(handles=legend_elements, loc="best", fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_top{top_n}" if top_n else "_all_157"
    output_path = (
        output_dir / f"global_tsne{suffix}_authors_{model_key}_run{full_run}.png"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] Global 2D t-SNE plot: {output_path}")

    # Save author label mapping
    mapping_path = (
        output_dir / f"author_mapping{suffix}_{model_key}_run{full_run}.txt"
    )
    with mapping_path.open("w", encoding="utf-8") as f:
        f.write(f"Author Label Mapping for {model_key} (Run {full_run})\n")
        f.write(
            "Ranked by mimicry performance (see simple_vs_complex_*.csv for exact scores)\n"
        )
        f.write("NOTE: Distances live in the original embedding space, not t-SNE.\n")
        f.write("=" * 60 + "\n\n")
        for idx, author_id in enumerate(author_list):
            f.write(f"A{idx+1:2d} -> {author_id} (Rank {idx+1})\n")

    print(f"[SAVED] Author mapping: {mapping_path}")
    return output_path


def plot_global_tsne_3d(
    all_data: Dict,
    model_key: str,
    full_run: int,
    output_dir: Path,
    top_n: int | None = None,
    author_order: List[str] | None = None,
):
    """
    Create ONE global 3D t-SNE plot with all authors.
    3D often reduces projection overlaps compared to 2D.
    """
    all_embeddings = []
    all_types = []
    all_authors = []

    author_list = author_order if author_order is not None else sorted(all_data.keys())
    author_to_idx = {author: idx for idx, author in enumerate(author_list)}

    for author_id, embs in all_data.items():
        idx = author_to_idx[author_id]

        for emb in embs["training"]:
            all_embeddings.append(emb)
            all_types.append("training")
            all_authors.append(idx)

        for emb in embs["simple"]:
            all_embeddings.append(emb)
            all_types.append("simple")
            all_authors.append(idx)

        for emb in embs["complex"]:
            all_embeddings.append(emb)
            all_types.append("complex")
            all_authors.append(idx)

    all_embeddings = np.array(all_embeddings)
    all_authors = np.array(all_authors)
    types_array = np.array(all_types)

    n_docs = len(all_embeddings)
    n_authors = len(all_data)

    print(f"[INFO] Total documents: {n_docs}")
    print(f"[INFO] Total authors: {n_authors}")

    coords_3d = _safe_tsne(all_embeddings, n_components=3, label="3D t-SNE")

    print("[INFO] Creating 3D visualization...")

    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111, projection="3d")

    # Colormap for authors
    if n_authors <= 20:
        cmap = cm.get_cmap("tab20")
    else:
        cmap = cm.get_cmap("gist_rainbow")
    
    # Generate distinct colors for each author
    colors = [cmap(i / max(1, n_authors - 1)) for i in range(n_authors)]

    training_mask = types_array == "training"
    simple_mask = types_array == "simple"
    complex_mask = types_array == "complex"

    # Training
    for author_idx in range(n_authors):
        author_mask = all_authors == author_idx
        combined_mask = training_mask & author_mask
        if combined_mask.sum() == 0:
            continue

        ax.scatter(
            coords_3d[combined_mask, 0],
            coords_3d[combined_mask, 1],
            coords_3d[combined_mask, 2],
            c=[colors[author_idx]],
            marker="o",
            s=80,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
        )

    # Simple
    for author_idx in range(n_authors):
        author_mask = all_authors == author_idx
        combined_mask = simple_mask & author_mask
        if combined_mask.sum() == 0:
            continue

        ax.scatter(
            coords_3d[combined_mask, 0],
            coords_3d[combined_mask, 1],
            coords_3d[combined_mask, 2],
            c=[colors[author_idx]],
            marker="s",
            s=150,
            alpha=0.9,
            edgecolors="black",
            linewidths=1.5,
        )

    # Complex
    for author_idx in range(n_authors):
        author_mask = all_authors == author_idx
        combined_mask = complex_mask & author_mask
        if combined_mask.sum() == 0:
            continue

        ax.scatter(
            coords_3d[combined_mask, 0],
            coords_3d[combined_mask, 1],
            coords_3d[combined_mask, 2],
            c=[colors[author_idx]],
            marker="^",
            s=150,
            alpha=0.9,
            edgecolors="black",
            linewidths=1.5,
        )

    # Labels at training centroids
    print("[INFO] Adding author labels at training centroids (3D)...")
    for author_idx, author_id in enumerate(author_list):
        author_mask = all_authors == author_idx
        training_author_mask = training_mask & author_mask
        if training_author_mask.sum() == 0:
            continue

        training_centroid = np.mean(coords_3d[training_author_mask], axis=0)
        ax.text(
            training_centroid[0],
            training_centroid[1],
            training_centroid[2],
            f"A{author_idx+1}",
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=colors[author_idx],
                linewidth=2,
                alpha=0.9,
            ),
        )

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=10,
            alpha=0.6,
            label="Training (≈6 per author)",
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=12,
            alpha=0.9,
            label="Simple (2 per author)",
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=12,
            alpha=0.9,
            label="Complex (2 per author)",
            linestyle="None",
        ),
    ]

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12, fontweight="bold")
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12, fontweight="bold")
    ax.set_zlabel("t-SNE Dimension 3", fontsize=12, fontweight="bold")

    title_prefix = f"Top {top_n}" if top_n else f"All {n_authors}"
    ax.set_title(
        f"3D t-SNE: {title_prefix} Authors - Training + Simple + Complex\n"
        f"Model: {model_key}, Run: {full_run}",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    ax.legend(handles=legend_elements, loc="upper left", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_top{top_n}" if top_n else "_all_157"
    output_path = (
        output_dir / f"global_tsne_3d{suffix}_authors_{model_key}_run{full_run}.png"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] 3D Global t-SNE plot: {output_path}")
    return output_path


def plot_global_umap(
    all_data: Dict,
    model_key: str,
    full_run: int,
    output_dir: Path,
    top_n: int | None = None,
    author_order: List[str] | None = None,
    n_dims: int = 2,
):
    """
    Create global UMAP plot (2D or 3D).

    UMAP often preserves global structure better than t-SNE, so this can show
    author clusters and drift more faithfully.
    """
    if not UMAP_AVAILABLE:
        print("[WARNING] UMAP not installed. Install with: pip install umap-learn")
        return None

    all_embeddings = []
    all_types = []
    all_authors = []

    author_list = author_order if author_order is not None else sorted(all_data.keys())
    author_to_idx = {author: idx for idx, author in enumerate(author_list)}

    for author_id, embs in all_data.items():
        idx = author_to_idx[author_id]

        for emb in embs["training"]:
            all_embeddings.append(emb)
            all_types.append("training")
            all_authors.append(idx)

        for emb in embs["simple"]:
            all_embeddings.append(emb)
            all_types.append("simple")
            all_authors.append(idx)

        for emb in embs["complex"]:
            all_embeddings.append(emb)
            all_types.append("complex")
            all_authors.append(idx)

    all_embeddings = np.array(all_embeddings)
    all_authors = np.array(all_authors)
    types_array = np.array(all_types)

    n_docs = len(all_embeddings)
    n_authors = len(all_data)

    # Normalize embeddings to remove magnitude bias
    print(f"[INFO] Normalizing embeddings to unit length...")
    all_embeddings = _normalize_embeddings(all_embeddings)

    print(f"[INFO] Running {n_dims}D UMAP (n_docs={n_docs})...")

    umap_model = UMAP(
        n_components=n_dims,
        random_state=42,
        n_neighbors=min(15, max(5, n_docs // 10)),
        min_dist=0.1,
        metric="cosine",
        verbose=True,
    )
    coords = umap_model.fit_transform(all_embeddings)

    print("[INFO] Creating UMAP visualization...")

    # Colormap for authors
    if n_authors <= 20:
        cmap = cm.get_cmap("tab20")
    else:
        cmap = cm.get_cmap("gist_rainbow")
    
    # Generate distinct colors for each author
    colors = [cmap(i / max(1, n_authors - 1)) for i in range(n_authors)]

    training_mask = types_array == "training"
    simple_mask = types_array == "simple"
    complex_mask = types_array == "complex"

    if n_dims == 2:
        fig, ax = plt.subplots(figsize=(24, 20))

        # Training
        for author_idx in range(n_authors):
            author_mask = all_authors == author_idx
            combined_mask = training_mask & author_mask
            if combined_mask.sum() == 0:
                continue

            ax.scatter(
                coords[combined_mask, 0],
                coords[combined_mask, 1],
                c=[colors[author_idx]],
                marker="o",
                s=120,
                alpha=0.6,
                edgecolors="black",
                linewidths=0.5,
            )

        # Simple
        for author_idx in range(n_authors):
            author_mask = all_authors == author_idx
            combined_mask = simple_mask & author_mask
            if combined_mask.sum() == 0:
                continue

            ax.scatter(
                coords[combined_mask, 0],
                coords[combined_mask, 1],
                c=[colors[author_idx]],
                marker="s",
                s=200,
                alpha=0.9,
                edgecolors="black",
                linewidths=1.5,
            )

        # Complex
        for author_idx in range(n_authors):
            author_mask = all_authors == author_idx
            combined_mask = complex_mask & author_mask
            if combined_mask.sum() == 0:
                continue

            ax.scatter(
                coords[combined_mask, 0],
                coords[combined_mask, 1],
                c=[colors[author_idx]],
                marker="^",
                s=200,
                alpha=0.9,
                edgecolors="black",
                linewidths=1.5,
            )

        # Labels at training centroids
        for author_idx, author_id in enumerate(author_list):
            author_mask = all_authors == author_idx
            training_author_mask = training_mask & author_mask
            if training_author_mask.sum() == 0:
                continue

            training_centroid = np.mean(coords[training_author_mask], axis=0)
            ax.text(
                training_centroid[0],
                training_centroid[1],
                f"A{author_idx+1}",
                fontsize=11,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white",
                    edgecolor=colors[author_idx],
                    linewidth=2,
                    alpha=0.9,
                ),
                zorder=100,
            )

        ax.set_xlabel("UMAP Dimension 1", fontsize=16, fontweight="bold")
        ax.set_ylabel("UMAP Dimension 2", fontsize=16, fontweight="bold")
        ax.grid(True, alpha=0.3)

        suffix = f"_top{top_n}" if top_n else "_all_157"
        output_path = (
            output_dir / f"global_umap{suffix}_authors_{model_key}_run{full_run}.png"
        )

    else:  # 3D UMAP
        fig = plt.figure(figsize=(20, 16))
        ax = fig.add_subplot(111, projection="3d")

        # Training
        for author_idx in range(n_authors):
            author_mask = all_authors == author_idx
            combined_mask = training_mask & author_mask
            if combined_mask.sum() == 0:
                continue

            ax.scatter(
                coords[combined_mask, 0],
                coords[combined_mask, 1],
                coords[combined_mask, 2],
                c=[colors[author_idx]],
                marker="o",
                s=80,
                alpha=0.6,
                edgecolors="black",
                linewidths=0.5,
            )

        # Simple
        for author_idx in range(n_authors):
            author_mask = all_authors == author_idx
            combined_mask = simple_mask & author_mask
            if combined_mask.sum() == 0:
                continue

            ax.scatter(
                coords[combined_mask, 0],
                coords[combined_mask, 1],
                coords[combined_mask, 2],
                c=[colors[author_idx]],
                marker="s",
                s=150,
                alpha=0.9,
                edgecolors="black",
                linewidths=1.5,
            )

        # Complex
        for author_idx in range(n_authors):
            author_mask = all_authors == author_idx
            combined_mask = complex_mask & author_mask
            if combined_mask.sum() == 0:
                continue

            ax.scatter(
                coords[combined_mask, 0],
                coords[combined_mask, 1],
                coords[combined_mask, 2],
                c=[colors[author_idx]],
                marker="^",
                s=150,
                alpha=0.9,
                edgecolors="black",
                linewidths=1.5,
            )

        # Labels
        for author_idx, author_id in enumerate(author_list):
            author_mask = all_authors == author_idx
            training_author_mask = training_mask & author_mask
            if training_author_mask.sum() == 0:
                continue

            training_centroid = np.mean(coords[training_author_mask], axis=0)
            ax.text(
                training_centroid[0],
                training_centroid[1],
                training_centroid[2],
                f"A{author_idx+1}",
                fontsize=10,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=colors[author_idx],
                    linewidth=2,
                    alpha=0.9,
                ),
            )

        ax.set_xlabel("UMAP Dimension 1", fontsize=12, fontweight="bold")
        ax.set_ylabel("UMAP Dimension 2", fontsize=12, fontweight="bold")
        ax.set_zlabel("UMAP Dimension 3", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        suffix = f"_top{top_n}" if top_n else "_all_157"
        output_path = (
            output_dir
            / f"global_umap_3d{suffix}_authors_{model_key}_run{full_run}.png"
        )

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=12,
            alpha=0.6,
            label="Training (≈6 per author)",
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=14,
            alpha=0.9,
            label="Simple (2 per author)",
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=14,
            alpha=0.9,
            label="Complex (2 per author)",
            linestyle="None",
        ),
    ]

    title_prefix = f"Top {top_n}" if top_n else f"All {n_authors}"
    title = (
        f"{n_dims}D UMAP: {title_prefix} Authors - Training + Simple + Complex\n"
        f"Model: {model_key}, Run: {full_run}"
    )

    if n_dims == 2:
        ax.set_title(title, fontsize=18, fontweight="bold", pad=20)
        ax.legend(handles=legend_elements, loc="best", fontsize=14, framealpha=0.9)
    else:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.legend(handles=legend_elements, loc="upper left", fontsize=11, framealpha=0.9)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {n_dims}D UMAP plot: {output_path}")
    return output_path


def plot_global_interactive_3d(
    all_data: Dict,
    model_key: str,
    full_run: int,
    output_dir: Path,
    top_n: int | None = None,
    author_order: List[str] | None = None,
    method: str = "tsne",
):
    """
    Create interactive 3D plot using Plotly.
    You can rotate, zoom, and hover to inspect individual points.
    """
    if not PLOTLY_AVAILABLE:
        print("[WARNING] Plotly not installed. Install with: pip install plotly")
        return None

    all_embeddings = []
    all_labels = []
    all_types = []
    all_authors = []

    author_list = author_order if author_order is not None else sorted(all_data.keys())
    author_to_idx = {author: idx for idx, author in enumerate(author_list)}

    for author_id, embs in all_data.items():
        idx = author_to_idx[author_id]

        for emb in embs["training"]:
            all_embeddings.append(emb)
            all_labels.append(author_id)
            all_types.append("training")
            all_authors.append(idx)

        for emb in embs["simple"]:
            all_embeddings.append(emb)
            all_labels.append(author_id)
            all_types.append("simple")
            all_authors.append(idx)

        for emb in embs["complex"]:
            all_embeddings.append(emb)
            all_labels.append(author_id)
            all_types.append("complex")
            all_authors.append(idx)

    all_embeddings = np.array(all_embeddings)
    all_authors = np.array(all_authors)
    n_docs = len(all_embeddings)
    n_authors = len(all_data)

    # Normalize embeddings to remove magnitude bias
    print(f"[INFO] Normalizing embeddings to unit length...")
    all_embeddings = _normalize_embeddings(all_embeddings)

    # Dimensionality reduction
    if method == "umap":
        if not UMAP_AVAILABLE:
            print("[WARNING] UMAP not available, falling back to t-SNE")
            method = "tsne"
        else:
            print("[INFO] Running 3D UMAP for interactive plot...")
            reducer = UMAP(
                n_components=3,
                random_state=42,
                n_neighbors=min(15, max(5, n_docs // 10)),
                min_dist=0.1,
                metric="cosine",
                verbose=True,
            )
            coords_3d = reducer.fit_transform(all_embeddings)

    if method == "tsne":
        coords_3d = _safe_tsne(all_embeddings, n_components=3, label="3D t-SNE")

    print("[INFO] Creating interactive 3D visualization...")

    df = pd.DataFrame(
        {
            "x": coords_3d[:, 0],
            "y": coords_3d[:, 1],
            "z": coords_3d[:, 2],
            "author_id": all_labels,
            "author_idx": all_authors,
            "doc_type": all_types,
        }
    )
    df["author_label"] = [
        f"A{author_to_idx[aid] + 1}" for aid in df["author_id"].tolist()
    ]

    fig = go.Figure()

    # Color scheme for authors
    if n_authors <= 24:
        color_map = px.colors.qualitative.Dark24
    else:
        color_map = px.colors.sample_colorscale(
            px.colors.sequential.Rainbow,
            [i / (n_authors - 1) for i in range(n_authors)],
        )

    for author_idx, author_id in enumerate(author_list):
        author_df = df[df["author_id"] == author_id]
        color = color_map[author_idx % len(color_map)]

        train_df = author_df[author_df["doc_type"] == "training"]
        simple_df = author_df[author_df["doc_type"] == "simple"]
        complex_df = author_df[author_df["doc_type"] == "complex"]

        # Training
        if len(train_df) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=train_df["x"],
                    y=train_df["y"],
                    z=train_df["z"],
                    mode="markers",
                    name=f"A{author_idx+1} (training)",
                    marker=dict(
                        size=6,
                        color=color,
                        symbol="circle",
                        opacity=0.6,
                        line=dict(color="black", width=0.5),
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Type: Training<br>"
                        "Author: %{customdata[1]}<br>"
                        "<extra></extra>"
                    ),
                    customdata=train_df[["author_label", "author_id"]].values,
                    showlegend=True,
                    legendgroup=f"author_{author_idx}",
                )
            )

        # Simple
        if len(simple_df) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=simple_df["x"],
                    y=simple_df["y"],
                    z=simple_df["z"],
                    mode="markers",
                    name=f"A{author_idx+1} (simple)",
                    marker=dict(
                        size=10,
                        color=color,
                        symbol="square",
                        opacity=0.9,
                        line=dict(color="black", width=1),
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Type: Simple Generated<br>"
                        "Author: %{customdata[1]}<br>"
                        "<extra></extra>"
                    ),
                    customdata=simple_df[["author_label", "author_id"]].values,
                    showlegend=False,
                    legendgroup=f"author_{author_idx}",
                )
            )

        # Complex
        if len(complex_df) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=complex_df["x"],
                    y=complex_df["y"],
                    z=complex_df["z"],
                    mode="markers",
                    name=f"A{author_idx+1} (complex)",
                    marker=dict(
                        size=10,
                        color=color,
                        symbol="diamond",
                        opacity=0.9,
                        line=dict(color="black", width=1),
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Type: Complex Generated<br>"
                        "Author: %{customdata[1]}<br>"
                        "<extra></extra>"
                    ),
                    customdata=complex_df[["author_label", "author_id"]].values,
                    showlegend=False,
                    legendgroup=f"author_{author_idx}",
                )
            )

        # Label at training centroid
        if len(train_df) > 0:
            centroid = train_df[["x", "y", "z"]].mean()
            fig.add_trace(
                go.Scatter3d(
                    x=[centroid["x"]],
                    y=[centroid["y"]],
                    z=[centroid["z"]],
                    mode="text",
                    text=[f"A{author_idx+1}"],
                    textfont=dict(size=12, color="black", family="Arial Black"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    title_prefix = f"Top {top_n}" if top_n else f"All {n_authors}"
    method_name = method.upper()

    fig.update_layout(
        title=dict(
            text=(
                f"Interactive 3D {method_name}: {title_prefix} Authors<br>"
                "<sub>Training (circles) + Simple (squares) + Complex (diamonds)</sub><br>"
                f"<sub>Model: {model_key}, Run: {full_run}</sub>"
            ),
            x=0.5,
            xanchor="center",
        ),
        scene=dict(
            xaxis_title=f"{method_name} Dimension 1",
            yaxis_title=f"{method_name} Dimension 2",
            zaxis_title=f"{method_name} Dimension 3",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        width=1400,
        height=1000,
        hovermode="closest",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        font=dict(size=10),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_top{top_n}" if top_n else "_all_157"
    output_path = (
        output_dir
        / f"global_{method}_3d_interactive{suffix}_authors_{model_key}_run{full_run}.html"
    )

    fig.write_html(output_path)
    print(f"[SAVED] Interactive 3D {method_name} plot: {output_path}")
    print(f"[INFO] Open in browser: file://{output_path.absolute()}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate global TSNE/UMAP plots with all authors "
            "(training + simple + complex)."
        )
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default="style_embedding",
        choices=STYLE_MODEL_KEYS,
        help="Style embedding model to use",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Generate plots for all style embedding models",
    )
    parser.add_argument(
        "--llm-key",
        type=str,
        default="gpt-5.1",
        help="LLM used for generation",
    )
    parser.add_argument(
        "--full-run",
        type=int,
        default=1,
        choices=[1, 2],
        help="Experimental run number",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Only plot top N authors (ranked by mimicry performance). Default: all.",
    )
    parser.add_argument(
        "--rank-by",
        type=str,
        default="best",
        choices=["simple", "complex", "best", "average"],
        help=(
            "How to rank authors: 'simple', 'complex', 'best' (min of both), "
            "'average' (mean of both). Default: best."
        ),
    )
    parser.add_argument(
        "--plot-3d",
        action="store_true",
        help="Generate 3D plots (for TSNE/UMAP) in addition to 2D.",
    )
    parser.add_argument(
        "--viz-type",
        type=str,
        default="tsne",
        choices=["tsne", "umap", "all"],
        help="Visualization type(s) to generate.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate interactive 3D plots using Plotly.",
    )

    args = parser.parse_args()
    models = STYLE_MODEL_KEYS if args.all_models else [args.model_key]

    for model_key in models:
        print("\n" + "=" * 80)
        print(f"Generating global plot for model: {model_key}")
        print("=" * 80 + "\n")

        # Load CSV and rank authors by mimicry performance
        # Updated CSV path to include LLM key
        csv_path = (
            CONSISTENCY_DIR
            / f"simple_vs_complex_{model_key}_{args.llm_key}_fullrun{args.full_run}.csv"
        )
        if not csv_path.exists():
            print(f"[ERROR] CSV not found: {csv_path}")
            print(
                "Run: python src/analyse_simple_vs_complex.py "
                f"--model-key {model_key} --llm-key {args.llm_key} --full-run {args.full_run}"
            )
            continue

        df = pd.read_csv(csv_path)
        
        # Filter out comment rows (summary metrics at bottom of CSV)
        df = df[~df['author_id'].astype(str).str.startswith('#')]

        # HONEST metrics if available
        if "dist_to_training_simple" in df.columns:
            simple_col = "dist_to_training_simple"
            complex_col = "dist_to_training_complex"
            metric_name = "distance to training docs (HONEST)"
        else:
            simple_col = "dist_real_centroid_simple"
            complex_col = "dist_real_centroid_complex"
            metric_name = "distance to centroid (LEGACY; slightly optimistic)"
            print(
                "[WARNING] Using legacy centroid metric; "
                "rerun analyse_simple_vs_complex.py for HONEST metrics."
            )

        # Sort by ranking criterion
        if args.rank_by == "simple":
            df_sorted = df.sort_values(simple_col)
            rank_desc = f"simple prompt {metric_name}"
        elif args.rank_by == "complex":
            df_sorted = df.sort_values(complex_col)
            rank_desc = f"complex prompt {metric_name}"
        elif args.rank_by == "best":
            df["best_mimicry_dist"] = df[[simple_col, complex_col]].min(axis=1)
            df_sorted = df.sort_values("best_mimicry_dist")
            rank_desc = f"best ({metric_name})"
        else:
            df["avg_mimicry_dist"] = df[[simple_col, complex_col]].mean(axis=1)
            df_sorted = df.sort_values("avg_mimicry_dist")
            rank_desc = f"average ({metric_name})"

        # Get author IDs in ranked order (top-N or all)
        if args.top_n:
            author_ids = df_sorted.head(args.top_n)["author_id"].tolist()
            print(f"[INFO] Using top {args.top_n} authors by {rank_desc}")
        else:
            author_ids = df_sorted["author_id"].tolist()
            print(f"[INFO] Using all {len(author_ids)} authors ranked by {rank_desc}")

        # Sanity printout of top authors + distances
        display_count = min(10, len(author_ids))
        for idx, row in df_sorted.head(display_count).iterrows():
            better = "simple" if row[simple_col] < row[complex_col] else "complex"
            rank_num = author_ids.index(row['author_id']) + 1
            print(
                f"  A{rank_num:3d}: {row['author_id']}  "
                f"(simple={row[simple_col]:.4f}, "
                f"complex={row[complex_col]:.4f}, best={better})"
            )

        print(f"[INFO] Found {len(author_ids)} authors to plot")

        # Load selected indices used for prompts/analysis
        print("[INFO] Loading selected training indices...")
        selected_indices_map = load_selected_indices()

        # Load embeddings for all authors
        all_data: Dict[str, Dict] = {}
        missing_count = 0

        for author_id in tqdm(author_ids, desc="Loading embeddings"):
            embeddings = load_author_embeddings(
                author_id, model_key, args.llm_key, args.full_run, selected_indices_map
            )
            if embeddings is None:
                missing_count += 1
                continue
            all_data[author_id] = embeddings

        print(f"[INFO] Loaded {len(all_data)} authors successfully")
        if missing_count > 0:
            print(f"[WARNING] {missing_count} authors skipped (missing data)")

        if not all_data:
            print("[ERROR] No data to plot for this model, skipping.")
            continue

        suffix = f"_top{args.top_n}" if args.top_n else "_all"
        # Include LLM key in output path to separate GPT and Gemini plots
        output_dir = PLOTS_DIR / model_key / args.llm_key / f"fullrun{args.full_run}_global{suffix}"

        try:
            # TSNE-based plots
            if args.viz_type in ["tsne", "all"]:
                print("\n--- Generating 2D t-SNE plot ---")
                plot_global_tsne(
                    all_data, model_key, args.full_run, output_dir, args.top_n, author_ids
                )

                if args.plot_3d:
                    print("\n--- Generating 3D t-SNE plot ---")
                    plot_global_tsne_3d(
                        all_data,
                        model_key,
                        args.full_run,
                        output_dir,
                        args.top_n,
                        author_ids,
                    )

                if args.interactive:
                    print("\n--- Generating Interactive 3D t-SNE plot ---")
                    plot_global_interactive_3d(
                        all_data,
                        model_key,
                        args.full_run,
                        output_dir,
                        args.top_n,
                        author_ids,
                        method="tsne",
                    )

            # UMAP-based plots
            if args.viz_type in ["umap", "all"]:
                if not UMAP_AVAILABLE:
                    print(
                        "\n[WARNING] UMAP not available. Install with: pip install umap-learn"
                    )
                else:
                    print("\n--- Generating 2D UMAP plot ---")
                    plot_global_umap(
                        all_data,
                        model_key,
                        args.full_run,
                        output_dir,
                        args.top_n,
                        author_ids,
                        n_dims=2,
                    )

                    if args.plot_3d:
                        print("\n--- Generating 3D UMAP plot ---")
                        plot_global_umap(
                            all_data,
                            model_key,
                            args.full_run,
                            output_dir,
                            args.top_n,
                            author_ids,
                            n_dims=3,
                        )

                    if args.interactive:
                        print("\n--- Generating Interactive 3D UMAP plot ---")
                        plot_global_interactive_3d(
                            all_data,
                            model_key,
                            args.full_run,
                            output_dir,
                            args.top_n,
                            author_ids,
                            method="umap",
                        )

            print(f"\n[SUCCESS] Global plot(s) generated for {model_key}\n")

        except Exception as e:
            print(f"[ERROR] Failed to generate plot for {model_key}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()