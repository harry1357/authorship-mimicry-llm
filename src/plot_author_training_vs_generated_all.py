#!/usr/bin/env python3
"""
Generate Global t-SNE Plot: All 157 Authors (Training + Simple + Complex)

Creates ONE t-SNE visualization showing ALL documents from all 157 authors:
- 6 training documents per author (most consistent) = 942 docs
- 2 simple-generated documents per author = 314 docs
- 2 complex-generated documents per author = 314 docs
Total: 1,570 documents

Color-coded by author to visualize:
- Whether generated texts cluster with their author's training documents
- Overall separation between authors
- Relative performance of simple vs complex prompts

Supports multiple visualization types:
- t-SNE (2D/3D): Non-linear dimensionality reduction
- UMAP (2D/3D): Better global structure preservation
- Interactive 3D (Plotly): Rotate, zoom, hover over points

Usage:
    # Default: 2D t-SNE
    python src/plot_author_training_vs_generated_all.py --model-key style_embedding --full-run 1 --top-n 10
    
    # Interactive 3D t-SNE (RECOMMENDED!)
    python src/plot_author_training_vs_generated_all.py --model-key style_embedding --full-run 1 --top-n 10 --interactive
    
    # Interactive 3D UMAP (best global structure + interactivity)
    python src/plot_author_training_vs_generated_all.py --model-key style_embedding --full-run 1 --top-n 10 --viz-type umap --interactive
    
    # Generate all types
    python src/plot_author_training_vs_generated_all.py --model-key style_embedding --full-run 1 --top-n 10 --viz-type all --plot-3d --interactive
"""

import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
from tqdm import tqdm
import pandas as pd
import csv
import ast

# Try to import UMAP (optional)
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Try to import Plotly for interactive plots (optional)
try:
    import plotly.graph_objects as go
    import plotly.express as px
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
    Load pre-computed indices of the 6 most consistent reviews for each author.
    These are the EXACT training reviews used to generate the prompts and analysis.
    """
    mapping = {}
    
    if not REFERENCE_CONSISTENCY_CSV.exists():
        print(f"[WARNING] Reference consistency CSV not found: {REFERENCE_CONSISTENCY_CSV}")
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
                indices = [int(x) for x in raw.replace("[", "").replace("]", "").split(",") if x.strip()]
            indices = [int(i) for i in indices]
            if indices:
                mapping[author_id] = indices
    
    return mapping


def load_author_embeddings(
    author_id: str,
    model_key: str,
    llm_key: str,
    full_run: int,
    selected_indices_map: Dict[str, list] = None,
) -> Dict:
    """
    Load all embeddings for an author: training + simple + complex.
    
    Uses the EXACT 6 training docs that were used for prompts (via selected_indices).
    
    Returns:
        Dict with keys: 'training', 'simple', 'complex'
        Each value is a 2D numpy array of embeddings
    """
    # Load training embeddings (6 most consistent docs)
    train_path = EMBEDDINGS_DIR / model_key / f"{author_id}.npz"
    if not train_path.exists():
        return None
    
    train_data = np.load(train_path, allow_pickle=True)
    
    # Priority: Use selected_indices from CSV (same as prompts/analysis)
    if selected_indices_map and author_id in selected_indices_map:
        selected_idx = selected_indices_map[author_id]
        training_embs = train_data['embeddings'][selected_idx]
    elif 'selected_indices' in train_data:
        # Fallback: indices in npz (unlikely)
        training_embs = train_data['embeddings'][train_data['selected_indices']]
    else:
        # Last resort: first 6 (inconsistent with prompts!)
        training_embs = train_data['embeddings'][:6]
    
    # Load simple generated embeddings
    simple_path = EMBEDDINGS_DIR / "generated" / model_key / llm_key / "simple" / f"fullrun{full_run}" / f"{author_id}.npz"
    if simple_path.exists():
        simple_data = np.load(simple_path, allow_pickle=True)
        simple_embs = simple_data['embeddings']
    else:
        simple_embs = None
    
    # Load complex generated embeddings
    complex_path = EMBEDDINGS_DIR / "generated" / model_key / llm_key / "complex" / f"fullrun{full_run}" / f"{author_id}.npz"
    if complex_path.exists():
        complex_data = np.load(complex_path, allow_pickle=True)
        complex_embs = complex_data['embeddings']
    else:
        complex_embs = None
    
    if simple_embs is None or complex_embs is None:
        return None
    
    return {
        'training': training_embs,
        'simple': simple_embs,
        'complex': complex_embs,
    }


def plot_global_tsne(
    all_data: Dict,
    model_key: str,
    full_run: int,
    output_dir: Path,
    top_n: int = None,
    author_order: List[str] = None,
):
    """
    Create ONE global t-SNE plot with all authors.
    Each author gets a unique color.
    
    Args:
        all_data: Dict mapping author_id -> embeddings dict
        model_key: Style embedding model
        full_run: Experimental run number
        output_dir: Where to save the plot
        top_n: Number of top authors (for title), or None for all
        author_order: List of author IDs in ranking order (best to worst)
    """
    # Collect all embeddings and metadata
    all_embeddings = []
    all_labels = []
    all_types = []  # 'training', 'simple', 'complex'
    all_authors = []
    
    # Create author to index mapping using the provided order (not alphabetical!)
    # This way A1 = best performer, A2 = second best, etc.
    if author_order is not None:
        author_list = author_order
    else:
        author_list = sorted(all_data.keys())
    
    author_to_idx = {author: idx for idx, author in enumerate(author_list)}
    
    for author_id, embs in all_data.items():
        # Training
        for emb in embs['training']:
            all_embeddings.append(emb)
            all_labels.append(author_id)
            all_types.append('training')
            all_authors.append(author_to_idx[author_id])
        
        # Simple
        for emb in embs['simple']:
            all_embeddings.append(emb)
            all_labels.append(author_id)
            all_types.append('simple')
            all_authors.append(author_to_idx[author_id])
        
        # Complex
        for emb in embs['complex']:
            all_embeddings.append(emb)
            all_labels.append(author_id)
            all_types.append('complex')
            all_authors.append(author_to_idx[author_id])
    
    all_embeddings = np.array(all_embeddings)
    all_authors = np.array(all_authors)
    n_docs = len(all_embeddings)
    n_authors = len(all_data)
    
    print(f"[INFO] Total documents: {n_docs}")
    print(f"[INFO] Total authors: {n_authors}")
    print(f"[INFO] Running t-SNE (this may take a few minutes)...")
    
    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(50, n_docs // 4),
        max_iter=1000,
        verbose=1,
    )
    coords_2d = tsne.fit_transform(all_embeddings)
    
    print("[INFO] Creating visualization...")
    
    # Create plot with larger figure
    fig, ax = plt.subplots(figsize=(24, 20))
    
    # Get colormap for authors
    colors = cm.get_cmap('tab20', n_authors)
    if n_authors > 20:
        # Use hsv for many authors
        colors = cm.get_cmap('hsv', n_authors)
    
    # Convert types to arrays for indexing
    types_array = np.array(all_types)
    
    # Plot each author with their unique color
    training_mask = types_array == 'training'
    simple_mask = types_array == 'simple'
    complex_mask = types_array == 'complex'
    
    # Plot training documents
    for author_idx in range(n_authors):
        author_mask = all_authors == author_idx
        combined_mask = training_mask & author_mask
        
        if combined_mask.sum() > 0:
            ax.scatter(
                coords_2d[combined_mask, 0],
                coords_2d[combined_mask, 1],
                c=[colors(author_idx)],
                marker='o',
                s=120,
                alpha=0.6,
                edgecolors='black',
                linewidths=0.5
            )
    
    # Plot simple generated (larger, more visible)
    for author_idx in range(n_authors):
        author_mask = all_authors == author_idx
        combined_mask = simple_mask & author_mask
        
        if combined_mask.sum() > 0:
            ax.scatter(
                coords_2d[combined_mask, 0],
                coords_2d[combined_mask, 1],
                c=[colors(author_idx)],
                marker='s',
                s=200,
                alpha=0.9,
                edgecolors='black',
                linewidths=1.5
            )
    
    # Plot complex generated (larger, more visible)
    for author_idx in range(n_authors):
        author_mask = all_authors == author_idx
        combined_mask = complex_mask & author_mask
        
        if combined_mask.sum() > 0:
            ax.scatter(
                coords_2d[combined_mask, 0],
                coords_2d[combined_mask, 1],
                c=[colors(author_idx)],
                marker='^',
                s=200,
                alpha=0.9,
                edgecolors='black',
                linewidths=1.5
            )
    
    # Add text labels at the TRAINING centroid of each author
    # This shows the "real" author position, and you can see how far generated texts drift
    print("[INFO] Adding author labels at training centroids...")
    for author_idx, author_id in enumerate(author_list):
        author_mask = all_authors == author_idx
        training_author_mask = training_mask & author_mask
        
        if training_author_mask.sum() > 0:
            # Calculate centroid of ONLY training documents
            training_centroid = np.mean(coords_2d[training_author_mask], axis=0)
            
            # Add text label with background box
            ax.text(
                training_centroid[0], training_centroid[1],
                f'A{author_idx+1}',
                fontsize=11,
                fontweight='bold',
                ha='center',
                va='center',
                bbox=dict(
                    boxstyle='round,pad=0.4',
                    facecolor='white',
                    edgecolor=colors(author_idx),
                    linewidth=2,
                    alpha=0.9
                ),
                zorder=100
            )
    
    # Create legend with document types only (not individual authors)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markeredgecolor='black', markersize=12, alpha=0.6, label='Training (6 per author)', linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markeredgecolor='black', markersize=14, alpha=0.9, label='Simple (2 per author)', linestyle='None'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
               markeredgecolor='black', markersize=14, alpha=0.9, label='Complex (2 per author)', linestyle='None'),
    ]
    
    # Styling
    ax.set_xlabel('t-SNE Dimension 1', fontsize=16, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=16, fontweight='bold')
    
    title_prefix = f"Top {top_n}" if top_n else f"All {n_authors}"
    ax.set_title(
        f'Global t-SNE: {title_prefix} Authors - Training + Simple + Complex\n'
        f'Each author has a unique color. Model: {model_key}, Run: {full_run}',
        fontsize=18,
        fontweight='bold',
        pad=20
    )
    
    ax.legend(handles=legend_elements, loc='best', fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_top{top_n}" if top_n else "_all_157"
    output_path = output_dir / f"global_tsne{suffix}_authors_{model_key}_run{full_run}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] Global plot: {output_path}")
    
    # Save author mapping to a text file
    mapping_path = output_dir / f"author_mapping{suffix}_{model_key}_run{full_run}.txt"
    with open(mapping_path, 'w') as f:
        f.write(f"Author Label Mapping for {model_key} (Run {full_run})\n")
        f.write(f"Ranked by best mimicry performance (min of simple/complex distance)\n")
        f.write(f"NOTE: Distances shown are in embedding space, not t-SNE 2D space\n")
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
    top_n: int = None,
    author_order: List[str] = None,
):
    """
    Create ONE global 3D t-SNE plot with all authors.
    Avoids 2D projection artifacts for better visualization of clustering.
    """
    # Collect all embeddings and metadata
    all_embeddings = []
    all_labels = []
    all_types = []
    all_authors = []
    
    if author_order is not None:
        author_list = author_order
    else:
        author_list = sorted(all_data.keys())
    
    author_to_idx = {author: idx for idx, author in enumerate(author_list)}
    
    for author_id, embs in all_data.items():
        # Training
        for emb in embs['training']:
            all_embeddings.append(emb)
            all_labels.append(author_id)
            all_types.append('training')
            all_authors.append(author_to_idx[author_id])
        
        # Simple
        for emb in embs['simple']:
            all_embeddings.append(emb)
            all_labels.append(author_id)
            all_types.append('simple')
            all_authors.append(author_to_idx[author_id])
        
        # Complex
        for emb in embs['complex']:
            all_embeddings.append(emb)
            all_labels.append(author_id)
            all_types.append('complex')
            all_authors.append(author_to_idx[author_id])
    
    all_embeddings = np.array(all_embeddings)
    all_authors = np.array(all_authors)
    n_docs = len(all_embeddings)
    n_authors = len(all_data)
    
    print(f"[INFO] Total documents: {n_docs}")
    print(f"[INFO] Total authors: {n_authors}")
    print(f"[INFO] Running 3D t-SNE (this may take a few minutes)...")
    
    # Run 3D t-SNE
    tsne = TSNE(
        n_components=3,
        random_state=42,
        perplexity=min(50, n_docs // 4),
        max_iter=1000,
        verbose=1,
    )
    coords_3d = tsne.fit_transform(all_embeddings)
    
    print("[INFO] Creating 3D visualization...")
    
    # Create 3D plot
    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get colormap for authors
    colors = cm.get_cmap('tab20', n_authors)
    if n_authors > 20:
        colors = cm.get_cmap('hsv', n_authors)
    
    # Convert types to arrays for indexing
    types_array = np.array(all_types)
    
    # Plot each author with their unique color
    training_mask = types_array == 'training'
    simple_mask = types_array == 'simple'
    complex_mask = types_array == 'complex'
    
    # Plot training documents
    for author_idx in range(n_authors):
        author_mask = all_authors == author_idx
        combined_mask = training_mask & author_mask
        
        if combined_mask.sum() > 0:
            ax.scatter(
                coords_3d[combined_mask, 0],
                coords_3d[combined_mask, 1],
                coords_3d[combined_mask, 2],
                c=[colors(author_idx)],
                marker='o',
                s=80,
                alpha=0.6,
                edgecolors='black',
                linewidths=0.5
            )
    
    # Plot simple generated
    for author_idx in range(n_authors):
        author_mask = all_authors == author_idx
        combined_mask = simple_mask & author_mask
        
        if combined_mask.sum() > 0:
            ax.scatter(
                coords_3d[combined_mask, 0],
                coords_3d[combined_mask, 1],
                coords_3d[combined_mask, 2],
                c=[colors(author_idx)],
                marker='s',
                s=150,
                alpha=0.9,
                edgecolors='black',
                linewidths=1.5
            )
    
    # Plot complex generated
    for author_idx in range(n_authors):
        author_mask = all_authors == author_idx
        combined_mask = complex_mask & author_mask
        
        if combined_mask.sum() > 0:
            ax.scatter(
                coords_3d[combined_mask, 0],
                coords_3d[combined_mask, 1],
                coords_3d[combined_mask, 2],
                c=[colors(author_idx)],
                marker='^',
                s=150,
                alpha=0.9,
                edgecolors='black',
                linewidths=1.5
            )
    
    # Add text labels at training centroid
    print("[INFO] Adding author labels at training centroids...")
    for author_idx, author_id in enumerate(author_list):
        author_mask = all_authors == author_idx
        training_author_mask = training_mask & author_mask
        
        if training_author_mask.sum() > 0:
            # Calculate centroid of ONLY training documents
            training_centroid = np.mean(coords_3d[training_author_mask], axis=0)
            
            # Add text label
            ax.text(
                training_centroid[0],
                training_centroid[1],
                training_centroid[2],
                f'A{author_idx+1}',
                fontsize=10,
                fontweight='bold',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='white',
                    edgecolor=colors(author_idx),
                    linewidth=2,
                    alpha=0.9
                )
            )
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markeredgecolor='black', markersize=10, alpha=0.6, label='Training (6 per author)', linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markeredgecolor='black', markersize=12, alpha=0.9, label='Simple (2 per author)', linestyle='None'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
               markeredgecolor='black', markersize=12, alpha=0.9, label='Complex (2 per author)', linestyle='None'),
    ]
    
    # Styling
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax.set_zlabel('t-SNE Dimension 3', fontsize=12, fontweight='bold')
    
    title_prefix = f"Top {top_n}" if top_n else f"All {n_authors}"
    ax.set_title(
        f'3D t-SNE: {title_prefix} Authors - Training + Simple + Complex\n'
        f'Model: {model_key}, Run: {full_run}',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_top{top_n}" if top_n else "_all_157"
    output_path = output_dir / f"global_tsne_3d{suffix}_authors_{model_key}_run{full_run}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] 3D Global plot: {output_path}")
    
    return output_path


def plot_global_umap(
    all_data: Dict,
    model_key: str,
    full_run: int,
    output_dir: Path,
    top_n: int = None,
    author_order: List[str] = None,
    n_dims: int = 2,
):
    """
    Create global UMAP plot - often better than t-SNE for global structure.
    """
    if not UMAP_AVAILABLE:
        print("[WARNING] UMAP not installed. Install with: pip install umap-learn")
        return None
    
    # Collect embeddings (same as t-SNE)
    all_embeddings = []
    all_labels = []
    all_types = []
    all_authors = []
    
    if author_order is not None:
        author_list = author_order
    else:
        author_list = sorted(all_data.keys())
    
    author_to_idx = {author: idx for idx, author in enumerate(author_list)}
    
    for author_id, embs in all_data.items():
        for emb in embs['training']:
            all_embeddings.append(emb)
            all_labels.append(author_id)
            all_types.append('training')
            all_authors.append(author_to_idx[author_id])
        
        for emb in embs['simple']:
            all_embeddings.append(emb)
            all_labels.append(author_id)
            all_types.append('simple')
            all_authors.append(author_to_idx[author_id])
        
        for emb in embs['complex']:
            all_embeddings.append(emb)
            all_labels.append(author_id)
            all_types.append('complex')
            all_authors.append(author_to_idx[author_id])
    
    all_embeddings = np.array(all_embeddings)
    all_authors = np.array(all_authors)
    n_docs = len(all_embeddings)
    n_authors = len(all_data)
    
    print(f"[INFO] Running {n_dims}D UMAP...")
    
    # Run UMAP
    umap_model = UMAP(
        n_components=n_dims,
        random_state=42,
        n_neighbors=min(15, n_docs // 10),
        min_dist=0.1,
        metric='cosine',
        verbose=True
    )
    coords = umap_model.fit_transform(all_embeddings)
    
    print("[INFO] Creating UMAP visualization...")
    
    # Get colormap
    colors = cm.get_cmap('tab20', n_authors)
    if n_authors > 20:
        colors = cm.get_cmap('hsv', n_authors)
    
    types_array = np.array(all_types)
    training_mask = types_array == 'training'
    simple_mask = types_array == 'simple'
    complex_mask = types_array == 'complex'
    
    if n_dims == 2:
        # 2D plot
        fig, ax = plt.subplots(figsize=(24, 20))
        
        # Plot training
        for author_idx in range(n_authors):
            author_mask = all_authors == author_idx
            combined_mask = training_mask & author_mask
            
            if combined_mask.sum() > 0:
                ax.scatter(
                    coords[combined_mask, 0], coords[combined_mask, 1],
                    c=[colors(author_idx)], marker='o', s=120,
                    alpha=0.6, edgecolors='black', linewidths=0.5
                )
        
        # Plot simple
        for author_idx in range(n_authors):
            author_mask = all_authors == author_idx
            combined_mask = simple_mask & author_mask
            
            if combined_mask.sum() > 0:
                ax.scatter(
                    coords[combined_mask, 0], coords[combined_mask, 1],
                    c=[colors(author_idx)], marker='s', s=200,
                    alpha=0.9, edgecolors='black', linewidths=1.5
                )
        
        # Plot complex
        for author_idx in range(n_authors):
            author_mask = all_authors == author_idx
            combined_mask = complex_mask & author_mask
            
            if combined_mask.sum() > 0:
                ax.scatter(
                    coords[combined_mask, 0], coords[combined_mask, 1],
                    c=[colors(author_idx)], marker='^', s=200,
                    alpha=0.9, edgecolors='black', linewidths=1.5
                )
        
        # Labels at training centroids
        for author_idx, author_id in enumerate(author_list):
            author_mask = all_authors == author_idx
            training_author_mask = training_mask & author_mask
            
            if training_author_mask.sum() > 0:
                training_centroid = np.mean(coords[training_author_mask], axis=0)
                ax.text(
                    training_centroid[0], training_centroid[1],
                    f'A{author_idx+1}',
                    fontsize=11, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                             edgecolor=colors(author_idx), linewidth=2, alpha=0.9),
                    zorder=100
                )
        
        ax.set_xlabel('UMAP Dimension 1', fontsize=16, fontweight='bold')
        ax.set_ylabel('UMAP Dimension 2', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        suffix = f"_top{top_n}" if top_n else "_all_157"
        output_path = output_dir / f"global_umap{suffix}_authors_{model_key}_run{full_run}.png"
        
    else:  # 3D
        fig = plt.figure(figsize=(20, 16))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot training
        for author_idx in range(n_authors):
            author_mask = all_authors == author_idx
            combined_mask = training_mask & author_mask
            
            if combined_mask.sum() > 0:
                ax.scatter(
                    coords[combined_mask, 0], coords[combined_mask, 1], coords[combined_mask, 2],
                    c=[colors(author_idx)], marker='o', s=80,
                    alpha=0.6, edgecolors='black', linewidths=0.5
                )
        
        # Plot simple
        for author_idx in range(n_authors):
            author_mask = all_authors == author_idx
            combined_mask = simple_mask & author_mask
            
            if combined_mask.sum() > 0:
                ax.scatter(
                    coords[combined_mask, 0], coords[combined_mask, 1], coords[combined_mask, 2],
                    c=[colors(author_idx)], marker='s', s=150,
                    alpha=0.9, edgecolors='black', linewidths=1.5
                )
        
        # Plot complex
        for author_idx in range(n_authors):
            author_mask = all_authors == author_idx
            combined_mask = complex_mask & author_mask
            
            if combined_mask.sum() > 0:
                ax.scatter(
                    coords[combined_mask, 0], coords[combined_mask, 1], coords[combined_mask, 2],
                    c=[colors(author_idx)], marker='^', s=150,
                    alpha=0.9, edgecolors='black', linewidths=1.5
                )
        
        # Labels
        for author_idx, author_id in enumerate(author_list):
            author_mask = all_authors == author_idx
            training_author_mask = training_mask & author_mask
            
            if training_author_mask.sum() > 0:
                training_centroid = np.mean(coords[training_author_mask], axis=0)
                ax.text(
                    training_centroid[0], training_centroid[1], training_centroid[2],
                    f'A{author_idx+1}',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor=colors(author_idx), linewidth=2, alpha=0.9)
                )
        
        ax.set_xlabel('UMAP Dimension 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('UMAP Dimension 2', fontsize=12, fontweight='bold')
        ax.set_zlabel('UMAP Dimension 3', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        suffix = f"_top{top_n}" if top_n else "_all_157"
        output_path = output_dir / f"global_umap_3d{suffix}_authors_{model_key}_run{full_run}.png"
    
    # Common elements
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markeredgecolor='black', markersize=12, alpha=0.6, label='Training (6 per author)', linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markeredgecolor='black', markersize=14, alpha=0.9, label='Simple (2 per author)', linestyle='None'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
               markeredgecolor='black', markersize=14, alpha=0.9, label='Complex (2 per author)', linestyle='None'),
    ]
    
    title_prefix = f"Top {top_n}" if top_n else f"All {n_authors}"
    title = f'{n_dims}D UMAP: {title_prefix} Authors - Training + Simple + Complex\n' \
            f'Model: {model_key}, Run: {full_run}'
    
    if n_dims == 2:
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax.legend(handles=legend_elements, loc='best', fontsize=14, framealpha=0.9)
    else:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] {n_dims}D UMAP plot: {output_path}")
    return output_path


def plot_global_interactive_3d(
    all_data: Dict,
    model_key: str,
    full_run: int,
    output_dir: Path,
    top_n: int = None,
    author_order: List[str] = None,
    method: str = 'tsne',
):
    """
    Create interactive 3D plot using Plotly.
    Allows rotation, zoom, hover to inspect individual points.
    
    Args:
        method: 'tsne' or 'umap'
    """
    if not PLOTLY_AVAILABLE:
        print("[WARNING] Plotly not installed. Install with: pip install plotly")
        return None
    
    # Collect embeddings
    all_embeddings = []
    all_labels = []
    all_types = []
    all_authors = []
    
    if author_order is not None:
        author_list = author_order
    else:
        author_list = sorted(all_data.keys())
    
    author_to_idx = {author: idx for idx, author in enumerate(author_list)}
    
    for author_id, embs in all_data.items():
        for emb in embs['training']:
            all_embeddings.append(emb)
            all_labels.append(author_id)
            all_types.append('training')
            all_authors.append(author_to_idx[author_id])
        
        for emb in embs['simple']:
            all_embeddings.append(emb)
            all_labels.append(author_id)
            all_types.append('simple')
            all_authors.append(author_to_idx[author_id])
        
        for emb in embs['complex']:
            all_embeddings.append(emb)
            all_labels.append(author_id)
            all_types.append('complex')
            all_authors.append(author_to_idx[author_id])
    
    all_embeddings = np.array(all_embeddings)
    all_authors = np.array(all_authors)
    n_docs = len(all_embeddings)
    n_authors = len(all_data)
    
    # Run dimensionality reduction
    if method == 'umap':
        if not UMAP_AVAILABLE:
            print("[WARNING] UMAP not available, falling back to t-SNE")
            method = 'tsne'
        else:
            print(f"[INFO] Running 3D UMAP for interactive plot...")
            reducer = UMAP(
                n_components=3,
                random_state=42,
                n_neighbors=min(15, n_docs // 10),
                min_dist=0.1,
                metric='cosine',
                verbose=True
            )
            coords_3d = reducer.fit_transform(all_embeddings)
    
    if method == 'tsne':
        print(f"[INFO] Running 3D t-SNE for interactive plot...")
        reducer = TSNE(
            n_components=3,
            random_state=42,
            perplexity=min(50, n_docs // 4),
            max_iter=1000,
            verbose=1,
        )
        coords_3d = reducer.fit_transform(all_embeddings)
    
    print("[INFO] Creating interactive 3D visualization...")
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'x': coords_3d[:, 0],
        'y': coords_3d[:, 1],
        'z': coords_3d[:, 2],
        'author_id': all_labels,
        'author_label': [f'A{author_to_idx[aid]+1}' for aid in all_labels],
        'doc_type': all_types,
        'author_idx': all_authors,
    })
    
    # Create figure
    fig = go.Figure()
    
    # Color scheme - use discrete colors for each author
    color_map = px.colors.qualitative.Dark24 if n_authors <= 24 else px.colors.sample_colorscale(
        px.colors.sequential.Rainbow, [i/(n_authors-1) for i in range(n_authors)]
    )
    
    # Plot each author
    for author_idx, author_id in enumerate(author_list):
        author_df = df[df['author_id'] == author_id]
        color = color_map[author_idx % len(color_map)]
        
        # Training documents (circles)
        train_df = author_df[author_df['doc_type'] == 'training']
        if len(train_df) > 0:
            fig.add_trace(go.Scatter3d(
                x=train_df['x'],
                y=train_df['y'],
                z=train_df['z'],
                mode='markers',
                name=f'A{author_idx+1} (training)',
                marker=dict(
                    size=6,
                    color=color,
                    symbol='circle',
                    opacity=0.6,
                    line=dict(color='black', width=0.5)
                ),
                hovertemplate=(
                    '<b>%{customdata[0]}</b><br>' +
                    'Type: Training<br>' +
                    'Author: %{customdata[1]}<br>' +
                    '<extra></extra>'
                ),
                customdata=train_df[['author_label', 'author_id']].values,
                showlegend=True,
                legendgroup=f'author_{author_idx}',
            ))
        
        # Simple generated (squares)
        simple_df = author_df[author_df['doc_type'] == 'simple']
        if len(simple_df) > 0:
            fig.add_trace(go.Scatter3d(
                x=simple_df['x'],
                y=simple_df['y'],
                z=simple_df['z'],
                mode='markers',
                name=f'A{author_idx+1} (simple)',
                marker=dict(
                    size=10,
                    color=color,
                    symbol='square',
                    opacity=0.9,
                    line=dict(color='black', width=1)
                ),
                hovertemplate=(
                    '<b>%{customdata[0]}</b><br>' +
                    'Type: Simple Generated<br>' +
                    'Author: %{customdata[1]}<br>' +
                    '<extra></extra>'
                ),
                customdata=simple_df[['author_label', 'author_id']].values,
                showlegend=False,
                legendgroup=f'author_{author_idx}',
            ))
        
        # Complex generated (diamonds)
        complex_df = author_df[author_df['doc_type'] == 'complex']
        if len(complex_df) > 0:
            fig.add_trace(go.Scatter3d(
                x=complex_df['x'],
                y=complex_df['y'],
                z=complex_df['z'],
                mode='markers',
                name=f'A{author_idx+1} (complex)',
                marker=dict(
                    size=10,
                    color=color,
                    symbol='diamond',
                    opacity=0.9,
                    line=dict(color='black', width=1)
                ),
                hovertemplate=(
                    '<b>%{customdata[0]}</b><br>' +
                    'Type: Complex Generated<br>' +
                    'Author: %{customdata[1]}<br>' +
                    '<extra></extra>'
                ),
                customdata=complex_df[['author_label', 'author_id']].values,
                showlegend=False,
                legendgroup=f'author_{author_idx}',
            ))
        
        # Add text label at training centroid
        if len(train_df) > 0:
            centroid = train_df[['x', 'y', 'z']].mean()
            fig.add_trace(go.Scatter3d(
                x=[centroid['x']],
                y=[centroid['y']],
                z=[centroid['z']],
                mode='text',
                text=[f'A{author_idx+1}'],
                textfont=dict(size=12, color='black', family='Arial Black'),
                showlegend=False,
                hoverinfo='skip',
            ))
    
    # Update layout
    title_prefix = f"Top {top_n}" if top_n else f"All {n_authors}"
    method_name = method.upper()
    
    fig.update_layout(
        title=dict(
            text=f'Interactive 3D {method_name}: {title_prefix} Authors<br>' +
                 f'<sub>Training (circles) + Simple (squares) + Complex (diamonds)</sub><br>' +
                 f'<sub>Model: {model_key}, Run: {full_run}</sub>',
            x=0.5,
            xanchor='center',
        ),
        scene=dict(
            xaxis_title=f'{method_name} Dimension 1',
            yaxis_title=f'{method_name} Dimension 2',
            zaxis_title=f'{method_name} Dimension 3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1400,
        height=1000,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        font=dict(size=10),
    )
    
    # Save as HTML (interactive)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_top{top_n}" if top_n else "_all_157"
    output_path = output_dir / f"global_{method}_3d_interactive{suffix}_authors_{model_key}_run{full_run}.html"
    
    fig.write_html(output_path)
    print(f"[SAVED] Interactive 3D {method_name} plot: {output_path}")
    print(f"[INFO] Open in browser: file://{output_path.absolute()}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate global t-SNE plot with all 157 authors (training + simple + complex)"
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default="style_embedding",
        help="Style embedding model to use",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Generate plots for all models",
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
        help="Only plot top N authors (by mimicry performance). Default: all authors",
    )
    parser.add_argument(
        "--rank-by",
        type=str,
        default="best",
        choices=["simple", "complex", "best", "average"],
        help="How to rank authors: 'simple' (simple only), 'complex' (complex only), 'best' (min of both), 'average' (avg of both). Default: best",
    )
    parser.add_argument(
        "--plot-3d",
        action="store_true",
        help="Generate 3D plots (for t-SNE/UMAP) to avoid 2D projection artifacts",
    )
    parser.add_argument(
        "--viz-type",
        type=str,
        default="tsne",
        choices=["tsne", "umap", "all"],
        help="Visualization type: 'tsne' (default), 'umap' (better global structure), 'all' (both)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate interactive 3D plots using Plotly (can rotate, zoom, inspect points)",
    )
    
    args = parser.parse_args()
    
    models = STYLE_MODEL_KEYS if args.all_models else [args.model_key]
    
    for model_key in models:
        print(f"\n{'='*80}")
        print(f"Generating global plot for model: {model_key}")
        print(f"{'='*80}\n")
        
        # Get list of authors
        if args.top_n:
            # Load from analysis CSV to get top performers
            from generation_config import CONSISTENCY_DIR
            import pandas as pd
            
            csv_path = CONSISTENCY_DIR / f"simple_vs_complex_{model_key}_fullrun{args.full_run}.csv"
            if not csv_path.exists():
                print(f"[ERROR] CSV not found: {csv_path}")
                print(f"Run: python src/analyse_simple_vs_complex.py --model-key {model_key} --full-run {args.full_run}")
                continue
            
            df = pd.read_csv(csv_path)
            
            # Use HONEST metrics (distance to training docs) if available, fallback to legacy (centroid)
            if 'dist_to_training_simple' in df.columns:
                simple_col = 'dist_to_training_simple'
                complex_col = 'dist_to_training_complex'
                metric_name = "distance to training docs (HONEST)"
            else:
                simple_col = 'dist_real_centroid_simple'
                complex_col = 'dist_real_centroid_complex'
                metric_name = "distance to centroid (LEGACY - may be optimistic)"
                print(f"[WARNING] Using legacy centroid metric - rerun analyse_simple_vs_complex.py for honest metrics")
            
            # Sort by chosen ranking method
            if args.rank_by == "simple":
                df_sorted = df.sort_values(simple_col)
                rank_desc = f"simple prompt {metric_name}"
            elif args.rank_by == "complex":
                df_sorted = df.sort_values(complex_col)
                rank_desc = f"complex prompt {metric_name}"
            elif args.rank_by == "best":
                df['best_mimicry_dist'] = df[[simple_col, complex_col]].min(axis=1)
                df_sorted = df.sort_values('best_mimicry_dist')
                rank_desc = f"best {metric_name}"
            else:  # average
                df['avg_mimicry_dist'] = df[[simple_col, complex_col]].mean(axis=1)
                df_sorted = df.sort_values('avg_mimicry_dist')
                rank_desc = f"average {metric_name}"
            
            author_ids = df_sorted.head(args.top_n)['author_id'].tolist()
            
            print(f"[INFO] Using top {args.top_n} authors by {rank_desc}")
            
            # Show which authors and their distances
            for idx, row in df_sorted.head(args.top_n).iterrows():
                better = "simple" if row[simple_col] < row[complex_col] else "complex"
                print(f"  A{author_ids.index(row['author_id'])+1}: {row['author_id']} "
                      f"(simple={row[simple_col]:.4f}, "
                      f"complex={row[complex_col]:.4f}, "
                      f"best={better})")
        else:
            # Get all authors with generated texts
            gen_simple_dir = EMBEDDINGS_DIR / "generated" / model_key / args.llm_key / "simple" / f"fullrun{args.full_run}"
            
            if not gen_simple_dir.exists():
                print(f"[ERROR] Generated embeddings not found: {gen_simple_dir}")
                continue
            
            author_files = sorted(gen_simple_dir.glob("*.npz"))
            author_ids = [f.stem for f in author_files]
        
        print(f"[INFO] Found {len(author_ids)} authors to plot")
        
        # Load selected indices (same as used for prompts/analysis)
        print(f"[INFO] Loading selected training indices...")
        selected_indices_map = load_selected_indices()
        print(f"[INFO] Loaded selected indices for {len(selected_indices_map)} authors")
        
        # Load all author data
        all_data = {}
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
        
        if len(all_data) == 0:
            print("[ERROR] No data to plot!")
            continue
        
        # Output directory
        suffix = f"_top{args.top_n}" if args.top_n else "_all"
        output_dir = PLOTS_DIR / model_key / f"fullrun{args.full_run}_global{suffix}"
        
        # Generate global plot(s) based on visualization type
        try:
            if args.viz_type in ["tsne", "all"]:
                # 2D t-SNE
                print("\n--- Generating 2D t-SNE plot ---")
                plot_global_tsne(all_data, model_key, args.full_run, output_dir, args.top_n, author_ids)
                
                # 3D t-SNE if requested
                if args.plot_3d:
                    print("\n--- Generating 3D t-SNE plot ---")
                    plot_global_tsne_3d(all_data, model_key, args.full_run, output_dir, args.top_n, author_ids)
                
                # Interactive 3D t-SNE if requested
                if args.interactive:
                    print("\n--- Generating Interactive 3D t-SNE plot ---")
                    plot_global_interactive_3d(all_data, model_key, args.full_run, output_dir, args.top_n, author_ids, method='tsne')
            
            if args.viz_type in ["umap", "all"]:
                if not UMAP_AVAILABLE:
                    print("\n[WARNING] UMAP not available. Install with: pip install umap-learn")
                else:
                    # 2D UMAP
                    print("\n--- Generating 2D UMAP plot ---")
                    plot_global_umap(all_data, model_key, args.full_run, output_dir, args.top_n, author_ids, n_dims=2)
                    
                    # 3D UMAP if requested
                    if args.plot_3d:
                        print("\n--- Generating 3D UMAP plot ---")
                        plot_global_umap(all_data, model_key, args.full_run, output_dir, args.top_n, author_ids, n_dims=3)
                    
                    # Interactive 3D UMAP if requested
                    if args.interactive:
                        print("\n--- Generating Interactive 3D UMAP plot ---")
                        plot_global_interactive_3d(all_data, model_key, args.full_run, output_dir, args.top_n, author_ids, method='umap')
            
            print(f"\n[SUCCESS] Global plot(s) generated for {model_key}\n")
        except Exception as e:
            print(f"[ERROR] Failed to generate plot: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
