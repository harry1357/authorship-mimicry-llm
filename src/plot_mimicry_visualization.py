#!/usr/bin/env python3
"""
Specialized Mimicry Visualization

Creates plots optimized for ASSESSING MIMICRY QUALITY, not author separation.

Unlike the global 157-author plots (which maximize separation), these plots:
1. Show each author individually with their generated texts
2. Use color to indicate distance from training centroid  
3. Display TRUE embedding distances (not distorted visual distances)
4. Create per-author subplots for easy comparison

Usage:
    # Individual author plots for top 10
    python src/plot_mimicry_visualization.py --model-key style_embedding --full-run 1 --top-n 10
    
    # Grid view of all top authors
    python src/plot_mimicry_visualization.py --model-key style_embedding --full-run 1 --top-n 10 --grid-view
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import csv
import ast

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

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
                indices = [int(x) for x in raw.replace("[", "").replace("]", "").split(",") if x.strip()]
            indices = [int(i) for i in indices]
            if indices:
                mapping[author_id] = indices
    return mapping


def load_author_embeddings(author_id, model_key, llm_key, full_run, selected_indices_map):
    """Load training + generated embeddings with consistent training doc selection."""
    train_path = EMBEDDINGS_DIR / model_key / f"{author_id}.npz"
    if not train_path.exists():
        return None
    
    train_data = np.load(train_path, allow_pickle=True)
    
    # Use selected indices from CSV (same as analysis/prompts)
    if author_id in selected_indices_map:
        selected_idx = selected_indices_map[author_id]
        training_embs = train_data['embeddings'][selected_idx]
    elif 'selected_indices' in train_data:
        training_embs = train_data['embeddings'][train_data['selected_indices']]
    else:
        training_embs = train_data['embeddings'][:6]
    
    simple_path = EMBEDDINGS_DIR / "generated" / model_key / llm_key / "simple" / f"fullrun{full_run}" / f"{author_id}.npz"
    if not simple_path.exists():
        return None
    simple_embs = np.load(simple_path, allow_pickle=True)['embeddings']
    
    complex_path = EMBEDDINGS_DIR / "generated" / model_key / llm_key / "complex" / f"fullrun{full_run}" / f"{author_id}.npz"
    if not complex_path.exists():
        return None
    complex_embs = np.load(complex_path, allow_pickle=True)['embeddings']
    
    return {
        'training': training_embs,
        'simple': simple_embs,
        'complex': complex_embs,
    }


def plot_single_author_mimicry(
    author_id,
    author_data,
    rank,
    model_key,
    full_run,
    output_dir,
    method='pca'
):
    """
    Create a single-author plot showing mimicry quality.
    Uses color to show distance from training centroid.
    """
    # Combine all embeddings
    all_embs = np.vstack([
        author_data['training'],
        author_data['simple'],
        author_data['complex']
    ])
    
    # Compute HONEST distances: mean distance to ALL training docs (not just centroid)
    # This is more accurate than distance to centroid which can be optimistic
    training_docs = author_data['training']
    all_dists_to_training = cdist(all_embs, training_docs, metric='cosine')
    all_mean_dists_to_training = np.mean(all_dists_to_training, axis=1)  # Mean distance to all training docs
    
    # Project to 2D for visualization
    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(all_embs)
        method_name = "PCA"
    else:  # tsne
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, len(all_embs)//2))
        coords_2d = reducer.fit_transform(all_embs)
        method_name = "t-SNE"
    
    # Split coordinates
    n_train = len(author_data['training'])
    n_simple = len(author_data['simple'])
    n_complex = len(author_data['complex'])
    
    train_coords = coords_2d[:n_train]
    simple_coords = coords_2d[n_train:n_train+n_simple]
    complex_coords = coords_2d[n_train+n_simple:]
    
    train_dists = all_mean_dists_to_training[:n_train]
    simple_dists = all_mean_dists_to_training[n_train:n_train+n_simple]
    complex_dists = all_mean_dists_to_training[n_train+n_simple:]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot training docs (gray circles)
    scatter_train = ax.scatter(
        train_coords[:, 0], train_coords[:, 1],
        c=train_dists, cmap='Greens_r', vmin=0.05, vmax=0.5,
        marker='o', s=200, alpha=0.7,
        edgecolors='black', linewidths=2,
        label='Training'
    )
    
    # Plot simple generated (color by distance)
    scatter_simple = ax.scatter(
        simple_coords[:, 0], simple_coords[:, 1],
        c=simple_dists, cmap='RdYlGn_r', vmin=0.05, vmax=0.5,
        marker='s', s=250, alpha=0.9,
        edgecolors='black', linewidths=2,
        label='Simple Generated'
    )
    
    # Plot complex generated (color by distance)
    scatter_complex = ax.scatter(
        complex_coords[:, 0], complex_coords[:, 1],
        c=complex_dists, cmap='RdYlGn_r', vmin=0.05, vmax=0.5,
        marker='^', s=250, alpha=0.9,
        edgecolors='black', linewidths=2,
        label='Complex Generated'
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter_simple, ax=ax, pad=0.02)
    cbar.set_label('HONEST Distance: Mean to All Training Docs\n(Not Centroid - More Accurate)', 
                   fontsize=11, fontweight='bold')
    
    # Annotate each point with its distance
    for i, (x, y, d) in enumerate(zip(train_coords[:, 0], train_coords[:, 1], train_dists)):
        ax.annotate(f'{d:.3f}', (x, y), fontsize=8, ha='center', va='bottom')
    
    for i, (x, y, d) in enumerate(zip(simple_coords[:, 0], simple_coords[:, 1], simple_dists)):
        ax.annotate(f'S{i+1}\n{d:.3f}', (x, y), fontsize=9, ha='center', va='bottom', 
                   fontweight='bold', color='darkred')
    
    for i, (x, y, d) in enumerate(zip(complex_coords[:, 0], complex_coords[:, 1], complex_dists)):
        ax.annotate(f'C{i+1}\n{d:.3f}', (x, y), fontsize=9, ha='center', va='bottom',
                   fontweight='bold', color='darkblue')
    
    # Compute average distances
    avg_simple = np.mean(simple_dists)
    avg_complex = np.mean(complex_dists)
    avg_overall = (avg_simple + avg_complex) / 2
    
    # Determine quality (same thresholds as true_distances.py)
    if avg_overall < 0.25:
        quality = "EXCELLENT"
        quality_color = "darkgreen"
    elif avg_overall < 0.35:
        quality = "GOOD"
        quality_color = "green"
    elif avg_overall < 0.45:
        quality = "FAIR"
        quality_color = "orange"
    else:
        quality = "POOR"
        quality_color = "red"
    
    # Title with mimicry stats
    ax.set_title(
        f'Mimicry Quality: Rank #{rank} - {author_id} - {quality}\n'
        f'Average Distance: Simple={avg_simple:.4f}, Complex={avg_complex:.4f}, Overall={avg_overall:.4f}\n'
        f'Green=Close (Good Mimicry), Red=Far (Poor Mimicry)',
        fontsize=14, fontweight='bold', pad=20, color=quality_color
    )
    
    ax.set_xlabel(f'{method_name} Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{method_name} Dimension 2', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add text box with explanation
    textstr = (
        'Color indicates TRUE distance in 512D embedding space\n'
        f'NOT the visual distance in this 2D {method_name} projection!\n'
        'Lower distance = Better mimicry'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"mimicry_rank{rank:02d}_{author_id}_{model_key}_run{full_run}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] {output_path}")
    return output_path


def create_mimicry_grid(
    all_author_data,
    author_ids_ranked,
    model_key,
    full_run,
    output_dir
):
    """
    Create a grid of subplots showing mimicry for multiple authors.
    """
    n_authors = len(author_ids_ranked)
    n_cols = min(3, n_authors)
    n_rows = (n_authors + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 7*n_rows))
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
        
        # Combine embeddings
        all_embs = np.vstack([
            author_data['training'],
            author_data['simple'],
            author_data['complex']
        ])
        
        # Compute centroid and distances
        training_centroid = np.mean(author_data['training'], axis=0, keepdims=True)
        all_dists = cdist(all_embs, training_centroid, metric='cosine').flatten()
        
        # Project to 2D
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(all_embs)
        
        # Split
        n_train = len(author_data['training'])
        n_simple = len(author_data['simple'])
        
        train_coords = coords_2d[:n_train]
        simple_coords = coords_2d[n_train:n_train+n_simple]
        complex_coords = coords_2d[n_train+n_simple:]
        
        train_dists = all_dists[:n_train]
        simple_dists = all_dists[n_train:n_train+n_simple]
        complex_dists = all_dists[n_train+n_simple:]
        
        # Plot
        ax.scatter(train_coords[:, 0], train_coords[:, 1], 
                  c=train_dists, cmap='Greens_r', vmin=0, vmax=0.3,
                  marker='o', s=100, alpha=0.6, edgecolors='black', linewidths=1)
        
        ax.scatter(simple_coords[:, 0], simple_coords[:, 1],
                  c=simple_dists, cmap='RdYlGn_r', vmin=0, vmax=0.3,
                  marker='s', s=120, alpha=0.9, edgecolors='black', linewidths=1.5)
        
        ax.scatter(complex_coords[:, 0], complex_coords[:, 1],
                  c=complex_dists, cmap='RdYlGn_r', vmin=0, vmax=0.3,
                  marker='^', s=120, alpha=0.9, edgecolors='black', linewidths=1.5)
        
        # Stats
        avg_dist = (np.mean(simple_dists) + np.mean(complex_dists)) / 2
        
        ax.set_title(f'Rank #{idx+1}: {author_id[:12]}...\nAvg Dist: {avg_dist:.4f}',
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('PCA 1', fontsize=8)
        ax.set_ylabel('PCA 2', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_authors, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    # Super title
    fig.suptitle(
        f'Mimicry Quality Grid: Top {n_authors} Authors\n'
        f'Color: Green=Good Mimicry (close to training), Red=Poor Mimicry (far from training)',
        fontsize=16, fontweight='bold', y=0.995
    )
    
    plt.tight_layout()
    output_path = output_dir / f"mimicry_grid_top{n_authors}_{model_key}_run{full_run}.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] Grid view: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create mimicry-focused visualizations")
    parser.add_argument("--model-key", type=str, default="style_embedding")
    parser.add_argument("--full-run", type=int, default=1, choices=[1, 2])
    parser.add_argument("--llm-key", type=str, default="gpt-5.1")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--rank-by", type=str, default="average", 
                       choices=["simple", "complex", "best", "average"])
    parser.add_argument("--grid-view", action="store_true",
                       help="Create grid view with all authors")
    parser.add_argument("--method", type=str, default="pca", choices=["pca", "tsne"])
    
    args = parser.parse_args()
    
    # Load ranking
    csv_path = CONSISTENCY_DIR / f"simple_vs_complex_{args.model_key}_fullrun{args.full_run}.csv"
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Use HONEST metrics if available, fallback to legacy
    if 'dist_to_training_simple' in df.columns:
        simple_col = 'dist_to_training_simple'
        complex_col = 'dist_to_training_complex'
        print("[INFO] Using HONEST metric: distance to actual training docs")
    else:
        simple_col = 'dist_real_centroid_simple'
        complex_col = 'dist_real_centroid_complex'
        print("[WARNING] Using LEGACY metric (centroid) - rerun analyse_simple_vs_complex.py")
    
    # Sort
    if args.rank_by == "average":
        df['avg_mimicry_dist'] = (df[simple_col] + df[complex_col]) / 2
        df_sorted = df.sort_values('avg_mimicry_dist')
    elif args.rank_by == "best":
        df['best_mimicry_dist'] = df[[simple_col, complex_col]].min(axis=1)
        df_sorted = df.sort_values('best_mimicry_dist')
    elif args.rank_by == "simple":
        df_sorted = df.sort_values(simple_col)
    else:
        df_sorted = df.sort_values(complex_col)
    
    author_ids = df_sorted.head(args.top_n)['author_id'].tolist()
    
    print(f"\n{'='*80}")
    print(f"Mimicry Visualization: {args.model_key}, Run {args.full_run}")
    print(f"Creating specialized plots for top {args.top_n} authors")
    print(f"{'='*80}\n")
    
    # Load selected indices (consistent with analysis/prompts)
    print("[INFO] Loading selected training indices...")
    selected_indices_map = load_selected_indices()
    
    # Load all data
    print("[INFO] Loading embeddings...")
    all_author_data = {}
    for author_id in tqdm(author_ids, desc="Loading"):
        data = load_author_embeddings(author_id, args.model_key, args.llm_key, args.full_run, selected_indices_map)
        if data is not None:
            all_author_data[author_id] = data
    
    if len(all_author_data) == 0:
        print("[ERROR] No data loaded!")
        return
    
    output_dir = PLOTS_DIR / args.model_key / f"fullrun{args.full_run}_mimicry_focus"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create individual plots
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
                method=args.method
            )
    
    # Create grid view
    if args.grid_view:
        print("\n[INFO] Creating grid view...")
        create_mimicry_grid(
            all_author_data,
            [aid for aid in author_ids if aid in all_author_data],
            args.model_key,
            args.full_run,
            output_dir
        )
    
    print(f"\n[SUCCESS] Mimicry visualizations saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
