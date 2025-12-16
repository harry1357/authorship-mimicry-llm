#!/usr/bin/env python3
"""
Task 2: Compare 157 Experimental Authors Against All Corpus Authors

Creates visualizations showing how unique/distinctive the 157 experimental
authors are compared to all ~2,144 authors in the corpus.

Uses ONLY training documents (no generated texts).

Generates:
1. t-SNE plot (2D): 2D visualization of all authors
2. PCA plot (2D): Shows first 2 principal components in embedding space
3. Distance distribution plot: Shows how far 157 authors are from others

Usage:
    # Generate both t-SNE and PCA
    python src/plot_global_157_vs_all.py --model-key style_embedding --full-run 1
    
    # Only regenerate PCA (when t-SNE is already good)
    python src/plot_global_157_vs_all.py --model-key style_embedding --full-run 1 --plot-type pca
    
    # Generate for all models (only PCA)
    python src/plot_global_157_vs_all.py --all-models --full-run 1 --plot-type pca
"""

import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from tqdm import tqdm

from generation_config import (
    EMBEDDINGS_DIR,
    STYLE_MODEL_KEYS,
    CONSISTENCY_DIR,
)
from model_configs import PLOTS_DIR


def load_all_author_embeddings(model_key: str) -> Dict[str, np.ndarray]:
    """
    Load embeddings for ALL authors in the corpus.
    
    Returns:
        Dict mapping author_id -> centroid embedding (mean of 6 selected docs)
    """
    embeddings_dir = EMBEDDINGS_DIR / model_key
    
    if not embeddings_dir.exists():
        print(f"[ERROR] Embeddings directory not found: {embeddings_dir}")
        return {}
    
    author_embeddings = {}
    
    for npz_file in tqdm(list(embeddings_dir.glob("*.npz")), desc="Loading all authors"):
        author_id = npz_file.stem
        
        try:
            data = np.load(npz_file, allow_pickle=True)
            
            # Get the 6 most consistent training docs
            if 'selected_indices' in data:
                selected_idx = data['selected_indices']
                selected_embs = data['embeddings'][selected_idx]
            else:
                # Fallback: use first 6
                selected_embs = data['embeddings'][:6]
            
            # Use centroid as author representation
            centroid = np.mean(selected_embs, axis=0)
            author_embeddings[author_id] = centroid
            
        except Exception as e:
            print(f"[WARNING] Failed to load {author_id}: {e}")
            continue
    
    return author_embeddings


def get_experimental_authors(model_key: str, full_run: int) -> Set[str]:
    """
    Get the list of experimental authors (100 for Phase 2, 157 for original runs).
    """
    # First try Phase 2 consistency CSV
    phase2_csv = CONSISTENCY_DIR / f"{model_key}_phase2_top100.csv"
    if phase2_csv.exists():
        import pandas as pd
        df = pd.read_csv(phase2_csv)
        return set(df['author_id'].tolist())
    
    # Fallback to original runs
    csv_path = CONSISTENCY_DIR / f"simple_vs_complex_{model_key}_fullrun{full_run}.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        return set(df['author_id'].tolist())
    else:
        # Fallback: get from generated embeddings directory
        gen_dir = EMBEDDINGS_DIR / "generated" / model_key / "gpt-5.1" / "simple" / f"fullrun{full_run}"
        if gen_dir.exists():
            return {f.stem for f in gen_dir.glob("*.npz")}
    
    return set()


def calculate_uniqueness_scores(
    exp_coords: np.ndarray,
    other_coords: np.ndarray,
    exp_ids: List[str],
) -> List[Tuple[str, float, int]]:
    """
    Calculate uniqueness score for each experimental author.
    
    Uniqueness = average distance to 10 nearest other authors.
    Higher score = more unique style.
    
    Returns:
        List of (author_id, uniqueness_score, rank) sorted by uniqueness
    """
    from scipy.spatial.distance import cdist
    
    # Calculate distances from each experimental author to all other authors
    distances = cdist(exp_coords, other_coords, metric='euclidean')
    
    # For each experimental author, get average distance to 10 nearest others
    uniqueness_scores = []
    for idx, author_id in enumerate(exp_ids):
        # Get distances to all other authors
        dists = distances[idx]
        # Average of 10 nearest neighbors
        nearest_10 = np.sort(dists)[:10]
        avg_dist = np.mean(nearest_10)
        uniqueness_scores.append((author_id, avg_dist))
    
    # Sort by uniqueness (highest = most unique)
    uniqueness_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Add rank
    ranked = [(aid, score, rank+1) for rank, (aid, score) in enumerate(uniqueness_scores)]
    
    return ranked


def plot_tsne_comparison(
    all_embeddings: Dict[str, np.ndarray],
    experimental_ids: Set[str],
    model_key: str,
    full_run: int,
    output_dir: Path,
):
    """
    Create t-SNE plot comparing 157 experimental authors vs all others.
    With density heatmap and uniqueness labels.
    """
    print("[INFO] Preparing data for t-SNE...")
    
    # Separate experimental and other authors
    exp_ids = []
    exp_embs = []
    other_ids = []
    other_embs = []
    
    for author_id, emb in all_embeddings.items():
        if author_id in experimental_ids:
            exp_ids.append(author_id)
            exp_embs.append(emb)
        else:
            other_ids.append(author_id)
            other_embs.append(emb)
    
    exp_embs = np.array(exp_embs)
    other_embs = np.array(other_embs)
    
    # Combine all
    all_embs = np.vstack([exp_embs, other_embs])
    
    n_exp = len(exp_ids)
    n_other = len(other_ids)
    n_total = n_exp + n_other
    
    print(f"[INFO] Experimental authors: {n_exp}")
    print(f"[INFO] Other authors: {n_other}")
    print(f"[INFO] Total authors: {n_total}")
    print(f"[INFO] Running t-SNE (this may take several minutes)...")
    
    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(50, n_total // 4),
        max_iter=1000,
        verbose=1,
    )
    coords_2d = tsne.fit_transform(all_embs)
    
    # Split coordinates
    exp_coords = coords_2d[:n_exp]
    other_coords = coords_2d[n_exp:]
    
    print("[INFO] Creating t-SNE visualization...")
    
    # Calculate uniqueness scores for experimental authors
    uniqueness_scores = calculate_uniqueness_scores(exp_coords, other_coords, exp_ids)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(24, 20))
    
    # Create density heatmap from other authors (background)
    print("[INFO] Computing density heatmap...")
    try:
        kde = gaussian_kde(other_coords.T)
        
        # Create grid
        x_min, x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
        y_min, y_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()
        xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        density = kde(positions).reshape(xx.shape)
        
        # Plot density as heatmap
        im = ax.contourf(xx, yy, density, levels=20, cmap='YlOrRd', alpha=0.3)
        plt.colorbar(im, ax=ax, label='Style Space Density', pad=0.02)
    except Exception as e:
        print(f"[WARNING] Could not compute density heatmap: {e}")
    
    # Plot other authors (gray, small, background)
    ax.scatter(
        other_coords[:, 0],
        other_coords[:, 1],
        c='lightgray',
        marker='o',
        s=15,
        alpha=0.4,
        label=f'Other authors (n={n_other})',
        edgecolors='none',
        zorder=1
    )
    
    # Plot experimental authors (colored by uniqueness)
    # Most unique = dark red, most typical = light red
    uniqueness_dict = {aid: score for aid, score, _ in uniqueness_scores}
    colors_exp = [uniqueness_dict[aid] for aid in exp_ids]
    
    scatter = ax.scatter(
        exp_coords[:, 0],
        exp_coords[:, 1],
        c=colors_exp,
        cmap='RdYlGn_r',  # Red = unique, Yellow/Green = typical
        marker='o',
        s=100,
        alpha=0.8,
        label=f'{n_exp} Experimental authors',
        edgecolors='black',
        linewidths=1.5,
        zorder=3
    )
    
    cbar = plt.colorbar(scatter, ax=ax, label='Uniqueness Score\n(higher = more unique)', pad=0.02)
    
    # Label top 5 most unique and top 5 most typical
    print("[INFO] Labeling most unique and typical authors...")
    
    # Top 5 unique (highest scores)
    for author_id, score, rank in uniqueness_scores[:5]:
        idx = exp_ids.index(author_id)
        ax.annotate(
            f'{author_id[:8]}...\n(Rank {rank})',
            xy=(exp_coords[idx, 0], exp_coords[idx, 1]),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color='darkred',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', edgecolor='darkred', linewidth=2, alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
            zorder=10
        )
    
    # Top 5 typical (lowest scores)
    for author_id, score, rank in uniqueness_scores[-5:]:
        idx = exp_ids.index(author_id)
        ax.annotate(
            f'{author_id[:8]}...\n(Rank {rank})',
            xy=(exp_coords[idx, 0], exp_coords[idx, 1]),
            xytext=(10, -20),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color='darkgreen',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='darkgreen', linewidth=2, alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5),
            zorder=10
        )
    
    # Styling
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    ax.set_title(
        f'{n_exp} Experimental Authors vs All Corpus Authors\n'
        f'Model: {model_key}, Total: {n_total} authors',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    
    ax.legend(loc='best', fontsize=12, framealpha=0.9, markerscale=2)
    ax.grid(True, alpha=0.3)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"tsne_157_vs_all_{model_key}_run{full_run}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] t-SNE plot: {output_path}")
    
    # Save uniqueness ranking to text file
    ranking_path = output_dir / f"uniqueness_ranking_{model_key}_run{full_run}.txt"
    with open(ranking_path, 'w') as f:
        f.write(f"Author Uniqueness Ranking for {model_key} (Run {full_run})\n")
        f.write(f"Uniqueness = avg distance to 10 nearest other authors in t-SNE space\n")
        f.write(f"Higher score = more unique/distinctive style\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("TOP 10 MOST UNIQUE AUTHORS:\n")
        f.write("-" * 70 + "\n")
        for author_id, score, rank in uniqueness_scores[:10]:
            f.write(f"Rank {rank:3d}: {author_id:20s} (uniqueness = {score:.4f})\n")
        
        f.write("\n" + "=" * 70 + "\n\n")
        
        f.write("TOP 10 MOST TYPICAL AUTHORS:\n")
        f.write("-" * 70 + "\n")
        for author_id, score, rank in uniqueness_scores[-10:]:
            f.write(f"Rank {rank:3d}: {author_id:20s} (uniqueness = {score:.4f})\n")
    
    print(f"[SAVED] Uniqueness ranking: {ranking_path}")
    
    return output_path


def plot_pca_comparison(
    all_embeddings: Dict[str, np.ndarray],
    experimental_ids: Set[str],
    model_key: str,
    full_run: int,
    output_dir: Path,
):
    """
    Create PCA plot showing first 2 principal components (raw embedding space).
    With density heatmap and uniqueness labels like t-SNE.
    """
    print("[INFO] Preparing data for PCA...")
    
    # Separate experimental and other authors
    exp_ids = []
    exp_embs = []
    other_ids = []
    other_embs = []
    
    for author_id, emb in all_embeddings.items():
        if author_id in experimental_ids:
            exp_ids.append(author_id)
            exp_embs.append(emb)
        else:
            other_ids.append(author_id)
            other_embs.append(emb)
    
    exp_embs = np.array(exp_embs)
    other_embs = np.array(other_embs)
    
    # Combine all
    all_embs = np.vstack([exp_embs, other_embs])
    
    n_exp = len(exp_ids)
    n_other = len(other_ids)
    
    print(f"[INFO] Running PCA on {len(all_embs)} authors...")
    
    # Run PCA
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(all_embs)
    
    # Split coordinates
    exp_coords = coords_2d[:n_exp]
    other_coords = coords_2d[n_exp:]
    
    print(f"[INFO] PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")
    
    # Calculate uniqueness scores
    uniqueness_scores = calculate_uniqueness_scores(exp_coords, other_coords, exp_ids)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(24, 20))
    
    # Create density heatmap from other authors (background)
    print("[INFO] Computing density heatmap...")
    try:
        kde = gaussian_kde(other_coords.T)
        
        # Create grid
        x_min, x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
        y_min, y_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()
        xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        density = kde(positions).reshape(xx.shape)
        
        # Plot density as heatmap
        im = ax.contourf(xx, yy, density, levels=20, cmap='YlOrRd', alpha=0.3)
        plt.colorbar(im, ax=ax, label='Style Space Density', pad=0.02)
    except Exception as e:
        print(f"[WARNING] Could not compute density heatmap: {e}")
    
    # Plot other authors (gray, small, background)
    ax.scatter(
        other_coords[:, 0],
        other_coords[:, 1],
        c='lightgray',
        marker='o',
        s=15,
        alpha=0.4,
        label=f'Other authors (n={n_other})',
        edgecolors='none',
        zorder=1
    )
    
    # Plot experimental authors (colored by uniqueness)
    uniqueness_dict = {aid: score for aid, score, _ in uniqueness_scores}
    colors_exp = [uniqueness_dict[aid] for aid in exp_ids]
    
    scatter = ax.scatter(
        exp_coords[:, 0],
        exp_coords[:, 1],
        c=colors_exp,
        cmap='RdYlGn_r',
        marker='o',
        s=100,
        alpha=0.8,
        label=f'157 Experimental authors',
        edgecolors='black',
        linewidths=1.5,
        zorder=3
    )
    
    cbar = plt.colorbar(scatter, ax=ax, label='Uniqueness Score\n(higher = more unique)', pad=0.02)
    
    # Label top 5 most unique and top 5 most typical
    print("[INFO] Labeling most unique and typical authors...")
    
    # Top 5 unique
    for author_id, score, rank in uniqueness_scores[:5]:
        idx = exp_ids.index(author_id)
        ax.annotate(
            f'{author_id[:8]}...\n(Rank {rank})',
            xy=(exp_coords[idx, 0], exp_coords[idx, 1]),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color='darkred',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', edgecolor='darkred', linewidth=2, alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
            zorder=10
        )
    
    # Top 5 typical
    for author_id, score, rank in uniqueness_scores[-5:]:
        idx = exp_ids.index(author_id)
        ax.annotate(
            f'{author_id[:8]}...\n(Rank {rank})',
            xy=(exp_coords[idx, 0], exp_coords[idx, 1]),
            xytext=(10, -20),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color='darkgreen',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='darkgreen', linewidth=2, alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5),
            zorder=10
        )
    
    # Styling
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14, fontweight='bold')
    ax.set_title(
        f'157 Experimental Authors vs All Corpus Authors (PCA)\n'
        f'Embedding Space Visualization - Model: {model_key}',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    
    ax.legend(loc='best', fontsize=12, framealpha=0.9, markerscale=2)
    ax.grid(True, alpha=0.3)
    
    # Save
    output_path = output_dir / f"pca_157_vs_all_{model_key}_run{full_run}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] PCA plot: {output_path}")
    
    return output_path


def plot_tsne_3d(
    all_embeddings: Dict[str, np.ndarray],
    experimental_ids: Set[str],
    model_key: str,
    full_run: int,
    output_dir: Path,
):
    """
    Create 3D t-SNE plot to avoid 2D projection artifacts.
    """
    print("[INFO] Preparing data for 3D t-SNE...")
    
    # Separate experimental and other authors
    exp_ids = []
    exp_embs = []
    other_ids = []
    other_embs = []
    
    for author_id, emb in all_embeddings.items():
        if author_id in experimental_ids:
            exp_ids.append(author_id)
            exp_embs.append(emb)
        else:
            other_ids.append(author_id)
            other_embs.append(emb)
    
    exp_embs = np.array(exp_embs)
    other_embs = np.array(other_embs)
    
    # Combine all
    all_embs = np.vstack([exp_embs, other_embs])
    
    n_exp = len(exp_ids)
    n_other = len(other_ids)
    n_total = n_exp + n_other
    
    print(f"[INFO] Experimental authors: {n_exp}")
    print(f"[INFO] Other authors: {n_other}")
    print(f"[INFO] Total authors: {n_total}")
    print(f"[INFO] Running 3D t-SNE (this may take several minutes)...")
    
    # Run t-SNE with 3 components
    tsne = TSNE(
        n_components=3,
        random_state=42,
        perplexity=min(50, n_total // 4),
        max_iter=1000,
        verbose=1,
    )
    coords_3d = tsne.fit_transform(all_embs)
    
    # Split coordinates
    exp_coords = coords_3d[:n_exp]
    other_coords = coords_3d[n_exp:]
    
    # Calculate uniqueness scores (using 3D coordinates)
    uniqueness_scores = calculate_uniqueness_scores(exp_coords, other_coords, exp_ids)
    uniqueness_dict = {aid: score for aid, score, _ in uniqueness_scores}
    colors_exp = [uniqueness_dict[aid] for aid in exp_ids]
    
    print("[INFO] Creating 3D visualization...")
    
    # Create 3D plot
    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot other authors (gray, small, background)
    ax.scatter(
        other_coords[:, 0],
        other_coords[:, 1],
        other_coords[:, 2],
        c='lightgray',
        marker='o',
        s=10,
        alpha=0.2,
        label=f'Other authors (n={n_other})',
        edgecolors='none'
    )
    
    # Plot experimental authors (colored by uniqueness)
    scatter = ax.scatter(
        exp_coords[:, 0],
        exp_coords[:, 1],
        exp_coords[:, 2],
        c=colors_exp,
        cmap='RdYlGn_r',
        marker='o',
        s=80,
        alpha=0.8,
        label=f'157 Experimental authors',
        edgecolors='black',
        linewidths=1
    )
    
    cbar = plt.colorbar(scatter, ax=ax, label='Uniqueness Score\n(higher = more unique)', pad=0.1, shrink=0.8)
    
    # Styling
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax.set_zlabel('t-SNE Dimension 3', fontsize=12, fontweight='bold')
    ax.set_title(
        f'3D t-SNE: 157 Experimental Authors vs All Corpus Authors\n'
        f'Model: {model_key}, Total: {n_total} authors',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"tsne_3d_157_vs_all_{model_key}_run{full_run}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] 3D t-SNE plot: {output_path}")
    
    return output_path


def plot_pca_3d(
    all_embeddings: Dict[str, np.ndarray],
    experimental_ids: Set[str],
    model_key: str,
    full_run: int,
    output_dir: Path,
):
    """
    Create 3D PCA plot showing first 3 principal components.
    """
    print("[INFO] Preparing data for 3D PCA...")
    
    # Separate experimental and other authors
    exp_ids = []
    exp_embs = []
    other_ids = []
    other_embs = []
    
    for author_id, emb in all_embeddings.items():
        if author_id in experimental_ids:
            exp_ids.append(author_id)
            exp_embs.append(emb)
        else:
            other_ids.append(author_id)
            other_embs.append(emb)
    
    exp_embs = np.array(exp_embs)
    other_embs = np.array(other_embs)
    
    # Combine all
    all_embs = np.vstack([exp_embs, other_embs])
    
    n_exp = len(exp_ids)
    n_other = len(other_ids)
    
    print(f"[INFO] Running 3D PCA on {len(all_embs)} authors...")
    
    # Run PCA with 3 components
    pca = PCA(n_components=3, random_state=42)
    coords_3d = pca.fit_transform(all_embs)
    
    # Split coordinates
    exp_coords = coords_3d[:n_exp]
    other_coords = coords_3d[n_exp:]
    
    print(f"[INFO] PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.1%}, PC3={pca.explained_variance_ratio_[2]:.1%}")
    
    # Calculate uniqueness scores
    uniqueness_scores = calculate_uniqueness_scores(exp_coords, other_coords, exp_ids)
    uniqueness_dict = {aid: score for aid, score, _ in uniqueness_scores}
    colors_exp = [uniqueness_dict[aid] for aid in exp_ids]
    
    # Create 3D plot
    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot other authors (gray, small, background)
    ax.scatter(
        other_coords[:, 0],
        other_coords[:, 1],
        other_coords[:, 2],
        c='lightgray',
        marker='o',
        s=10,
        alpha=0.2,
        label=f'Other authors (n={n_other})',
        edgecolors='none'
    )
    
    # Plot experimental authors (colored by uniqueness)
    scatter = ax.scatter(
        exp_coords[:, 0],
        exp_coords[:, 1],
        exp_coords[:, 2],
        c=colors_exp,
        cmap='RdYlGn_r',
        marker='o',
        s=80,
        alpha=0.8,
        label=f'157 Experimental authors',
        edgecolors='black',
        linewidths=1
    )
    
    cbar = plt.colorbar(scatter, ax=ax, label='Uniqueness Score\n(higher = more unique)', pad=0.1, shrink=0.8)
    
    # Styling
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12, fontweight='bold')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'3D PCA: 157 Experimental Authors vs All Corpus Authors\n'
        f'Embedding Space Visualization - Model: {model_key}',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Save
    output_path = output_dir / f"pca_3d_157_vs_all_{model_key}_run{full_run}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] 3D PCA plot: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare 157 experimental authors against all corpus authors"
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
        "--full-run",
        type=int,
        default=1,
        choices=[1, 2],
        help="Experimental run number",
    )
    parser.add_argument(
        "--plot-3d",
        action="store_true",
        help="Generate 3D plots (t-SNE and PCA) to avoid 2D projection artifacts",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        default="both",
        choices=["tsne", "pca", "both"],
        help="Which plots to generate: tsne, pca, or both (default: both)",
    )
    
    args = parser.parse_args()
    
    models = STYLE_MODEL_KEYS if args.all_models else [args.model_key]
    
    for model_key in models:
        print(f"\n{'='*80}")
        print(f"Processing model: {model_key}")
        print(f"{'='*80}\n")
        
        # Load all author embeddings
        all_embeddings = load_all_author_embeddings(model_key)
        
        if not all_embeddings:
            print(f"[ERROR] No embeddings loaded for {model_key}")
            continue
        
        # Get experimental authors
        experimental_ids = get_experimental_authors(model_key, args.full_run)
        
        if not experimental_ids:
            print(f"[ERROR] No experimental authors found for {model_key}")
            continue
        
        print(f"[INFO] Found {len(experimental_ids)} experimental authors")
        print(f"[INFO] Total corpus authors: {len(all_embeddings)}")
        
        # Output directory
        output_dir = PLOTS_DIR / model_key / f"fullrun{args.full_run}_global_comparison"
        
        # Generate plots
        try:
            if args.plot_type in ["tsne", "both"]:
                print("\n--- Generating 2D t-SNE plot ---")
                plot_tsne_comparison(all_embeddings, experimental_ids, model_key, args.full_run, output_dir)
                
                if args.plot_3d:
                    print("\n--- Generating 3D t-SNE plot ---")
                    plot_tsne_3d(all_embeddings, experimental_ids, model_key, args.full_run, output_dir)
            
            if args.plot_type in ["pca", "both"]:
                print("\n--- Generating 2D PCA plot ---")
                plot_pca_comparison(all_embeddings, experimental_ids, model_key, args.full_run, output_dir)
                
                if args.plot_3d:
                    print("\n--- Generating 3D PCA plot ---")
                    plot_pca_3d(all_embeddings, experimental_ids, model_key, args.full_run, output_dir)
            
            print(f"\n[SUCCESS] Plots generated for {model_key}\n")
        except Exception as e:
            print(f"[ERROR] Failed to generate plots: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
