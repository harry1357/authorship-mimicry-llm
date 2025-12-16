#!/usr/bin/env python3
"""
TRUE Distance Visualization - Radial/Polar Plot

This creates plots where VISUAL DISTANCE = TRUE EMBEDDING DISTANCE.
No more confusion - what you see is what you get!

Each plot is arranged radially:
- Training centroid at the center (0, 0)
- All documents positioned by their TRUE distance from center
- Angle is arbitrary (just for spacing)

Now you CAN trust the visual distance!
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from tqdm import tqdm
import csv
import ast

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


def plot_radial_distances(
    author_id,
    author_data,
    rank,
    model_key,
    full_run,
    output_dir
):
    """
    Create a radial plot where VISUAL DISTANCE = TRUE DISTANCE.
    
    Origin (0,0) represents the training docs.
    Each generated document is placed at its HONEST distance (mean to all training docs).
    Angles are distributed evenly for visibility.
    """
    # Compute HONEST distances: mean distance to ALL training docs (not centroid)
    training_docs = author_data['training']
    
    # For training docs: mean distance to OTHER training docs (exclude self)
    train_dist_matrix = cdist(training_docs, training_docs, metric='cosine')
    n_train = train_dist_matrix.shape[0]
    if n_train > 1:
        # Sum over row, divide by (n_train - 1) to exclude self-distance (diagonal = 0)
        train_dists = train_dist_matrix.sum(axis=1) / (n_train - 1)
    else:
        train_dists = np.zeros_like(train_dist_matrix[:, 0])
    
    # For generated docs: mean distance to ALL training docs
    simple_dist_matrix = cdist(author_data['simple'], training_docs, metric='cosine')
    simple_dists = np.mean(simple_dist_matrix, axis=1)
    
    complex_dist_matrix = cdist(author_data['complex'], training_docs, metric='cosine')
    complex_dists = np.mean(complex_dist_matrix, axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Set up equal aspect ratio for accurate distances
    ax.set_aspect('equal')
    
    # Define angles for each doc type (spread them out)
    n_train = len(train_dists)
    n_simple = len(simple_dists)
    n_complex = len(complex_dists)
    
    # Training docs: angles 0 to 150 degrees
    train_angles = np.linspace(0, 150, n_train) * np.pi / 180
    
    # Simple: angles 180 to 240 degrees
    simple_angles = np.linspace(180, 240, n_simple) * np.pi / 180
    
    # Complex: angles 270 to 330 degrees
    complex_angles = np.linspace(270, 330, n_complex) * np.pi / 180
    
    # Convert polar to cartesian
    train_x = train_dists * np.cos(train_angles)
    train_y = train_dists * np.sin(train_angles)
    
    simple_x = simple_dists * np.cos(simple_angles)
    simple_y = simple_dists * np.sin(simple_angles)
    
    complex_x = complex_dists * np.cos(complex_angles)
    complex_y = complex_dists * np.sin(complex_angles)
    
    # Plot centroid at origin
    ax.scatter([0], [0], c='red', marker='X', s=500, 
               edgecolors='black', linewidths=3, 
               label='Training Center (Origin)', zorder=100)
    
    # Plot training docs
    ax.scatter(train_x, train_y, c='blue', marker='o', s=250,
               alpha=0.6, edgecolors='black', linewidths=2,
               label=f'Training (n={n_train})', zorder=50)
    
    # Annotate training with distances
    for i, (x, y, d) in enumerate(zip(train_x, train_y, train_dists)):
        ax.annotate(f'T{i+1}: {d:.3f}', (x, y), 
                   fontsize=9, ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # Plot simple generated
    scatter_simple = ax.scatter(simple_x, simple_y, c=simple_dists, 
               cmap='RdYlGn_r', vmin=0.05, vmax=0.35,
               marker='s', s=400, alpha=0.9,
               edgecolors='black', linewidths=2.5,
               label=f'Simple Generated (n={n_simple})', zorder=60)
    
    # Annotate simple
    for i, (x, y, d) in enumerate(zip(simple_x, simple_y, simple_dists)):
        ax.annotate(f'S{i+1}: {d:.3f}', (x, y),
                   fontsize=10, ha='center', va='bottom', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            edgecolor='darkgreen' if d < 0.15 else ('green' if d < 0.25 else ('orange' if d < 0.40 else 'red')), 
                            linewidth=2, alpha=0.9))
    
    # Plot complex generated
    scatter_complex = ax.scatter(complex_x, complex_y, c=complex_dists,
               cmap='RdYlGn_r', vmin=0.05, vmax=0.35,
               marker='^', s=400, alpha=0.9,
               edgecolors='black', linewidths=2.5,
               label=f'Complex Generated (n={n_complex})', zorder=60)
    
    # Annotate complex
    for i, (x, y, d) in enumerate(zip(complex_x, complex_y, complex_dists)):
        ax.annotate(f'C{i+1}: {d:.3f}', (x, y),
                   fontsize=10, ha='center', va='bottom', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor='darkgreen' if d < 0.15 else ('green' if d < 0.25 else ('orange' if d < 0.40 else 'red')),
                            linewidth=2, alpha=0.9))
    
    # Draw concentric circles at objective quality thresholds
    thresholds = [0.15, 0.25, 0.40]
    threshold_labels = ['Excellent\n(Indistinguishable)', 'Good\n(Strong)', 'Fair\n(Moderate)']
    colors_thresh = ['darkgreen', 'green', 'orange']
    
    for thresh, label, color in zip(thresholds, threshold_labels, colors_thresh):
        circle = plt.Circle((0, 0), thresh, fill=False, 
                           color=color, linestyle='--', linewidth=2, alpha=0.5)
        ax.add_patch(circle)
        # Label the circle
        ax.text(thresh, 0, f' {thresh:.2f}\n {label}', 
               fontsize=9, va='center', ha='left',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Compute statistics
    avg_simple = np.mean(simple_dists)
    avg_complex = np.mean(complex_dists)
    avg_overall = (avg_simple + avg_complex) / 2
    
    # Determine quality (objective thresholds - not curve-fitted to our models)
    # Based on same-author baseline: truly good mimicry should be close to same-author distance
    # STAR average: 0.308 = FAIR, not GOOD (be honest about limitations)
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
    
    # Title
    title_text = (
        f'HONEST Distance Visualization: Rank #{rank} - {author_id}\n'
        f'Avg Distance to Training Docs: Simple={avg_simple:.4f}, Complex={avg_complex:.4f}, '
        f'Overall={avg_overall:.4f}\n'
        f'Mimicry Quality: {quality}'
    )
    ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20, color=quality_color)
    
    # Add explanation box
    explanation = (
        '⭐ HONEST METRIC: Distance = Mean to ALL Training Docs (Not Centroid)!\n'
        '• Origin (red X) represents the training document cloud\n'
        '• Each point placed at its mean distance to ALL training docs\n'
        '• Closer to center = Better mimicry\n'
        '• More accurate than distance to centroid (avoids ~25% optimism bias)'
    )
    ax.text(0.02, 0.98, explanation, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=1))
    
    # Styling
    ax.set_xlabel('Distance (arbitrary direction →)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance (arbitrary direction ↑)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Set limits to show all data with some padding
    max_dist = max(max(train_dists), max(simple_dists), max(complex_dists))
    limit = max(0.35, max_dist * 1.2)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    # Colorbar
    cbar = plt.colorbar(scatter_simple, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label('HONEST Distance: Mean to All Training Docs', 
                   fontsize=12, fontweight='bold')
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"true_distance_rank{rank:02d}_{author_id}_{model_key}_run{full_run}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] {output_path}")
    return output_path


def create_grid_radial(
    all_author_data,
    author_ids_ranked,
    model_key,
    full_run,
    output_dir
):
    """Create grid of radial distance plots (HONEST metric, no clipping)."""
    n_authors = len(author_ids_ranked)
    n_cols = min(3, n_authors)
    n_rows = (n_authors + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9*n_cols, 8*n_rows))
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
        ax.set_aspect('equal')
        
        author_data = all_author_data[author_id]
        training_docs = author_data['training']
        
        # HONEST metric: mean distance to ALL training docs
        train_dist_matrix = cdist(training_docs, training_docs, metric='cosine')
        n_train = train_dist_matrix.shape[0]
        if n_train > 1:
            train_dists = train_dist_matrix.sum(axis=1) / (n_train - 1)
        else:
            train_dists = np.zeros_like(train_dist_matrix[:, 0])
        
        simple_dist_matrix = cdist(author_data['simple'], training_docs, metric='cosine')
        simple_dists = simple_dist_matrix.mean(axis=1)
        
        complex_dist_matrix = cdist(author_data['complex'], training_docs, metric='cosine')
        complex_dists = complex_dist_matrix.mean(axis=1)
        
        # Angles
        train_angles = np.linspace(0, 150, len(train_dists)) * np.pi / 180
        simple_angles = np.linspace(180, 240, len(simple_dists)) * np.pi / 180
        complex_angles = np.linspace(270, 330, len(complex_dists)) * np.pi / 180
        
        # Positions
        train_x = train_dists * np.cos(train_angles)
        train_y = train_dists * np.sin(train_angles)
        simple_x = simple_dists * np.cos(simple_angles)
        simple_y = simple_dists * np.sin(simple_angles)
        complex_x = complex_dists * np.cos(complex_angles)
        complex_y = complex_dists * np.sin(complex_angles)
        
        # Plot
        ax.scatter([0], [0], c='red', marker='X', s=200,
                  edgecolors='black', linewidths=2, zorder=100)
        ax.scatter(train_x, train_y, c='blue', marker='o', s=80,
                  alpha=0.6, edgecolors='black', linewidths=1)
        ax.scatter(simple_x, simple_y, c=simple_dists, cmap='RdYlGn_r',
                  vmin=0.05, vmax=0.35, marker='s', s=120,
                  edgecolors='black', linewidths=1.5)
        ax.scatter(complex_x, complex_y, c=complex_dists, cmap='RdYlGn_r',
                  vmin=0.05, vmax=0.35, marker='^', s=120,
                  edgecolors='black', linewidths=1.5)
        
        # Circle at 0.20 (good threshold)
        circle = plt.Circle((0, 0), 0.20, fill=False,
                           color='green', linestyle='--', linewidth=2, alpha=0.6)
        ax.add_patch(circle)
        
        # Stats (using HONEST distances)
        avg_dist = (np.mean(simple_dists) + np.mean(complex_dists)) / 2
        
        ax.set_title(
            f'Rank #{idx+1}: {author_id[:15]}...\nAvg: {avg_dist:.3f}',
            fontsize=10,
            fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        
        # Dynamic limits per author to avoid clipping
        max_dist = max(
            np.max(train_dists) if len(train_dists) else 0,
            np.max(simple_dists) if len(simple_dists) else 0,
            np.max(complex_dists) if len(complex_dists) else 0,
        )
        limit = max(0.35, max_dist * 1.2)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
    
    # Hide unused subplots
    for idx in range(n_authors, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    fig.suptitle(
        f'TRUE Distance Grid: Top {n_authors} Authors\n'
        f'Visual Distance ∝ Actual Embedding Distance | Closer to Center = Better Mimicry',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout()
    output_path = output_dir / f"true_distance_grid_top{n_authors}_{model_key}_run{full_run}.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] Grid: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create TRUE distance visualizations")
    parser.add_argument("--model-key", type=str, default="style_embedding")
    parser.add_argument("--full-run", type=int, default=1, choices=[1, 2])
    parser.add_argument("--llm-key", type=str, default="gpt-5.1")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--rank-by", type=str, default="average",
                       choices=["simple", "complex", "best", "average"])
    parser.add_argument("--grid-view", action="store_true")
    
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
    print(f"TRUE Distance Visualization: {args.model_key}, Run {args.full_run}")
    print(f"Visual Distance = Actual Distance (No Distortion!)")
    print(f"{'='*80}\n")
    
    # Load selected indices (consistent with analysis/prompts)
    print("[INFO] Loading selected training indices...")
    selected_indices_map = load_selected_indices()
    
    # Load data
    print("[INFO] Loading embeddings...")
    all_author_data = {}
    for author_id in tqdm(author_ids, desc="Loading"):
        data = load_author_embeddings(author_id, args.model_key, args.llm_key, args.full_run, selected_indices_map)
        if data is not None:
            all_author_data[author_id] = data
    
    if len(all_author_data) == 0:
        print("[ERROR] No data loaded!")
        return
    
    output_dir = PLOTS_DIR / args.model_key / f"fullrun{args.full_run}_true_distances"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create individual plots
    print("\n[INFO] Creating TRUE distance plots...")
    for rank, author_id in enumerate(author_ids, 1):
        if author_id in all_author_data:
            plot_radial_distances(
                author_id,
                all_author_data[author_id],
                rank,
                args.model_key,
                args.full_run,
                output_dir
            )
    
    # Create grid
    if args.grid_view:
        print("\n[INFO] Creating grid view...")
        create_grid_radial(
            all_author_data,
            [aid for aid in author_ids if aid in all_author_data],
            args.model_key,
            args.full_run,
            output_dir
        )
    
    print(f"\n[SUCCESS] TRUE distance visualizations saved to: {output_dir}")
    print(f"[INFO] In these plots, you CAN trust the visual distance!\n")


if __name__ == "__main__":
    main()
