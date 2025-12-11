#!/usr/bin/env python3
"""
Task 3: Same-Author vs Different-Author Distance Analysis

Analyzes distance distributions to understand separability:
- H1 (Same Author): Distances between documents by the same author
- H2 (Different Authors): Distances between documents by different authors

Generates:
1. Distribution plots comparing same vs different author distances
2. Statistical tests (t-test, KS test)
3. ROC curve for authorship verification
4. Distance threshold analysis

Usage:
    python src/analyze_same_vs_diff_author.py --model-key style_embedding --full-run 1
    
    # For all models
    python src/analyze_same_vs_diff_author.py --all-models --full-run 1
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform
from scipy import stats
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import pandas as pd

from generation_config import (
    EMBEDDINGS_DIR,
    STYLE_MODEL_KEYS,
    CONSISTENCY_DIR,
)
from model_configs import PLOTS_DIR


def load_author_training_embeddings(author_id: str, model_key: str) -> np.ndarray:
    """
    Load the 6 most consistent training documents for an author.
    
    Returns:
        Array of shape (6, embedding_dim)
    """
    npz_path = EMBEDDINGS_DIR / model_key / f"{author_id}.npz"
    
    if not npz_path.exists():
        return None
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        if 'selected_indices' in data:
            selected_idx = data['selected_indices']
            embeddings = data['embeddings'][selected_idx]
        else:
            embeddings = data['embeddings'][:6]
        
        return embeddings
    except Exception as e:
        print(f"[WARNING] Failed to load {author_id}: {e}")
        return None


def compute_same_author_distances(embeddings: np.ndarray, metric: str = 'cosine') -> np.ndarray:
    """
    Compute pairwise distances between documents by the same author.
    
    Args:
        embeddings: Array of shape (n_docs, embedding_dim)
        metric: Distance metric ('euclidean' or 'cosine')
    
    Returns:
        Array of distances (n_docs choose 2)
    """
    # pdist computes all pairwise distances
    distances = pdist(embeddings, metric=metric)
    return distances


def compute_different_author_distances(
    emb1: np.ndarray,
    emb2: np.ndarray,
    metric: str = 'cosine',
) -> np.ndarray:
    """
    Compute distances between documents from two different authors.
    
    Args:
        emb1: Embeddings from author 1 (n1, embedding_dim)
        emb2: Embeddings from author 2 (n2, embedding_dim)
        metric: Distance metric ('euclidean' or 'cosine')
    
    Returns:
        Flattened array of all pairwise distances (n1 * n2)
    """
    # cdist computes distances between two sets
    distances = cdist(emb1, emb2, metric=metric)
    return distances.flatten()


def analyze_distance_distributions(
    model_key: str,
    full_run: int,
    max_diff_pairs: int = 100000,
    metric: str = 'cosine',
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute same-author and different-author distance distributions.
    
    Args:
        model_key: Style embedding model
        full_run: Experimental run number
        max_diff_pairs: Maximum number of different-author pairs to sample
        metric: Distance metric ('euclidean' or 'cosine')
    
    Returns:
        (same_author_dists, diff_author_dists, author_ids)
    """
    # Get experimental authors
    csv_path = CONSISTENCY_DIR / f"simple_vs_complex_{model_key}_fullrun{full_run}.csv"
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        author_ids = df['author_id'].tolist()
    else:
        gen_dir = EMBEDDINGS_DIR / "generated" / model_key / "gpt-5.1" / "simple" / f"fullrun{full_run}"
        author_ids = [f.stem for f in gen_dir.glob("*.npz")]
    
    print(f"[INFO] Found {len(author_ids)} experimental authors")
    
    # Load all embeddings
    all_embeddings = {}
    for author_id in tqdm(author_ids, desc="Loading embeddings"):
        embs = load_author_training_embeddings(author_id, model_key)
        if embs is not None and len(embs) == 6:
            all_embeddings[author_id] = embs
    
    author_ids = list(all_embeddings.keys())
    n_authors = len(author_ids)
    
    print(f"[INFO] Loaded {n_authors} authors with complete data")
    
    # Compute same-author distances
    print(f"[INFO] Computing same-author distances (metric: {metric})...")
    same_author_dists = []
    
    for author_id in tqdm(author_ids, desc="Same-author pairs"):
        embs = all_embeddings[author_id]
        dists = compute_same_author_distances(embs, metric=metric)
        same_author_dists.extend(dists)
    
    same_author_dists = np.array(same_author_dists)
    
    print(f"[INFO] Computed {len(same_author_dists)} same-author pairs")
    print(f"[INFO] Expected: {n_authors} authors × 15 pairs = {n_authors * 15}")
    
    # Compute different-author distances (sample if too many)
    print("[INFO] Computing different-author distances...")
    
    total_possible_pairs = (n_authors * (n_authors - 1) // 2) * 36  # 157C2 * 6^2
    print(f"[INFO] Total possible different-author pairs: {total_possible_pairs:,}")
    
    if total_possible_pairs > max_diff_pairs:
        print(f"[INFO] Sampling {max_diff_pairs:,} pairs for efficiency")
        
        # Randomly sample author pairs
        n_samples = max_diff_pairs // 36  # How many author pairs to sample
        author_indices = np.random.choice(n_authors, size=(n_samples, 2), replace=True)
        
        diff_author_dists = []
        for idx1, idx2 in tqdm(author_indices, desc="Different-author pairs"):
            if idx1 == idx2:
                continue
            
            aid1 = author_ids[idx1]
            aid2 = author_ids[idx2]
            
            dists = compute_different_author_distances(
                all_embeddings[aid1],
                all_embeddings[aid2],
                metric=metric
            )
            diff_author_dists.extend(dists)
    else:
        # Compute all pairs
        diff_author_dists = []
        
        for i, aid1 in enumerate(tqdm(author_ids, desc="Different-author pairs")):
            for j, aid2 in enumerate(author_ids):
                if i >= j:
                    continue
                
                dists = compute_different_author_distances(
                    all_embeddings[aid1],
                    all_embeddings[aid2],
                    metric=metric
                )
                diff_author_dists.extend(dists)
    
    diff_author_dists = np.array(diff_author_dists)
    
    print(f"[INFO] Computed {len(diff_author_dists):,} different-author pairs")
    
    return same_author_dists, diff_author_dists, author_ids


def plot_distributions(
    same_dists: np.ndarray,
    diff_dists: np.ndarray,
    model_key: str,
    full_run: int,
    output_dir: Path,
    metric: str = 'cosine',
):
    """
    Plot distance distributions and compute statistics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    distance_label = 'Cosine Distance' if metric == 'cosine' else 'Euclidean Distance'
    
    # 1. Overlapping histograms
    ax = axes[0, 0]
    ax.hist(same_dists, bins=100, alpha=0.6, label='Same Author', color='blue', density=True)
    ax.hist(diff_dists, bins=100, alpha=0.6, label='Different Authors', color='red', density=True)
    ax.set_xlabel(distance_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Distance Distributions: Same vs Different Authors', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    same_mean = np.mean(same_dists)
    diff_mean = np.mean(diff_dists)
    ax.axvline(same_mean, color='blue', linestyle='--', linewidth=2, label=f'Same mean: {same_mean:.3f}')
    ax.axvline(diff_mean, color='red', linestyle='--', linewidth=2, label=f'Diff mean: {diff_mean:.3f}')
    
    # 2. Box plots
    ax = axes[0, 1]
    ax.boxplot([same_dists, diff_dists], labels=['Same Author', 'Different Authors'])
    ax.set_ylabel(distance_label, fontsize=12, fontweight='bold')
    ax.set_title('Distance Distribution Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Cumulative distributions
    ax = axes[1, 0]
    ax.hist(same_dists, bins=100, alpha=0.6, label='Same Author', color='blue', cumulative=True, density=True)
    ax.hist(diff_dists, bins=100, alpha=0.6, label='Different Authors', color='red', cumulative=True, density=True)
    ax.set_xlabel(distance_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Distribution Functions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 4. ROC Curve
    ax = axes[1, 1]
    
    # Create labels (0 = same author, 1 = different authors)
    y_true = np.concatenate([
        np.zeros(len(same_dists)),
        np.ones(len(diff_dists))
    ])
    
    # Distances as scores (higher distance = more likely different authors)
    y_scores = np.concatenate([same_dists, diff_dists])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve for Authorship Verification', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Same vs Different Author Distance Analysis ({metric.title()} Distance)\nModel: {model_key}, Run: {full_run}',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"same_vs_diff_distances_{metric}_{model_key}_run{full_run}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] Distribution plot: {output_path}")
    
    return roc_auc


def save_statistics(
    same_dists: np.ndarray,
    diff_dists: np.ndarray,
    roc_auc: float,
    model_key: str,
    full_run: int,
    output_dir: Path,
    metric: str = 'cosine',
):
    """
    Compute and save statistical comparisons.
    """
    # Compute statistics
    same_stats = {
        'mean': np.mean(same_dists),
        'median': np.median(same_dists),
        'std': np.std(same_dists),
        'min': np.min(same_dists),
        'max': np.max(same_dists),
        'q25': np.percentile(same_dists, 25),
        'q75': np.percentile(same_dists, 75),
    }
    
    diff_stats = {
        'mean': np.mean(diff_dists),
        'median': np.median(diff_dists),
        'std': np.std(diff_dists),
        'min': np.min(diff_dists),
        'max': np.max(diff_dists),
        'q25': np.percentile(diff_dists, 25),
        'q75': np.percentile(diff_dists, 75),
    }
    
    # Statistical tests
    t_stat, t_pval = stats.ttest_ind(same_dists, diff_dists)
    ks_stat, ks_pval = stats.ks_2samp(same_dists, diff_dists)
    
    # Cohen's d (effect size)
    pooled_std = np.sqrt((np.std(same_dists)**2 + np.std(diff_dists)**2) / 2)
    cohens_d = (diff_stats['mean'] - same_stats['mean']) / pooled_std
    
    # Overlap coefficient
    overlap = max(0, min(same_stats['max'], diff_stats['max']) - max(same_stats['min'], diff_stats['min'])) / \
              (max(same_stats['max'], diff_stats['max']) - min(same_stats['min'], diff_stats['min']))
    
    # Save to file
    stats_path = output_dir / f"distance_statistics_{metric}_{model_key}_run{full_run}.txt"
    
    with open(stats_path, 'w') as f:
        f.write(f"Same vs Different Author Distance Statistics\n")
        f.write(f"Model: {model_key}, Run: {full_run}, Metric: {metric}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"SAME AUTHOR PAIRS (n={len(same_dists):,}):\n")
        f.write("-" * 70 + "\n")
        for key, val in same_stats.items():
            f.write(f"  {key:10s}: {val:.4f}\n")
        
        f.write(f"\nDIFFERENT AUTHOR PAIRS (n={len(diff_dists):,}):\n")
        f.write("-" * 70 + "\n")
        for key, val in diff_stats.items():
            f.write(f"  {key:10s}: {val:.4f}\n")
        
        f.write(f"\nSTATISTICAL TESTS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  T-test statistic:  {t_stat:.4f}\n")
        f.write(f"  T-test p-value:    {t_pval:.4e}\n")
        f.write(f"  KS test statistic: {ks_stat:.4f}\n")
        f.write(f"  KS test p-value:   {ks_pval:.4e}\n")
        f.write(f"  Cohen's d:         {cohens_d:.4f}\n")
        f.write(f"  ROC AUC:           {roc_auc:.4f}\n")
        f.write(f"  Overlap:           {overlap:.4f}\n")
        
        f.write(f"\nINTERPRETATION:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Mean difference:   {diff_stats['mean'] - same_stats['mean']:.4f}\n")
        f.write(f"  Relative increase: {((diff_stats['mean'] / same_stats['mean']) - 1) * 100:.1f}%\n")
        
        if cohens_d < 0.2:
            effect = "negligible"
        elif cohens_d < 0.5:
            effect = "small"
        elif cohens_d < 0.8:
            effect = "medium"
        else:
            effect = "large"
        
        f.write(f"  Effect size:       {effect}\n")
        
        if roc_auc > 0.9:
            separability = "excellent"
        elif roc_auc > 0.8:
            separability = "good"
        elif roc_auc > 0.7:
            separability = "fair"
        else:
            separability = "poor"
        
        f.write(f"  Separability:      {separability} (AUC = {roc_auc:.3f})\n")
    
    print(f"[SAVED] Statistics: {stats_path}")
    
    # Print summary
    print(f"\n[RESULTS] Summary:")
    print(f"  Same-author mean distance:       {same_stats['mean']:.4f}")
    print(f"  Different-author mean distance:  {diff_stats['mean']:.4f}")
    print(f"  Difference:                      {diff_stats['mean'] - same_stats['mean']:.4f}")
    print(f"  Cohen's d:                       {cohens_d:.4f} ({effect})")
    print(f"  ROC AUC:                         {roc_auc:.4f} ({separability})")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze same-author vs different-author distance distributions"
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
        help="Run analysis for all models",
    )
    parser.add_argument(
        "--full-run",
        type=int,
        default=1,
        choices=[1, 2],
        help="Experimental run number",
    )
    parser.add_argument(
        "--max-diff-pairs",
        type=int,
        default=100000,
        help="Maximum number of different-author pairs to sample (default: 100k)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["euclidean", "cosine"],
        help="Distance metric to use (default: cosine)",
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(42)
    
    models = STYLE_MODEL_KEYS if args.all_models else [args.model_key]
    
    for model_key in models:
        print(f"\n{'='*80}")
        print(f"Analyzing model: {model_key}")
        print(f"{'='*80}\n")
        
        # Compute distributions
        same_dists, diff_dists, author_ids = analyze_distance_distributions(
            model_key, args.full_run, args.max_diff_pairs, args.metric
        )
        
        # Output directory
        output_dir = PLOTS_DIR / model_key / f"fullrun{args.full_run}_distance_analysis"
        
        # Generate plots
        roc_auc = plot_distributions(same_dists, diff_dists, model_key, args.full_run, output_dir, args.metric)
        
        # Save statistics
        save_statistics(same_dists, diff_dists, roc_auc, model_key, args.full_run, output_dir, args.metric)
        
        print(f"\n[SUCCESS] Analysis complete for {model_key}\n")


if __name__ == "__main__":
    main()
