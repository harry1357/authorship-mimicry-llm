#!/usr/bin/env python3
"""
Authorship Verification Experiment (Phase 1: Model Selection)

Following the supervisor's protocol:
1. Randomly select 2 reviews from each author (~2,100 authors)
2. Create 2,100 same-author (SA) pairs
3. Sample 2,100 different-author (DA) pairs randomly
4. For each DA pair, randomly select 1 of 4 possible cross-comparisons
5. Compute cosine distance for all SA and DA pairs
6. Generate ROC curves and statistics
7. Select best model based on AUC

Usage:
    # Run for single model
    python src/authorship_verification_experiment.py --model-key star --seed 42
    
    # Run for all models
    python src/authorship_verification_experiment.py --all-models --seed 42
    
    # Use split-average embeddings
    python src/authorship_verification_experiment.py --model-key star --use-split-embeddings
    
    # Repeat multiple times to check consistency
    python src/authorship_verification_experiment.py --model-key star --n-repeats 5
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve, auc
from itertools import combinations
from tqdm import tqdm

from generation_config import STYLE_MODEL_KEYS
from model_configs import EMBEDDINGS_DIR, PLOTS_DIR


def load_embeddings(
    embeddings_dir: Path
) -> Dict[str, np.ndarray]:
    """
    Load embeddings for all authors from a given directory.
    
    Returns dict mapping author_id -> 2D array of document embeddings.
    """
    author_embeddings: Dict[str, np.ndarray] = {}
    
    npz_files = list(embeddings_dir.glob("*.npz"))
    if not npz_files:
        print(f"[WARNING] No .npz files found in {embeddings_dir}")
    
    for npz_file in tqdm(npz_files, desc="Loading embeddings"):
        author_id = npz_file.stem
        data = np.load(npz_file, allow_pickle=True)
        embeddings = data['embeddings']
        author_embeddings[author_id] = embeddings
    
    return author_embeddings


def select_two_docs_per_author(
    author_embeddings: Dict[str, np.ndarray],
    seed: int = 42
) -> Dict[str, Tuple[int, int]]:
    """
    For each author, randomly select 2 document indices.
    
    This is Step 1 in the protocol and is shared between SA and DA construction.
    
    Returns:
        A dict: author_id -> (idx1, idx2)
    """
    random.seed(seed)
    selected_indices = {}
    
    for author_id, embeddings in author_embeddings.items():
        if len(embeddings) < 2:
            continue  # Skip authors with < 2 docs
        
        # Randomly select 2 indices
        idx1, idx2 = random.sample(range(len(embeddings)), 2)
        selected_indices[author_id] = (idx1, idx2)
    
    return selected_indices


def create_same_author_pairs(
    author_embeddings: Dict[str, np.ndarray],
    selected_indices: Dict[str, Tuple[int, int]]
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Create same-author pairs using pre-selected indices.
    
    Returns list of (emb1, emb2, author_id) tuples.
    """
    sa_pairs = []
    
    for author_id, (idx1, idx2) in selected_indices.items():
        embeddings = author_embeddings[author_id]
        emb1 = embeddings[idx1]
        emb2 = embeddings[idx2]
        
        sa_pairs.append((emb1, emb2, author_id))
    
    return sa_pairs


def create_different_author_pairs(
    author_embeddings: Dict[str, np.ndarray],
    selected_indices: Dict[str, Tuple[int, int]],
    n_pairs: int,
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray, str, str]]:
    """
    Sample different-author pairs using pre-selected indices.
    
    For each author pair, use the SAME two pre-selected docs per author
    and randomly select 1 of the 4 possible cross-comparisons.
    
    Returns list of (emb1, emb2, author1_id, author2_id) tuples.
    """
    random.seed(seed)
    
    # Only consider authors that have 2 selected docs
    author_ids = list(selected_indices.keys())
    
    # All possible different-author pairs
    all_da_pairs = list(combinations(author_ids, 2))
    
    # Randomly sample n_pairs
    if len(all_da_pairs) < n_pairs:
        print(f"[WARNING] Only {len(all_da_pairs)} possible DA pairs, requested {n_pairs}")
        sampled_pairs = all_da_pairs
    else:
        sampled_pairs = random.sample(all_da_pairs, n_pairs)
    
    da_pairs = []
    
    for author1_id, author2_id in sampled_pairs:
        # Get the pre-selected indices
        idx1_1, idx1_2 = selected_indices[author1_id]
        idx2_1, idx2_2 = selected_indices[author2_id]
        
        # Get embeddings
        embs1 = author_embeddings[author1_id]
        embs2 = author_embeddings[author2_id]
        
        # Four possible cross-comparisons using the SAME two docs per author
        candidates = [
            (embs1[idx1_1], embs2[idx2_1]),
            (embs1[idx1_1], embs2[idx2_2]),
            (embs1[idx1_2], embs2[idx2_1]),
            (embs1[idx1_2], embs2[idx2_2]),
        ]
        
        # Randomly select one combination
        emb1, emb2 = random.choice(candidates)
        
        da_pairs.append((emb1, emb2, author1_id, author2_id))
    
    return da_pairs


def compute_cosine_distances(
    pairs: List[Tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """Compute cosine distance for each pair."""
    distances = []
    
    for emb1, emb2, *_ in tqdm(pairs, desc="Computing distances"):
        dist = cosine(emb1, emb2)
        distances.append(dist)
    
    return np.array(distances)


def plot_distributions(
    sa_distances: np.ndarray,
    da_distances: np.ndarray,
    model_key: str,
    output_dir: Path,
    use_split: bool = False,
    repeat_idx: int = None
):
    """Plot SA vs DA distance distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram
    ax1.hist(sa_distances, bins=50, alpha=0.6, label=f'Same Author (n={len(sa_distances)})', color='blue')
    ax1.hist(da_distances, bins=50, alpha=0.6, label=f'Different Authors (n={len(da_distances)})', color='red')
    ax1.axvline(np.mean(sa_distances), color='blue', linestyle='--', linewidth=2, label=f'SA Mean: {np.mean(sa_distances):.3f}')
    ax1.axvline(np.mean(da_distances), color='red', linestyle='--', linewidth=2, label=f'DA Mean: {np.mean(da_distances):.3f}')
    ax1.set_xlabel('Cosine Distance', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    
    title_suffix = f" (Repeat {repeat_idx + 1})" if repeat_idx is not None else ""
    ax1.set_title(f'Distance Distributions: {model_key}{title_suffix}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Box plot
    ax2.boxplot([sa_distances, da_distances], labels=['Same Author', 'Different Authors'])
    ax2.set_ylabel('Cosine Distance', fontsize=12, fontweight='bold')
    ax2.set_title(f'Distribution Comparison: {model_key}{title_suffix}', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    split_label = "_split" if use_split else ""
    repeat_label = f"_repeat{repeat_idx + 1:02d}" if repeat_idx is not None else ""
    plt.tight_layout()
    plt.savefig(output_dir / f"authorship_verification_distributions_{model_key}{split_label}{repeat_label}.png", dpi=200, bbox_inches='tight')
    plt.close()


def plot_roc_curve_single(
    sa_distances: np.ndarray,
    da_distances: np.ndarray,
    model_key: str,
    output_dir: Path,
    use_split: bool = False,
    repeat_idx: int = None
) -> float:
    """
    Plot ROC curve for authorship verification.
    
    Returns AUC score.
    """
    # Create labels: SA = 1 (positive), DA = 0 (negative)
    y_true = np.concatenate([np.ones(len(sa_distances)), np.zeros(len(da_distances))])
    
    # Scores: We want SA pairs to have LOWER distance (so negate for scoring)
    # Or equivalently, use (1 - distance) as similarity
    scores = np.concatenate([1 - sa_distances, 1 - da_distances])
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, color='darkblue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    
    title_suffix = f" (Repeat {repeat_idx + 1})" if repeat_idx is not None else ""
    ax.set_title(f'ROC Curve: {model_key}{title_suffix} (AUC={roc_auc:.4f})', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    
    split_label = "_split" if use_split else ""
    repeat_label = f"_repeat{repeat_idx + 1:02d}" if repeat_idx is not None else ""
    plt.tight_layout()
    plt.savefig(output_dir / f"authorship_verification_roc_{model_key}{split_label}{repeat_label}.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    return roc_auc


def save_statistics(
    sa_distances: np.ndarray,
    da_distances: np.ndarray,
    roc_auc: float,
    model_key: str,
    output_dir: Path,
    use_split: bool = False,
    repeat_idx: int = None
):
    """
    Save statistics to text file, including the optimal CosS / CosD threshold
    derived from the ROC curve (Youden's J statistic).
    """
    split_label = "_split" if use_split else ""
    repeat_label = f"_repeat{repeat_idx + 1:02d}" if repeat_idx is not None else ""
    output_path = output_dir / f"authorship_verification_stats_{model_key}{split_label}{repeat_label}.txt"
    
    # Re-create labels + scores for threshold estimation
    y_true = np.concatenate([
        np.ones(len(sa_distances)),
        np.zeros(len(da_distances))
    ])
    scores = np.concatenate([
        1.0 - sa_distances,  # Convert distance to similarity
        1.0 - da_distances
    ])
    
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr  # Youden's J statistic
    best_idx = np.argmax(j)
    best_sim = thresholds[best_idx]  # CosS threshold (on similarity)
    best_dist = 1.0 - best_sim        # CosD threshold (on distance)
    
    with open(output_path, 'w') as f:
        f.write(f"Authorship Verification Statistics: {model_key}\n")
        f.write(f"{'='*70}\n\n")
        
        f.write(f"Same-Author Pairs (n={len(sa_distances)}):\n")
        f.write(f"  Mean distance:   {np.mean(sa_distances):.4f}\n")
        f.write(f"  Std deviation:   {np.std(sa_distances):.4f}\n")
        f.write(f"  Median:          {np.median(sa_distances):.4f}\n")
        f.write(f"  Min:             {np.min(sa_distances):.4f}\n")
        f.write(f"  Max:             {np.max(sa_distances):.4f}\n\n")
        
        f.write(f"Different-Author Pairs (n={len(da_distances)}):\n")
        f.write(f"  Mean distance:   {np.mean(da_distances):.4f}\n")
        f.write(f"  Std deviation:   {np.std(da_distances):.4f}\n")
        f.write(f"  Median:          {np.median(da_distances):.4f}\n")
        f.write(f"  Min:             {np.min(da_distances):.4f}\n")
        f.write(f"  Max:             {np.max(da_distances):.4f}\n\n")
        
        f.write(f"Separation:\n")
        diff_mean = np.mean(da_distances) - np.mean(sa_distances)
        pooled_sd = np.sqrt((np.std(sa_distances)**2 + np.std(da_distances)**2) / 2)
        cohens_d = diff_mean / pooled_sd if pooled_sd > 0 else float('inf')
        f.write(f"  Difference (DA - SA): {diff_mean:.4f}\n")
        f.write(f"  Cohen's d:            {cohens_d:.4f}\n\n")
        
        f.write(f"ROC Analysis:\n")
        f.write(f"  AUC:                        {roc_auc:.4f}\n")
        f.write(f"  Optimal similarity (CosS):  {best_sim:.4f}\n")
        f.write(f"  Optimal distance  (CosD):   {best_dist:.4f}\n")
        
        # Quality assessment
        if roc_auc >= 0.95:
            quality = "EXCELLENT"
        elif roc_auc >= 0.90:
            quality = "GOOD"
        elif roc_auc >= 0.80:
            quality = "FAIR"
        else:
            quality = "POOR"
        f.write(f"  Quality:                    {quality}\n")
    
    print(f"[SAVED] Statistics: {output_path}")
    print(f"[INFO] Optimal CosS: {best_sim:.4f}, Optimal CosD: {best_dist:.4f}")


def plot_combined_roc_curves(
    all_roc_data: List[Tuple[np.ndarray, np.ndarray, float]],
    model_key: str,
    output_dir: Path,
    use_split: bool = False
):
    """Plot all ROC curves from multiple repeats overlaid on one plot."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(all_roc_data)))
    
    for idx, (fpr, tpr, roc_auc) in enumerate(all_roc_data):
        ax.plot(fpr, tpr, color=colors[idx], lw=1.5, alpha=0.7, 
                label=f'Repeat {idx + 1} (AUC = {roc_auc:.4f})')
    
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(f'Combined ROC Curves: {model_key} ({len(all_roc_data)} repeats)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    
    split_label = "_split" if use_split else ""
    plt.tight_layout()
    plt.savefig(output_dir / f"authorship_verification_roc_{model_key}{split_label}_combined.png", 
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Combined ROC plot with {len(all_roc_data)} repeats")


def plot_combined_distributions(
    all_sa_distances: List[np.ndarray],
    all_da_distances: List[np.ndarray],
    model_key: str,
    output_dir: Path,
    use_split: bool = False
):
    """Plot combined histograms showing distribution across all repeats."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram showing all repeats
    colors_sa = plt.cm.Blues(np.linspace(0.4, 0.9, len(all_sa_distances)))
    colors_da = plt.cm.Reds(np.linspace(0.4, 0.9, len(all_da_distances)))
    
    for idx, (sa_dist, da_dist) in enumerate(zip(all_sa_distances, all_da_distances)):
        alpha_val = 0.3 if len(all_sa_distances) > 1 else 0.6
        ax1.hist(sa_dist, bins=50, alpha=alpha_val, color=colors_sa[idx], 
                 label=f'SA Repeat {idx + 1}' if len(all_sa_distances) <= 3 else None)
        ax1.hist(da_dist, bins=50, alpha=alpha_val, color=colors_da[idx],
                 label=f'DA Repeat {idx + 1}' if len(all_da_distances) <= 3 else None)
    
    # Overall means
    all_sa_flat = np.concatenate(all_sa_distances)
    all_da_flat = np.concatenate(all_da_distances)
    ax1.axvline(np.mean(all_sa_flat), color='blue', linestyle='--', linewidth=3,
                label=f'SA Mean (all): {np.mean(all_sa_flat):.3f}')
    ax1.axvline(np.mean(all_da_flat), color='red', linestyle='--', linewidth=3,
                label=f'DA Mean (all): {np.mean(all_da_flat):.3f}')
    
    ax1.set_xlabel('Cosine Distance', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(f'Distance Distributions: {model_key} ({len(all_sa_distances)} repeats)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    # Box plot showing variation across repeats
    sa_means = [np.mean(sa) for sa in all_sa_distances]
    da_means = [np.mean(da) for da in all_da_distances]
    
    ax2.boxplot([sa_means, da_means], labels=['Same Author', 'Different Authors'])
    ax2.scatter([1] * len(sa_means), sa_means, alpha=0.5, color='blue', s=50, label='SA means')
    ax2.scatter([2] * len(da_means), da_means, alpha=0.5, color='red', s=50, label='DA means')
    ax2.set_ylabel('Mean Cosine Distance', fontsize=12, fontweight='bold')
    ax2.set_title(f'Variation Across Repeats: {model_key}', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, axis='y')
    
    split_label = "_split" if use_split else ""
    plt.tight_layout()
    plt.savefig(output_dir / f"authorship_verification_distributions_{model_key}{split_label}_combined.png",
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Combined distribution plot with {len(all_sa_distances)} repeats")


def run_experiment(
    model_key: str,
    use_split: bool = False,
    seed: int = 42,
    n_repeats: int = 1
):
    """Run authorship verification experiment for a single model."""
    print(f"\n{'='*80}")
    print(f"Authorship Verification Experiment: {model_key}")
    print(f"Split-average embeddings: {use_split}")
    print(f"Seed: {seed}")
    print(f"Repeats: {n_repeats}")
    print(f"{'='*80}\n")
    
    # Load embeddings
    if use_split:
        embeddings_dir = EMBEDDINGS_DIR / "split_average" / model_key
    else:
        embeddings_dir = EMBEDDINGS_DIR / model_key
    
    if not embeddings_dir.exists():
        print(f"[ERROR] Embeddings not found: {embeddings_dir}")
        return
    
    print(f"[INFO] Loading embeddings from: {embeddings_dir}")
    author_embeddings = load_embeddings(embeddings_dir)
    
    # Filter to authors with at least 2 docs
    author_embeddings = {k: v for k, v in author_embeddings.items() if len(v) >= 2}
    print(f"[INFO] Loaded {len(author_embeddings)} authors with >= 2 docs")
    
    # Output directory
    output_dir = PLOTS_DIR / model_key / "authorship_verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiment (potentially multiple times)
    all_aucs = []
    all_roc_data = []  # Store (fpr, tpr, auc) for combined plot
    all_sa_distances = []
    all_da_distances = []
    
    for repeat_idx in range(n_repeats):
        current_seed = seed + repeat_idx
        print(f"\n[INFO] Running experiment {repeat_idx + 1}/{n_repeats} (seed={current_seed})")
        
        # Step 1: Select 2 docs per author (shared for SA + DA)
        print("[INFO] Selecting 2 documents per author...")
        selected_indices = select_two_docs_per_author(author_embeddings, seed=current_seed)
        print(f"[INFO] Selected {len(selected_indices)} authors with 2 docs")
        
        # Create SA pairs
        print("[INFO] Creating same-author pairs...")
        sa_pairs = create_same_author_pairs(author_embeddings, selected_indices)
        print(f"[INFO] Created {len(sa_pairs)} SA pairs")
        
        # Create DA pairs (same count as SA, using same selected docs)
        print("[INFO] Creating different-author pairs...")
        da_pairs = create_different_author_pairs(author_embeddings, selected_indices, len(sa_pairs), seed=current_seed)
        print(f"[INFO] Created {len(da_pairs)} DA pairs")
        
        # Compute distances
        sa_distances = compute_cosine_distances(sa_pairs)
        da_distances = compute_cosine_distances(da_pairs)
        
        # Store for combined plots
        all_sa_distances.append(sa_distances)
        all_da_distances.append(da_distances)
        
        # Plot distributions (individual)
        plot_distributions(sa_distances, da_distances, model_key, output_dir, use_split, repeat_idx)
        
        # Compute ROC data
        y_true = np.concatenate([np.ones(len(sa_distances)), np.zeros(len(da_distances))])
        scores = np.concatenate([1 - sa_distances, 1 - da_distances])
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        all_roc_data.append((fpr, tpr, roc_auc))
        
        # Plot ROC curve (individual)
        _ = plot_roc_curve_single(sa_distances, da_distances, model_key, output_dir, use_split, repeat_idx)
        all_aucs.append(roc_auc)
        
        # Save statistics (individual)
        save_statistics(sa_distances, da_distances, roc_auc, model_key, output_dir, use_split, repeat_idx)
        
        print(f"\n[RESULTS] Repeat {repeat_idx + 1}:")
        print(f"  SA mean distance: {np.mean(sa_distances):.4f}")
        print(f"  DA mean distance: {np.mean(da_distances):.4f}")
        print(f"  ROC AUC:          {roc_auc:.4f}")
    
    # Create combined plots if multiple repeats
    if n_repeats > 1:
        print(f"\n[INFO] Creating combined plots across {n_repeats} repeats...")
        plot_combined_roc_curves(all_roc_data, model_key, output_dir, use_split)
        plot_combined_distributions(all_sa_distances, all_da_distances, model_key, output_dir, use_split)
        
        print(f"\n[OVERALL RESULTS] Across {n_repeats} repeats:")
        print(f"  Mean AUC: {np.mean(all_aucs):.4f} ± {np.std(all_aucs):.4f}")
        print(f"  Min AUC:  {np.min(all_aucs):.4f}")
        print(f"  Max AUC:  {np.max(all_aucs):.4f}")
    
    print(f"\n[SUCCESS] Experiment complete for {model_key}")


def main():
    parser = argparse.ArgumentParser(
        description="Authorship Verification Experiment for Model Selection"
    )
    parser.add_argument(
        "--model-key",
        type=str,
        choices=STYLE_MODEL_KEYS,
        help="Model to evaluate"
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run for all models"
    )
    parser.add_argument(
        "--use-split-embeddings",
        action="store_true",
        help="Use split-average embeddings"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1,
        help="Number of times to repeat experiment (for consistency check)"
    )
    
    args = parser.parse_args()
    
    if args.all_models:
        models = STYLE_MODEL_KEYS
    elif args.model_key:
        models = [args.model_key]
    else:
        parser.error("Must specify either --model-key or --all-models")
    
    for model_key in models:
        run_experiment(
            model_key=model_key,
            use_split=args.use_split_embeddings,
            seed=args.seed,
            n_repeats=args.n_repeats
        )


if __name__ == "__main__":
    main()
