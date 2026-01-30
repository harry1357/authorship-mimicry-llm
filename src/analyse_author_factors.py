#!/usr/bin/env python3
"""
Analyze why some authors are mimicked better than others.

Computes:
1. Training set tightness (how clustered are the 6 training documents?)
2. Author distinctiveness (how far from nearest neighbors?)
3. Correlations with mimicry performance

Note: Train-gen topic similarity is analyzed separately using independent TF-IDF metrics.


"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, cosine
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict


def fdr_correction(p_values):
    """
    Apply Benjamini-Hochberg FDR correction for multiple testing.
    Handles NaNs and edge cases safely.
    
    Args:
        p_values: Array of p-values
        
    Returns:
        Array of FDR-corrected q-values (NaN preserved where input is NaN)
    """
    p_values = np.array(p_values, dtype=float)
    q_values = np.full_like(p_values, np.nan, dtype=float)
    
    # Only process finite values
    valid = np.isfinite(p_values)
    if valid.sum() == 0:
        return q_values
    
    p = p_values[valid]
    n = len(p)
    
    # Sort and compute q-values
    order = np.argsort(p)
    p_sorted = p[order]
    
    q_sorted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        q = min(p_sorted[i] * n / (i + 1), prev)
        q_sorted[i] = q
        prev = q
    
    # Put q-values back in original order
    q = np.empty(n, dtype=float)
    q[order] = q_sorted
    q_values[valid] = q
    
    return q_values


def load_training_embeddings(model_key: str, base_path: Path) -> Dict[str, np.ndarray]:
    """Load training embeddings for all authors."""
    embeddings_dir = base_path / "data" / "embeddings" / model_key
    
    author_embeddings = {}
    for author_file in sorted(embeddings_dir.glob("*.npz")):
        author_id = author_file.stem
        data = np.load(author_file)
        embeddings = data['embeddings']  # Shape: (6, embedding_dim)
        author_embeddings[author_id] = embeddings
    
    print(f"Loaded training embeddings for {len(author_embeddings)} authors")
    return author_embeddings


def load_generated_embeddings(model_key: str, llm_key: str, prompt_variant: str, 
                              full_run: int, base_path: Path) -> Dict[str, np.ndarray]:
    """Load generated embeddings for all authors."""
    embeddings_dir = (base_path / "data" / "embeddings" / "generated" / 
                     model_key / llm_key / prompt_variant / f"fullrun{full_run}")
    
    author_embeddings = {}
    for author_file in sorted(embeddings_dir.glob("*.npz")):
        author_id = author_file.stem
        data = np.load(author_file)
        embeddings = data['embeddings']  # Shape: (2, embedding_dim)
        author_embeddings[author_id] = embeddings
    
    print(f"Loaded generated embeddings for {len(author_embeddings)} authors")
    return author_embeddings


def compute_training_tightness(training_embeddings: np.ndarray) -> Dict[str, float]:
    """
    Compute how tightly clustered the 6 training documents are.
    
    Returns:
        Dictionary with multiple tightness metrics:
        - mean_pairwise_dist: Average distance between all pairs
        - std_pairwise_dist: Standard deviation of pairwise distances
        - max_pairwise_dist: Maximum distance (diameter)
        - centroid_dispersion: Average distance to centroid
    """
    # L2-normalize embeddings for consistent cosine distance computation
    normed_embeddings = np.array([
        emb / (np.linalg.norm(emb) + 1e-12) for emb in training_embeddings
    ])
    
    # Compute all pairwise distances
    pairwise_dists = pdist(normed_embeddings, metric='cosine')
    
    # Compute centroid and distances to it (L2-normalize for cleaner cosine distance)
    centroid = normed_embeddings.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    centroid_dists = [cosine(emb, centroid) for emb in normed_embeddings]
    
    return {
        'mean_pairwise_dist': np.mean(pairwise_dists),
        'std_pairwise_dist': np.std(pairwise_dists),
        'max_pairwise_dist': np.max(pairwise_dists),
        'centroid_dispersion': np.mean(centroid_dists)
    }


def precompute_author_centroids(all_training_embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Precompute centroids for all authors (more efficient than recomputing per author).
    Uses per-document L2 normalization before averaging, then normalizes centroid.
    This prevents authors with varying embedding norms from having biased centroids.
    """
    centroids = {}
    for author_id, embeddings in all_training_embeddings.items():
        # Normalize each doc embedding first
        normed = np.array([e / (np.linalg.norm(e) + 1e-12) for e in embeddings])
        centroid = normed.mean(axis=0)
        
        # Normalize centroid for cosine distance
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        centroids[author_id] = centroid
    return centroids


def compute_author_distinctiveness(author_id: str, 
                                   author_centroids: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute how distinctive an author is compared to other authors.
    Uses precomputed normalized centroids for efficiency.
    
    Returns:
        Dictionary with distinctiveness metrics:
        - nearest_neighbor_dist: Distance to closest other author
        - mean_neighbor_dist: Average distance to all other authors
        - k_nearest_avg: Average distance to k nearest neighbors (k=5)
    """
    # Guard against edge case: < 2 authors total
    if len(author_centroids) < 2:
        return {
            'nearest_neighbor_dist': np.nan,
            'mean_neighbor_dist': np.nan,
            'k_nearest_avg': np.nan
        }
    
    # Get target author's centroid
    target_centroid = author_centroids[author_id]
    
    # Compute distances to all other authors
    distances = []
    for other_id, other_centroid in author_centroids.items():
        if other_id != author_id:
            dist = cosine(target_centroid, other_centroid)
            distances.append(dist)
    
    distances = np.array(sorted(distances))
    
    return {
        'nearest_neighbor_dist': distances[0] if len(distances) > 0 else np.nan,
        'mean_neighbor_dist': np.mean(distances) if len(distances) > 0 else np.nan,
        'k_nearest_avg': np.mean(distances[:5]) if len(distances) >= 5 else (np.mean(distances) if len(distances) > 0 else np.nan)
    }


def compute_train_gen_emb_similarity(training_embeddings: np.ndarray, generated_embeddings: np.ndarray) -> Dict[str, float]:
    """
    Compute embedding similarity between training and generated texts.
    
    NOTE: This measures similarity in the SAME embedding space used for mimicry.
    This is NOT true "lexicon overlap" - it's train-gen similarity that may
    correlate with mimicry distance by construction.
    
    For true lexicon/topic overlap, use TF-IDF or Jaccard on raw text instead.
    
    Returns:
        Dictionary with embedding similarity metrics:
        - mean_train_gen_emb_sim: Average similarity between training and generated
        - max_train_gen_emb_sim: Maximum similarity
        - min_train_gen_emb_sim: Minimum similarity
    """
    # L2-normalize embeddings for consistent cosine similarity
    def _l2norm(x):
        return x / (np.linalg.norm(x) + 1e-12)
    
    train_norm = np.array([_l2norm(e) for e in training_embeddings])
    gen_norm = np.array([_l2norm(e) for e in generated_embeddings])
    
    # Compute similarities between all training and generated pairs
    similarities = []
    for train_emb in train_norm:
        for gen_emb in gen_norm:
            # Cosine similarity = 1 - cosine distance
            sim = 1 - cosine(train_emb, gen_emb)
            similarities.append(sim)
    
    similarities = np.array(similarities)
    
    return {
        'mean_train_gen_emb_sim': np.mean(similarities),
        'max_train_gen_emb_sim': np.max(similarities),
        'min_train_gen_emb_sim': np.min(similarities)
    }


def load_mimicry_performance(model_key: str, llm_key: str, full_run: int, 
                             base_path: Path) -> pd.DataFrame:
    """Load mimicry performance metrics from analysis results."""
    # Load the simple vs complex analysis
    consistency_file = (base_path / "data" / "consistency" / 
                       f"simple_vs_complex_{model_key}_{llm_key}_fullrun{full_run}.csv")
    
    if consistency_file.exists():
        df = pd.read_csv(consistency_file)
        print(f"Loaded mimicry performance for {len(df)} authors")
        return df
    else:
        raise FileNotFoundError(f"Consistency file not found: {consistency_file}")


def analyze_correlations(df: pd.DataFrame, output_dir: Path):
    """Analyze correlations between author factors and mimicry performance."""
    
    # Hard check: ensure required performance columns exist
    required = ['dist_to_training_simple', 'dist_to_training_complex']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required performance columns: {missing}. "
                      f"Available columns: {list(df.columns)}")
    
    # Define factor columns - only training set properties and distinctiveness
    invariant_factors = [
        'mean_pairwise_dist', 'std_pairwise_dist', 'max_pairwise_dist', 'centroid_dispersion',
        'nearest_neighbor_dist', 'mean_neighbor_dist', 'k_nearest_avg'
    ]
    
    # Performance metrics (lower distance = better mimicry)
    perf_simple = 'dist_to_training_simple'
    perf_complex = 'dist_to_training_complex'
    
    # Compute average distance if not already present
    if 'average_distance' not in df.columns:
        df['average_distance'] = (df[perf_simple] + df[perf_complex]) / 2
    perf_average = 'average_distance'
    
    # Compute correlations with proper matching:
    # - Invariant factors correlate with all 3 performance metrics
    # - Simple embedding factors correlate ONLY with simple performance
    # - Complex embedding factors correlate ONLY with complex performance
    # - Average embedding factors correlate ONLY with average performance
    results = []
    
    # Invariant factors: correlate with all performance metrics
    for factor in invariant_factors:
        for perf in [perf_simple, perf_complex, perf_average]:
            if factor in df.columns and perf in df.columns:
                valid_mask = df[factor].notna() & df[perf].notna()
                if valid_mask.sum() > 2:
                    # Extract values and guard against constant/near-constant arrays
                    x = df.loc[valid_mask, factor].astype(float).values
                    y = df.loc[valid_mask, perf].astype(float).values
                    
                    # Skip if either variable is constant (prevents NaN correlations)
                    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
                        continue
                    
                    pearson_r, pearson_p = pearsonr(x, y)
                    spearman_r, spearman_p = spearmanr(x, y)
                    
                    results.append({
                        'factor': factor,
                        'performance_metric': perf,
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'spearman_r': spearman_r,
                        'spearman_p': spearman_p,
                        'n_samples': valid_mask.sum()
                    })
    
    corr_df = pd.DataFrame(results)
    
    # Apply FDR correction for multiple testing
    if len(corr_df) > 0:
        corr_df['pearson_q'] = fdr_correction(corr_df['pearson_p'].values)
        corr_df['spearman_q'] = fdr_correction(corr_df['spearman_p'].values)
    
    # Save results
    output_file = output_dir / "author_factors_correlations.csv"
    corr_df.to_csv(output_file, index=False)
    print(f"\nSaved correlation results to: {output_file}")
    
    # Print significant correlations (FDR q < 0.05)
    if len(corr_df) > 0:
        sig_corr = corr_df[corr_df['spearman_q'] < 0.05].sort_values('spearman_q')
        if len(sig_corr) > 0:
            print("\n=== SIGNIFICANT CORRELATIONS (FDR q < 0.05) ===")
            print(sig_corr[['factor', 'performance_metric', 'spearman_r', 'spearman_p', 'spearman_q']].to_string(index=False))
    else:
        print("\nNo valid correlations computed.")
    
    return corr_df


def generate_summary_report(
    merged_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    output_dir: Path,
    model_key: str,
    llm_key: str,
    prompt_variant: str,
    full_run: int,
) -> None:
    """Generate a markdown summary report of key findings."""
    summary_path = output_dir / f"summary_{prompt_variant}_fullrun{full_run}.md"
    
    with open(summary_path, 'w') as f:
        f.write(f"# Author Factor Analysis Summary\n\n")
        f.write(f"**Model**: {model_key}  \n")
        f.write(f"**LLM**: {llm_key}  \n")
        f.write(f"**Prompt Variant**: {prompt_variant}  \n")
        f.write(f"**Full Run**: {full_run}  \n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}  \n\n")
        
        f.write(f"---\n\n")
        
        # Dataset overview
        f.write(f"## Dataset Overview\n\n")
        f.write(f"- **Number of authors analyzed**: {len(merged_df)}\n")
        f.write(f"- **Embedding model**: {model_key}\n")
        f.write(f"- **Training documents per author**: 6 (selected most representative)\n")
        if prompt_variant == "avg":
            f.write(f"- **Generated documents per author**: 2 per prompt (4 total across prompts)\n")
            f.write(f"- **Author distinctiveness**: Computed within evaluated author set (n={len(merged_df)})\n\n")
        else:
            f.write(f"- **Generated documents per author**: 2\n")
            f.write(f"- **Author distinctiveness**: Computed within evaluated author set (n={len(merged_df)})\n\n")
        
        # Performance distribution
        if prompt_variant == "avg":
            perf_col = "average_distance"
        else:
            perf_col = f"dist_to_training_{prompt_variant}"
        if perf_col in merged_df.columns:
            f.write(f"## Mimicry Performance Distribution\n\n")
            f.write(f"**Metric**: `{perf_col}` (lower = better mimicry)\n\n")
            f.write(f"- **Mean**: {merged_df[perf_col].mean():.4f}\n")
            f.write(f"- **Std**: {merged_df[perf_col].std():.4f}\n")
            f.write(f"- **Min (best)**: {merged_df[perf_col].min():.4f}\n")
            f.write(f"- **Max (worst)**: {merged_df[perf_col].max():.4f}\n")
            f.write(f"- **Median**: {merged_df[perf_col].median():.4f}\n\n")
            
            # Best and worst authors
            best_authors = merged_df.nsmallest(5, perf_col)
            worst_authors = merged_df.nlargest(5, perf_col)
            
            f.write(f"### Top 5 Best Mimicked Authors\n\n")
            f.write(f"| Rank | Author ID | Distance |\n")
            f.write(f"|------|-----------|----------|\n")
            for i, (_, row) in enumerate(best_authors.iterrows(), 1):
                f.write(f"| {i} | {row['author_id']} | {row[perf_col]:.4f} |\n")
            f.write(f"\n")
            
            f.write(f"### Top 5 Worst Mimicked Authors\n\n")
            f.write(f"| Rank | Author ID | Distance |\n")
            f.write(f"|------|-----------|----------|\n")
            for i, (_, row) in enumerate(worst_authors.iterrows(), 1):
                f.write(f"| {i} | {row['author_id']} | {row[perf_col]:.4f} |\n")
            f.write(f"\n")
        
        # Key findings from correlations
        f.write(f"## Key Findings: What Predicts Mimicry Success?\n\n")
        
        # Filter correlations for this prompt variant
        relevant_corr = corr_df[corr_df['performance_metric'] == perf_col].copy()
        
        if len(relevant_corr) > 0:
            # Sort by absolute Spearman correlation
            relevant_corr['abs_spearman'] = relevant_corr['spearman_r'].abs()
            relevant_corr = relevant_corr.sort_values('abs_spearman', ascending=False)
            
            f.write(f"### Strongest Correlations (Spearman)\n\n")
            f.write(f"*Note: Negative correlation = factor helps mimicry (lower distance)*\n\n")
            f.write(f"| Factor | Spearman r | p-value | q-value (FDR) | n | Interpretation |\n")
            f.write(f"|--------|------------|---------|---------------|---|----------------|\n")
            
            for _, row in relevant_corr.head(10).iterrows():
                factor = row['factor']
                r = row['spearman_r']
                p = row['spearman_p']
                q = row.get('spearman_q', np.nan)
                n = int(row.get('n_samples', 0))
                
                # Significance based on FDR-corrected q-value
                if pd.notna(q) and q < 0.001:
                    sig = "***"
                elif pd.notna(q) and q < 0.01:
                    sig = "**"
                elif pd.notna(q) and q < 0.05:
                    sig = "*"
                else:
                    sig = "n.s."
                
                # Strength
                if abs(r) < 0.1:
                    strength = "negligible"
                elif abs(r) < 0.3:
                    strength = "weak"
                elif abs(r) < 0.5:
                    strength = "moderate"
                else:
                    strength = "strong"
                
                direction = "hurts" if r > 0 else "helps"
                
                f.write(f"| {factor} | {r:.3f} {sig} | {p:.4f} | {q:.4f} | {n} | {strength} {direction} mimicry |\n")
            
            f.write(f"\n**Significance codes** (based on FDR-corrected q-values): *** q<0.001, ** q<0.01, * q<0.05, n.s. = not significant\n\n")
            
            # Interpret key findings
            f.write(f"### Interpretation\n\n")
            
            # Training tightness
            tightness_corr = relevant_corr[relevant_corr['factor'] == 'mean_pairwise_dist']
            if len(tightness_corr) > 0:
                r = tightness_corr.iloc[0]['spearman_r']
                q = tightness_corr.iloc[0].get('spearman_q', np.nan)
                if pd.notna(q) and q < 0.05:
                    if r < 0:
                        f.write(f"✅ **Training Set Tightness HELPS**: Authors with more consistent training documents are easier to mimic (r={r:.3f}, q={q:.4f})\n\n")
                    else:
                        f.write(f"⚠️ **Training Set Tightness HURTS**: Surprisingly, more scattered training documents lead to better mimicry (r={r:.3f}, q={q:.4f})\n\n")
                else:
                    f.write(f"❓ **Training Set Tightness**: No significant relationship (r={r:.3f}, q={q:.4f})\n\n")
            
            # Author distinctiveness
            distinct_corr = relevant_corr[relevant_corr['factor'] == 'nearest_neighbor_dist']
            if len(distinct_corr) > 0:
                r = distinct_corr.iloc[0]['spearman_r']
                q = distinct_corr.iloc[0].get('spearman_q', np.nan)
                if pd.notna(q) and q < 0.05:
                    if r < 0:
                        f.write(f"✅ **Author Distinctiveness HELPS**: More distinctive authors are easier to mimic (r={r:.3f}, q={q:.4f})\n\n")
                    else:
                        f.write(f"⚠️ **Author Distinctiveness HURTS**: More distinctive authors are harder to mimic (r={r:.3f}, q={q:.4f})\n\n")
                else:
                    f.write(f"❓ **Author Distinctiveness**: No significant relationship (r={r:.3f}, q={q:.4f})\n\n")
            
            # Note about topic similarity
            f.write(f"\n**Note**: Train-gen topic similarity is analyzed separately using independent TF-IDF metrics (see topic similarity analysis).\n\n")
        
        f.write(f"\n---\n\n")
        f.write(f"*Generated by `analyse_author_factors.py`*\n")
    
    print(f"✓ Saved summary report to: {summary_path}")


def create_visualizations(df: pd.DataFrame, output_dir: Path, full_run: int):
    """Create improved, interpretable visualizations of relationships between factors and performance."""
    
    # Set style for better aesthetics
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    
    # Create figure with larger, clearer plots
    fig = plt.figure(figsize=(20, 16))
    
    # Define the key relationships to plot (using average distance)
    # Removed train-gen embedding similarity as we use independent similarity measures instead
    relationships = [
        ('mean_pairwise_dist', 'average_distance', 
         'Training Set Tightness\n(Mean Pairwise Distance)', 
         'Does training consistency help?'),
        ('nearest_neighbor_dist', 'average_distance',
         'Author Distinctiveness\n(Distance to Nearest Neighbor)',
         'Do unique authors help?'),
        ('centroid_dispersion', 'average_distance',
         'Training Set Dispersion\n(Distance to Centroid)',
         'Does training spread matter?'),
        ('k_nearest_avg', 'average_distance',
         'Mean Distance to 5 Nearest Authors',
         'Does neighborhood matter?'),
        ('std_pairwise_dist', 'average_distance',
         'Training Set Variability\n(Std of Pairwise Distances)',
         'Does training consistency matter?'),
    ]
    
    # Precompute p-values and FDR-corrected q-values for all valid relationships
    ps = []
    rs = []
    valid_rel = []
    
    for (x_col, y_col, x_label, subtitle) in relationships:
        if x_col in df.columns and y_col in df.columns:
            valid_mask = df[x_col].notna() & df[y_col].notna()
            x = df.loc[valid_mask, x_col]
            y = df.loc[valid_mask, y_col]
            # Guard against constant/near-constant arrays (prevents NaN correlations)
            if len(x) >= 3 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
                r, p = spearmanr(x, y)
                ps.append(p)
                rs.append(r)
                valid_rel.append((x_col, y_col, x_label, subtitle))
    
    # Apply FDR correction
    qs = fdr_correction(np.array(ps)) if ps else []
    
    # Plot each valid relationship
    for idx, ((x_col, y_col, x_label, subtitle), r, p, q) in enumerate(zip(valid_rel, rs, ps, qs), 1):
        ax = plt.subplot(3, 2, idx)
        
        # Remove NaN values
        valid_mask = df[x_col].notna() & df[y_col].notna()
        x = df.loc[valid_mask, x_col]
        y = df.loc[valid_mask, y_col]
        
        # Create scatter plot with regression line (no explicit colors)
        sns.regplot(x=x, y=y, ax=ax, 
                   scatter_kws={'alpha': 0.5, 's': 50},
                   line_kws={'linewidth': 2})
        
        # Determine significance based on FDR-corrected q-value
        if pd.notna(q) and q < 0.001:
            sig = '***'
        elif pd.notna(q) and q < 0.01:
            sig = '**'
        elif pd.notna(q) and q < 0.05:
            sig = '*'
        else:
            sig = 'n.s.'
        
        # Add correlation text box with FDR-corrected q-value
        textstr = f'Spearman r = {r:.3f}\nq-value (FDR) = {q:.4f}\n({sig})'
        
        # Add text box with white background
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top', bbox=props)
        
        # Labels and title
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Mimicry Distance\n(lower = better mimicry)', fontsize=12, fontweight='bold')
        ax.set_title(subtitle, fontsize=13, fontweight='bold', pad=10)
        
        # Add interpretation annotation (no explicit colors) using FDR-corrected q
        if pd.notna(q) and q < 0.05:
            if r < 0:
                interpretation = "✓ Factor HELPS mimicry"
            else:
                interpretation = "✗ Factor HURTS mimicry"
            ax.text(0.95, 0.05, interpretation, transform=ax.transAxes,
                   fontsize=11, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'),
                   fontweight='bold')
        
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Author Factors vs Average Mimicry Performance\n(FDR correction applied over {len(valid_rel)} plotted relationships)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure to unified plots directory
    viz_path = output_dir / f"author_factors_visualization_average_fullrun{full_run}.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved visualizations to: {viz_path}")
    
    # Create a second figure: Correlation heatmap
    create_correlation_heatmap(df, output_dir, full_run)


def create_correlation_heatmap(df: pd.DataFrame, output_dir: Path, full_run: int):
    """Create an improved correlation heatmap with clear interpretation."""
    
    # Define factor and performance columns (only training properties and distinctiveness)
    factor_cols = [
        'mean_pairwise_dist', 
        'std_pairwise_dist', 
        'centroid_dispersion',
        'nearest_neighbor_dist', 
        'mean_neighbor_dist', 
        'k_nearest_avg',
    ]
    
    perf_cols = [
        'dist_to_training_simple',
        'dist_to_training_complex',
        'average_distance'
    ]
    
    # Filter to available columns
    factor_cols = [c for c in factor_cols if c in df.columns]
    perf_cols = [c for c in perf_cols if c in df.columns]
    
    if not factor_cols or not perf_cols:
        print("Warning: Not enough columns for correlation heatmap")
        return
    
    # Compute correlation matrix (only factors vs performance)
    corr_data = []
    for factor in factor_cols:
        row = []
        for perf in perf_cols:
            valid_mask = df[factor].notna() & df[perf].notna()
            factor_vals = df.loc[valid_mask, factor]
            perf_vals = df.loc[valid_mask, perf]
            
            # Guard against constant/near-constant arrays
            if (len(factor_vals) >= 3 and 
                np.nanstd(factor_vals) > 0 and 
                np.nanstd(perf_vals) > 0):
                r, _ = spearmanr(factor_vals, perf_vals)
                row.append(r)
            else:
                row.append(np.nan)
        corr_data.append(row)
    
    corr_matrix = pd.DataFrame(corr_data, index=factor_cols, columns=perf_cols)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap with better color scheme
    # Use RdYlGn_r (reversed) so GREEN = negative correlation = helps mimicry
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Spearman Correlation'},
                linewidths=1, linecolor='white',
                ax=ax, annot_kws={'size': 11, 'weight': 'bold'})
    
    # Better labels
    ax.set_xlabel('Performance Metrics\n(lower distance = better mimicry)', 
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Author Factors', fontsize=13, fontweight='bold')
    ax.set_title('How Author Factors Correlate with Mimicry Performance\n' +
                'GREEN (negative) = Factor helps mimicry | RED (positive) = Factor hurts mimicry',
                fontsize=14, fontweight='bold', pad=15)
    
    # Improve tick labels
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.set_xticklabels(['Simple Prompt', 'Complex Prompt', 'Average'], 
                      rotation=45, ha='right', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure to unified plots directory
    heatmap_path = output_dir / f"correlation_heatmap_fullrun{full_run}.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved correlation heatmap to: {heatmap_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze why some authors are mimicked better than others"
    )
    parser.add_argument("--model-key", type=str, required=True,
                       help="Embedding model key (e.g., luar_mud_orig)")
    parser.add_argument("--llm-key", type=str, required=True,
                       help="LLM key (e.g., deepseek-reasoner, gpt-5.2-pro)")
    parser.add_argument("--full-run", type=int, default=1,
                       help="Full run number")
    
    args = parser.parse_args()
    
    base_path = Path(__file__).parent.parent
    
    print(f"\n=== Analyzing Author Factors for {args.llm_key} ===")
    print(f"Model: {args.model_key}")
    print(f"Full run: {args.full_run}\n")
    
    # Load embeddings (only training - no need for generated embeddings)
    print("Loading training embeddings...")
    training_embs = load_training_embeddings(args.model_key, base_path)
    
    # Load mimicry performance
    print("\nLoading mimicry performance...")
    perf_df = load_mimicry_performance(args.model_key, args.llm_key, args.full_run, base_path)
    
    # Compute author factors
    print("\nComputing author factors...")
    author_factors = []
    
    # Only process authors that appear in performance data
    common_authors = set(training_embs.keys()) & set(perf_df['author_id'].values)
    print(f"Found {len(common_authors)} authors with training embeddings and performance data")
    
    # Precompute centroids once for efficiency (only for common authors)
    print("Precomputing author centroids...")
    common_training_embs = {aid: training_embs[aid] for aid in common_authors}
    author_centroids = precompute_author_centroids(common_training_embs)
    
    for author_id in sorted(common_authors):
        
        # Training tightness
        tightness = compute_training_tightness(training_embs[author_id])
        
        # Author distinctiveness (using precomputed centroids)
        distinctiveness = compute_author_distinctiveness(author_id, author_centroids)
        
        # Combine all metrics (NO train-gen embedding similarity - analyzed separately with TF-IDF)
        factors = {
            'author_id': author_id,
            **tightness,
            **distinctiveness,
        }
        author_factors.append(factors)
    
    # Create DataFrame
    factors_df = pd.DataFrame(author_factors)
    
    # Merge with performance metrics
    merged_df = perf_df.merge(factors_df, on='author_id', how='inner')
    
    print(f"\nComputed factors for {len(merged_df)} authors")
    
    # Create output directory for data
    data_output_dir = base_path / "data" / "author_factors" / args.model_key / args.llm_key
    data_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output directory for plots
    plots_output_dir = base_path / "data" / "plots" / args.model_key / args.llm_key / "author_factors"
    plots_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save merged data (full dataset with all prompt variants)
    output_file = data_output_dir / f"author_factors_fullrun{args.full_run}.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"\nSaved author factors to: {output_file}")
    
    # Analyze correlations (save to data directory)
    print("\nAnalyzing correlations...")
    corr_df = analyze_correlations(merged_df, data_output_dir)
    
    # Generate summary reports for all three variants: simple, complex, and average (save to data directory)
    for prompt_variant in ["simple", "complex", "avg"]:
        print(f"\nGenerating summary report for {prompt_variant} prompt...")
        generate_summary_report(
            merged_df, 
            corr_df, 
            data_output_dir,
            args.model_key,
            args.llm_key,
            prompt_variant,
            args.full_run
        )
    
    # Create visualizations (save to plots directory)
    print("\nCreating visualizations...")
    create_visualizations(merged_df, plots_output_dir, args.full_run)
    
    print("\n=== Analysis Complete ===")
    print(f"\nData saved to: {data_output_dir}")
    print(f"Plots saved to: {plots_output_dir}")


if __name__ == "__main__":
    main()
