#!/usr/bin/env python3
"""
Create visualizations for independent similarity correlations.

This script creates scatter plots showing relationships between independent
topic similarity measures (SBERT, TF-IDF, Jaccard) and mimicry performance,
similar to the author factors visualization.

Usage:
    python src/plot_independent_similarity_correlations.py --model-key luar_mud_orig --llm-key gpt-5.2-pro --full-run 1
    
    # Process all models
    python src/plot_independent_similarity_correlations.py --all-models --full-run 1
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns


def fdr_correction(p_values):
    """
    Apply Benjamini-Hochberg FDR correction for multiple testing.
    Handles NaNs and edge cases safely.
    """
    p_values = np.array(p_values, dtype=float)
    q_values = np.full_like(p_values, np.nan, dtype=float)
    
    valid = np.isfinite(p_values)
    if valid.sum() == 0:
        return q_values
    
    p = p_values[valid]
    n = len(p)
    
    order = np.argsort(p)
    p_sorted = p[order]
    
    q_sorted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        q = min(p_sorted[i] * n / (i + 1), prev)
        q_sorted[i] = q
        prev = q
    
    q = np.empty(n, dtype=float)
    q[order] = q_sorted
    q_values[valid] = q
    
    return q_values


def create_visualizations(model_key: str, llm_key: str, full_run: int, base_path: Path):
    """Create visualization plots for independent similarity correlations."""
    
    print(f"\n{'='*80}")
    print(f"Creating Independent Similarity Visualizations")
    print(f"{'='*80}")
    print(f"Model: {model_key}")
    print(f"LLM: {llm_key}")
    print(f"Full run: {full_run}\n")
    
    # Load the merged dataset
    data_dir = base_path / "data" / "author_factors" / model_key / llm_key
    merged_file = data_dir / f"author_factors_with_independent_sim_fullrun{full_run}.csv"
    
    if not merged_file.exists():
        print(f"[ERROR] Merged data file not found: {merged_file}")
        print("Run: python src/integrate_independent_similarity.py first")
        return
    
    df = pd.read_csv(merged_file)
    print(f"Loaded data for {len(df)} authors")
    
    # Check if average_distance exists, if not compute it
    if 'average_distance' not in df.columns:
        if 'dist_to_training_simple' in df.columns and 'dist_to_training_complex' in df.columns:
            df['average_distance'] = (df['dist_to_training_simple'] + df['dist_to_training_complex']) / 2
            print("✓ Computed average_distance column")
        else:
            print("[ERROR] Missing distance columns")
            return
    
    # Set style
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    
    # Create figure with 6 subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Define the relationships to plot (using average distance)
    # Following the same pattern as author factors: 3 "help" + 3 "matter"
    relationships = [
        ('mean_sbert_sim_avg', 'average_distance',
         'SBERT Semantic Similarity\n(Mean across all pairs)',
         'Does semantic similarity help?'),
        ('mean_tfidf_sim_avg', 'average_distance',
         'TF-IDF Lexical Overlap\n(Mean across all pairs)',
         'Does lexical overlap help?'),
        ('mean_jaccard_sim_avg', 'average_distance',
         'Jaccard Token Overlap\n(Mean across all pairs)',
         'Does token overlap help?'),
        ('max_sbert_sim_avg', 'average_distance',
         'SBERT Semantic Similarity\n(Max across all pairs)',
         'Does best-case semantic match matter?'),
        ('max_tfidf_sim_avg', 'average_distance',
         'TF-IDF Lexical Overlap\n(Max across all pairs)',
         'Does best-case lexical match matter?'),
        ('max_jaccard_sim_avg', 'average_distance',
         'Jaccard Token Overlap\n(Max across all pairs)',
         'Does best-case token match matter?'),
    ]
    
    # Precompute p-values and FDR-corrected q-values
    ps = []
    rs = []
    valid_rel = []
    
    for (x_col, y_col, x_label, subtitle) in relationships:
        if x_col in df.columns and y_col in df.columns:
            valid_mask = df[x_col].notna() & df[y_col].notna()
            x = df.loc[valid_mask, x_col]
            y = df.loc[valid_mask, y_col]
            # Guard against constant arrays
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
        
        # Create scatter plot with regression line
        sns.regplot(x=x, y=y, ax=ax,
                   scatter_kws={'alpha': 0.5, 's': 50},
                   line_kws={'linewidth': 2})
        
        # Determine significance
        if pd.notna(q) and q < 0.001:
            sig = '***'
        elif pd.notna(q) and q < 0.01:
            sig = '**'
        elif pd.notna(q) and q < 0.05:
            sig = '*'
        else:
            sig = 'n.s.'
        
        # Add correlation text box
        textstr = f'Spearman r = {r:.3f}\nq-value (FDR) = {q:.4f}\n({sig})'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', bbox=props)
        
        # Labels and title
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Mimicry Distance\n(lower = better mimicry)', fontsize=12, fontweight='bold')
        ax.set_title(subtitle, fontsize=13, fontweight='bold', pad=10)
        
        # Add interpretation annotation
        if pd.notna(q) and q < 0.05:
            if r < 0:
                interpretation = "✓ Similarity HELPS mimicry"
            else:
                interpretation = "✗ Similarity HURTS mimicry"
            ax.text(0.95, 0.05, interpretation, transform=ax.transAxes,
                   fontsize=11, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'),
                   fontweight='bold')
        
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Independent Topic Similarity vs Mimicry Performance: {llm_key}\n' +
                f'(FDR correction applied over {len(valid_rel)} plotted relationships)',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    plots_dir = data_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    viz_path = plots_dir / f"independent_similarity_visualization_fullrun{full_run}.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved visualization to: {viz_path}")
    
    # Create comparison heatmap
    create_comparison_heatmap(df, data_dir, full_run, llm_key)


def create_comparison_heatmap(df: pd.DataFrame, output_dir: Path, full_run: int, llm_key: str):
    """Create heatmap comparing LUAR vs independent measures."""
    
    # Define similarity measures
    similarity_measures = [
        'mean_train_gen_emb_sim_avg',  # LUAR baseline
        'mean_sbert_sim_avg',
        'mean_tfidf_sim_avg',
        'mean_jaccard_sim_avg',
        'max_sbert_sim_avg',
        'max_tfidf_sim_avg',
        'max_jaccard_sim_avg',
    ]
    
    perf_cols = [
        'dist_to_training_simple',
        'dist_to_training_complex',
        'average_distance'
    ]
    
    # Filter to available columns
    similarity_measures = [c for c in similarity_measures if c in df.columns]
    perf_cols = [c for c in perf_cols if c in df.columns]
    
    if not similarity_measures or not perf_cols:
        print("Warning: Not enough columns for heatmap")
        return
    
    # Compute correlation matrix
    corr_data = []
    for measure in similarity_measures:
        row = []
        for perf in perf_cols:
            valid_mask = df[measure].notna() & df[perf].notna()
            measure_vals = df.loc[valid_mask, measure]
            perf_vals = df.loc[valid_mask, perf]
            
            if (len(measure_vals) >= 3 and
                np.nanstd(measure_vals) > 0 and
                np.nanstd(perf_vals) > 0):
                r, _ = spearmanr(measure_vals, perf_vals)
                row.append(r)
            else:
                row.append(np.nan)
        corr_data.append(row)
    
    corr_matrix = pd.DataFrame(corr_data, index=similarity_measures, columns=perf_cols)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r',
                center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Spearman Correlation'},
                linewidths=1, linecolor='white',
                ax=ax, annot_kws={'size': 11, 'weight': 'bold'})
    
    # Better labels
    ax.set_xlabel('Performance Metrics\n(lower distance = better mimicry)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Similarity Measures', fontsize=13, fontweight='bold')
    ax.set_title(f'LUAR vs Independent Similarity Measures: {llm_key}\n' +
                'GREEN (negative) = Similarity helps mimicry | RED (positive) = Similarity hurts mimicry',
                fontsize=14, fontweight='bold', pad=15)
    
    # Improve tick labels
    y_labels = []
    for label in similarity_measures:
        if 'train_gen_emb_sim' in label:
            y_labels.append('LUAR (baseline)')
        elif 'sbert' in label:
            stat = 'mean' if 'mean' in label else 'max'
            y_labels.append(f'SBERT ({stat})')
        elif 'tfidf' in label:
            stat = 'mean' if 'mean' in label else 'max'
            y_labels.append(f'TF-IDF ({stat})')
        elif 'jaccard' in label:
            stat = 'mean' if 'mean' in label else 'max'
            y_labels.append(f'Jaccard ({stat})')
        else:
            y_labels.append(label)
    
    ax.set_yticklabels(y_labels, rotation=0, fontsize=10)
    ax.set_xticklabels(['Simple Prompt', 'Complex Prompt', 'Average'],
                      rotation=45, ha='right', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = plots_dir / f"independent_similarity_heatmap_fullrun{full_run}.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved heatmap to: {heatmap_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create visualizations for independent similarity correlations"
    )
    parser.add_argument("--model-key", type=str, default="luar_mud_orig",
                       help="Embedding model key")
    parser.add_argument("--llm-key", type=str,
                       help="LLM key (required unless --all-models)")
    parser.add_argument("--all-models", action="store_true",
                       help="Process all LLM models")
    parser.add_argument("--full-run", type=int, default=1,
                       help="Full run number")
    
    args = parser.parse_args()
    
    base_path = Path(__file__).parent.parent
    
    # Determine which LLMs to process
    if args.all_models:
        llm_models = [
            "claude-opus-4-5-20251101",
            "deepseek-reasoner",
            "gemini-3-pro-preview",
            "gpt-5.2-2025-12-11",
            "gpt-5.2-pro",
            "grok-4-1-fast-reasoning"
        ]
    elif args.llm_key:
        llm_models = [args.llm_key]
    else:
        parser.error("Must specify --llm-key or --all-models")
    
    # Process each model
    for llm_key in llm_models:
        create_visualizations(args.model_key, llm_key, args.full_run, base_path)
        print()


if __name__ == "__main__":
    main()
