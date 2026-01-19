#!/usr/bin/env python3
"""
Integrate independent similarity measures into author factor analysis.

This script:
1. Merges independent similarity CSVs with existing author_factors CSVs
2. Re-runs correlation analysis including the new independent measures
3. Compares LUAR embedding similarity vs independent measures (TF-IDF, SBERT, Jaccard)

This allows us to determine if the strong LUAR correlation (r ≈ -0.99) is:
  - Topical: Independent measures also correlate strongly
  - Authorship-space artifact: Only LUAR correlates, independent measures don't

Usage:
    python src/integrate_independent_similarity.py --model-key luar_mud_orig --llm-key gpt-5.2-pro --full-run 1
    
    # Process all models
    python src/integrate_independent_similarity.py --all-models --full-run 1
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from typing import List


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


def integrate_and_analyze(model_key: str, llm_key: str, full_run: int, base_path: Path):
    """
    Integrate independent similarity measures and analyze correlations.
    """
    print(f"\n{'='*80}")
    print(f"Integrating Independent Similarity Measures")
    print(f"{'='*80}")
    print(f"Model: {model_key}")
    print(f"LLM: {llm_key}")
    print(f"Full run: {full_run}\n")
    
    # Load existing author factors
    factors_file = (base_path / "data" / "author_factors" / model_key / llm_key / 
                   f"author_factors_fullrun{full_run}.csv")
    
    if not factors_file.exists():
        print(f"[ERROR] Author factors file not found: {factors_file}")
        print("Run: python src/analyse_author_factors.py first")
        return None
    
    factors_df = pd.read_csv(factors_file)
    print(f"Loaded author factors: {len(factors_df)} authors")
    
    # Load independent similarity measures
    indep_file = (base_path / "data" / "independent_similarity" / model_key / llm_key / 
                 f"independent_similarity_fullrun{full_run}.csv")
    
    if not indep_file.exists():
        print(f"[ERROR] Independent similarity file not found: {indep_file}")
        print("Run: python src/compute_independent_similarity.py first")
        return None
    
    indep_df = pd.read_csv(indep_file)
    print(f"Loaded independent similarities: {len(indep_df)} authors")
    
    # Merge datasets
    merged_df = factors_df.merge(indep_df, on='author_id', how='inner')
    print(f"Merged dataset: {len(merged_df)} authors\n")
    
    # Check required performance columns
    required = ['dist_to_training_simple', 'dist_to_training_complex']
    missing = [c for c in required if c not in merged_df.columns]
    if missing:
        print(f"[ERROR] Missing performance columns: {missing}")
        return None
    
    # Compute average distance if not present
    if 'average_distance' not in merged_df.columns:
        merged_df['average_distance'] = (merged_df['dist_to_training_simple'] + 
                                         merged_df['dist_to_training_complex']) / 2
        print("✓ Computed average_distance column")
    
    # Define factors to correlate: independent measures + LUAR baseline
    independent_factors = {
        'simple': [
            # LUAR baseline (already in author_factors)
            'mean_train_gen_emb_sim_simple',
            # Independent measures
            'mean_sbert_sim_simple', 'max_sbert_sim_simple', 'min_sbert_sim_simple',
            'mean_tfidf_sim_simple', 'max_tfidf_sim_simple', 'min_tfidf_sim_simple',
            'mean_jaccard_sim_simple', 'max_jaccard_sim_simple', 'min_jaccard_sim_simple'
        ],
        'complex': [
            # LUAR baseline
            'mean_train_gen_emb_sim_complex',
            # Independent measures
            'mean_sbert_sim_complex', 'max_sbert_sim_complex', 'min_sbert_sim_complex',
            'mean_tfidf_sim_complex', 'max_tfidf_sim_complex', 'min_tfidf_sim_complex',
            'mean_jaccard_sim_complex', 'max_jaccard_sim_complex', 'min_jaccard_sim_complex'
        ],
        'avg': [
            # LUAR baseline
            'mean_train_gen_emb_sim_avg',
            # Independent measures
            'mean_sbert_sim_avg', 'max_sbert_sim_avg', 'min_sbert_sim_avg',
            'mean_tfidf_sim_avg', 'max_tfidf_sim_avg', 'min_tfidf_sim_avg',
            'mean_jaccard_sim_avg', 'max_jaccard_sim_avg', 'min_jaccard_sim_avg'
        ]
    }
    
    # Performance metrics
    perf_metrics = {
        'simple': 'dist_to_training_simple',
        'complex': 'dist_to_training_complex',
        'avg': 'average_distance'
    }
    
    # Compute correlations for independent measures
    results = []
    
    for variant in ['simple', 'complex', 'avg']:
        perf_col = perf_metrics[variant]
        factors = independent_factors[variant]
        
        for factor in factors:
            if factor not in merged_df.columns:
                continue
            
            valid_mask = merged_df[factor].notna() & merged_df[perf_col].notna()
            if valid_mask.sum() <= 2:
                continue
            
            x = merged_df.loc[valid_mask, factor].astype(float).values
            y = merged_df.loc[valid_mask, perf_col].astype(float).values
            
            # Skip constant arrays
            if np.nanstd(x) == 0 or np.nanstd(y) == 0:
                continue
            
            pearson_r, pearson_p = pearsonr(x, y)
            spearman_r, spearman_p = spearmanr(x, y)
            
            results.append({
                'factor': factor,
                'performance_metric': perf_col,
                'variant': variant,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'n_samples': valid_mask.sum()
            })
    
    if not results:
        print("[ERROR] No valid correlations computed")
        return None
    
    corr_df = pd.DataFrame(results)
    
    # Apply FDR correction PER VARIANT (not globally)
    # This treats simple/complex/avg as separate families of tests
    def apply_fdr_per_variant(group):
        group['pearson_q'] = fdr_correction(group['pearson_p'].values)
        group['spearman_q'] = fdr_correction(group['spearman_p'].values)
        return group
    
    corr_df = corr_df.groupby('variant', group_keys=False).apply(apply_fdr_per_variant)
    print("✓ Applied FDR correction per variant (simple/complex/avg treated as separate families)")
    
    # Save results
    output_dir = base_path / "data" / "author_factors" / model_key / llm_key
    output_file = output_dir / f"independent_similarity_correlations_fullrun{full_run}.csv"
    corr_df.to_csv(output_file, index=False)
    print(f"✓ Saved correlation results to: {output_file}")
    
    # Also save the merged dataset
    merged_file = output_dir / f"author_factors_with_independent_sim_fullrun{full_run}.csv"
    merged_df.to_csv(merged_file, index=False)
    print(f"✓ Saved merged dataset to: {merged_file}\n")
    
    # Print and save comparison summary
    summary_file = output_dir / f"independent_similarity_summary_fullrun{full_run}.txt"
    print_comparison_summary(corr_df, model_key, llm_key, save_to=summary_file)
    
    return corr_df


def print_comparison_summary(corr_df: pd.DataFrame, model_key: str, llm_key: str, save_to: Path = None):
    """
    Print a comparison of LUAR vs independent measure correlations.
    
    INTERPRETATION NOTE:
    - Performance metric is DISTANCE (lower = better mimicry)
    - Similarity measures are HIGHER = more similar
    - Therefore: NEGATIVE correlation means similarity HELPS mimicry
    - POSITIVE correlation means similarity HURTS mimicry (opposite direction)
    
    Args:
        corr_df: DataFrame with correlation results
        model_key: Embedding model key
        llm_key: LLM key
        save_to: Optional path to save summary text file
    """
    # Prepare output (both for print and save)
    output_lines = []
    
    def add_line(line=""):
        """Add line to output buffer and print"""
        output_lines.append(line)
        print(line)
    
    add_line("=" * 80)
    add_line(f"CORRELATION COMPARISON: LUAR vs Independent Measures")
    add_line("=" * 80)
    add_line(f"\nModel: {model_key}")
    add_line(f"LLM: {llm_key}\n")
    add_line("📊 INTERPRETATION GUIDE:")
    add_line("  • Performance metric = DISTANCE (lower = better mimicry)")
    add_line("  • Similarity measures = HIGHER = more similar")
    add_line("  • NEGATIVE r: similarity ↑ → distance ↓ → HELPS mimicry ✓")
    add_line("  • POSITIVE r: similarity ↑ → distance ↑ → HURTS mimicry ✗")
    add_line("  • |r| > 0.5 + q < 0.05 = strong evidence")
    add_line("=" * 80 + "\n")
    
    for variant in ['simple', 'complex', 'avg']:
        variant_data = corr_df[corr_df['variant'] == variant].copy()
        if len(variant_data) == 0:
            continue
        
        add_line(f"--- {variant.upper()} PROMPT ---\n")
        
        # Separate by measure type
        luar = variant_data[variant_data['factor'].str.contains('train_gen_emb_sim')]
        sbert = variant_data[variant_data['factor'].str.contains('sbert') & ~variant_data['factor'].str.contains('train_gen')]
        tfidf = variant_data[variant_data['factor'].str.contains('tfidf')]
        jaccard = variant_data[variant_data['factor'].str.contains('jaccard')]
        
        # LUAR baseline (should show r ≈ -0.99)
        add_line("LUAR (Authorship Embedding - Baseline):")
        if len(luar) > 0:
            mean_luar = luar[luar['factor'].str.contains('mean')]
            if len(mean_luar) > 0:
                row = mean_luar.iloc[0]
                add_line(f"  {row['factor']}")
                add_line(f"    Spearman r = {row['spearman_r']:.3f}, q = {row['spearman_q']:.4f}")
                direction = "✓ HELPS" if row['spearman_r'] < 0 else "✗ HURTS"
                sig = "SIG" if row['spearman_q'] < 0.05 else "NOT SIG"
                add_line(f"    Direction: {direction}, Significance: {sig}")
        else:
            add_line("  [Not available - LUAR similarity not in author_factors?]")
        
        # Helper function to print measure summary
        def print_measure_summary(measure_name, measure_data, actual_name=""):
            add_line(f"\n{measure_name}:")
            if len(measure_data) == 0:
                add_line("  [No data]")
                return
            
            # Show mean/max/min separately
            for stat in ['mean', 'max', 'min']:
                subset = measure_data[measure_data['factor'].str.contains(stat)]
                if len(subset) > 0:
                    row = subset.iloc[0]
                    r = row['spearman_r']
                    q = row['spearman_q']
                    direction = "✓ HELPS" if r < 0 else "✗ HURTS"
                    sig = "SIG" if q < 0.05 else "NOT SIG"
                    add_line(f"  {stat:4s}: r = {r:6.3f}, q = {q:.4f}  {direction:8s} ({sig})")
            
            # Highlight strongest helping (most negative significant)
            sig_subset = measure_data[measure_data['spearman_q'] < 0.05]
            if len(sig_subset) > 0:
                helping = sig_subset[sig_subset['spearman_r'] < 0]
                if len(helping) > 0:
                    strongest_help = helping.loc[helping['spearman_r'].idxmin()]
                    add_line(f"  → Strongest HELPING: {strongest_help['factor'].split('_')[-2]} "
                          f"(r = {strongest_help['spearman_r']:.3f})")
        
        print_measure_summary("SBERT (Generic Semantic Similarity)", sbert)
        print_measure_summary("TF-IDF (Lexical/Topic Overlap)", tfidf)
        print_measure_summary("Jaccard (Token-Set Overlap, unigrams)", jaccard)
        
        add_line("\n")
    
    # Overall summary
    add_line("=" * 80)
    add_line("KEY FINDINGS (focus on MEAN similarities for interpretability):\n")
    
    # Get MEAN correlations for each measure type
    mean_data = corr_df[corr_df['factor'].str.contains('mean')]
    
    for measure in ['train_gen_emb_sim', 'sbert', 'tfidf', 'jaccard']:
        if measure == 'train_gen_emb_sim':
            label = "LUAR   "
        else:
            label = measure.upper().ljust(7)
        
        measure_data = mean_data[mean_data['factor'].str.contains(measure)]
        if len(measure_data) > 0:
            # Get strongest effect across all variants
            strongest = measure_data.loc[measure_data['spearman_r'].abs().idxmax()]
            r = strongest['spearman_r']
            q = strongest['spearman_q']
            
            if r < 0:
                direction_str = "HELPS (negative)" if q < 0.05 else "helps? (negative, not sig)"
            else:
                direction_str = "HURTS (positive)" if q < 0.05 else "no effect (not sig)"
            
            sig = "✓ SIG" if q < 0.05 else "✗ NOT SIG"
            add_line(f"{label}: r = {r:6.3f}, q = {q:.4f}  {direction_str:30s} {sig}")
    
    add_line(f"\n{'='*80}")
    add_line("INTERPRETATION:\n")
    add_line("If independent measures (TF-IDF, SBERT, Jaccard) show:")
    add_line("  • Strong NEGATIVE correlations (r < -0.5, q < 0.05):")
    add_line("    → Topical similarity HELPS mimicry (confirmed)")
    add_line("    → Your supervisor's basketball-vs-beauty hypothesis is supported\n")
    add_line("  • Weak/no significant correlations (q > 0.05):")
    add_line("    → LUAR correlation is an authorship-space artifact")
    add_line("    → Mimicry success is about stylometric match, not topic\n")
    add_line("  • Strong POSITIVE correlations (r > +0.3, q < 0.05):")
    add_line("    → MORE topic similarity → WORSE mimicry (counterintuitive!)")
    add_line("    → Needs investigation (may be artifact of min/max variants)")
    add_line("=" * 80 + "\n")
    
    # Save to file if path provided
    if save_to:
        with open(save_to, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print(f"✓ Saved summary to: {save_to}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Integrate independent similarity measures and analyze correlations"
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
        result = integrate_and_analyze(args.model_key, llm_key, args.full_run, base_path)
        if result is None:
            print(f"[WARNING] Skipping {llm_key} due to missing data\n")


if __name__ == "__main__":
    main()
