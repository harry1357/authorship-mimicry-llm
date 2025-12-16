#!/usr/bin/env python3
"""
Extract and summarize authorship verification results from all repeat runs.
Reads the statistics files generated from 10 repeats and computes averages.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

from generation_config import STYLE_MODEL_KEYS
from model_configs import PLOTS_DIR


def parse_stats_file(stats_path: Path) -> dict:
    """Parse a single statistics file and extract key metrics."""
    with open(stats_path, 'r') as f:
        content = f.read()
    
    stats = {}
    
    # Extract SA mean distance
    match = re.search(r'Same-Author.*?Mean distance:\s+([\d.]+)', content, re.DOTALL)
    if match:
        stats['sa_mean_dist'] = float(match.group(1))
    
    # Extract DA mean distance
    match = re.search(r'Different-Author.*?Mean distance:\s+([\d.]+)', content, re.DOTALL)
    if match:
        stats['da_mean_dist'] = float(match.group(1))
    
    # Extract AUC
    match = re.search(r'AUC:\s+([\d.]+)', content)
    if match:
        stats['auc'] = float(match.group(1))
    
    # Extract optimal CosS
    match = re.search(r'Optimal similarity \(CosS\):\s+([\d.]+)', content)
    if match:
        stats['optimal_coss'] = float(match.group(1))
    
    # Extract optimal CosD
    match = re.search(r'Optimal distance\s+\(CosD\):\s+([\d.]+)', content)
    if match:
        stats['optimal_cosd'] = float(match.group(1))
    
    # Extract Cohen's d
    match = re.search(r"Cohen's d:\s+([\d.]+)", content)
    if match:
        stats['cohens_d'] = float(match.group(1))
    
    return stats


def extract_model_summary(model_key: str, use_split: bool = True) -> dict:
    """Extract summary statistics for a single model across all repeats."""
    model_dir = PLOTS_DIR / model_key / "authorship_verification"
    
    if not model_dir.exists():
        print(f"[WARNING] Directory not found: {model_dir}")
        return None
    
    split_label = "_split" if use_split else ""
    
    # Find all repeat stats files
    stats_files = sorted(model_dir.glob(f"*_stats_{model_key}{split_label}_repeat*.txt"))
    
    if not stats_files:
        print(f"[WARNING] No repeat stats files found for {model_key}")
        return None
    
    print(f"[INFO] Found {len(stats_files)} repeat files for {model_key}")
    
    # Parse all stats files
    all_stats = []
    for stats_file in stats_files:
        stats = parse_stats_file(stats_file)
        if stats:
            all_stats.append(stats)
    
    if not all_stats:
        return None
    
    # Compute averages
    summary = {
        'model': model_key,
        'n_repeats': len(all_stats),
        'mean_auc': np.mean([s['auc'] for s in all_stats]),
        'std_auc': np.std([s['auc'] for s in all_stats]),
        'min_auc': np.min([s['auc'] for s in all_stats]),
        'max_auc': np.max([s['auc'] for s in all_stats]),
        'mean_coss': np.mean([s['optimal_coss'] for s in all_stats]),
        'mean_cosd': np.mean([s['optimal_cosd'] for s in all_stats]),
        'sa_mean_dist': np.mean([s['sa_mean_dist'] for s in all_stats]),
        'da_mean_dist': np.mean([s['da_mean_dist'] for s in all_stats]),
        'mean_cohens_d': np.mean([s['cohens_d'] for s in all_stats]),
    }
    
    # Calculate separation
    summary['separation'] = summary['da_mean_dist'] - summary['sa_mean_dist']
    
    # Assign quality
    if summary['mean_auc'] >= 0.95:
        summary['quality'] = 'EXCELLENT'
    elif summary['mean_auc'] >= 0.90:
        summary['quality'] = 'GOOD'
    elif summary['mean_auc'] >= 0.80:
        summary['quality'] = 'FAIR'
    else:
        summary['quality'] = 'POOR'
    
    return summary


def main():
    print("="*80)
    print("EXTRACTING AUTHORSHIP VERIFICATION SUMMARY FROM ALL REPEATS")
    print("="*80)
    print()
    
    results = []
    
    for model_key in STYLE_MODEL_KEYS:
        print(f"Processing {model_key}...")
        summary = extract_model_summary(model_key, use_split=True)
        if summary:
            results.append(summary)
    
    if not results:
        print("[ERROR] No results found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns
    columns = [
        'model', 'n_repeats', 'mean_auc', 'std_auc', 'min_auc', 'max_auc',
        'mean_coss', 'mean_cosd', 'sa_mean_dist', 'da_mean_dist',
        'separation', 'mean_cohens_d', 'quality'
    ]
    df = df[columns]
    
    # Sort by Mean AUC descending
    df = df.sort_values('mean_auc', ascending=False)
    
    # Add ranking
    df.insert(0, 'rank', range(1, len(df) + 1))
    
    # Save to CSV
    output_path = Path('data/authorship_verification_summary_extracted.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format='%.4f')
    
    print()
    print("="*80)
    print("SUMMARY (Extracted from 10 repeats)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    print(f"\n✅ Summary saved to: {output_path}")
    
    # Print key findings
    print("\n🏆 KEY FINDINGS:")
    print(f"1. Best Model: {df.iloc[0]['model']:20s} (AUC = {df.iloc[0]['mean_auc']:.4f} ± {df.iloc[0]['std_auc']:.4f})")
    print(f"2. Runner-up:  {df.iloc[1]['model']:20s} (AUC = {df.iloc[1]['mean_auc']:.4f} ± {df.iloc[1]['std_auc']:.4f})")
    print(f"3. Third:      {df.iloc[2]['model']:20s} (AUC = {df.iloc[2]['mean_auc']:.4f} ± {df.iloc[2]['std_auc']:.4f})")
    
    print("\n📊 Model Consistency (lower std = more consistent):")
    for _, row in df.iterrows():
        print(f"   {row['model']:20s}: σ = {row['std_auc']:.4f}")
    
    print(f"\n🎯 Recommended for Phase 2: {df.iloc[0]['model']}")
    print(f"   Optimal CosD threshold: {df.iloc[0]['mean_cosd']:.4f}")
    print(f"   Separation (DA - SA):   {df.iloc[0]['separation']:.4f}")


if __name__ == "__main__":
    main()
