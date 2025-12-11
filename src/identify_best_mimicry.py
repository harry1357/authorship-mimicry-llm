#!/usr/bin/env python3
"""
Identify Best Mimicry Examples for Presentation

This script analyzes the simple vs complex CSV results and identifies the top
authors for each model based on how well the generated texts matched their style.

Usage:
    python src/identify_best_mimicry.py --model-key style_embedding --full-run 1 --top-n 10
    
    # Or analyze all models at once
    python src/identify_best_mimicry.py --all-models --full-run 1 --top-n 10
"""

import argparse
import csv
from pathlib import Path
from typing import List, Dict

import pandas as pd

from generation_config import CONSISTENCY_DIR, STYLE_MODEL_KEYS
from model_configs import PLOTS_DIR


def load_analysis_results(model_key: str, full_run: int) -> pd.DataFrame:
    """Load the CSV analysis results for a model."""
    csv_path = CONSISTENCY_DIR / f"simple_vs_complex_{model_key}_fullrun{full_run}.csv"
    
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        print(f"Run: python src/analyse_simple_vs_complex.py --model-key {model_key} --full-run {full_run}")
        return None
    
    df = pd.read_csv(csv_path)
    return df


def identify_best_authors(
    df: pd.DataFrame, 
    top_n: int = 10,
    criteria: str = "simple"
) -> List[Dict]:
    """
    Identify top authors based on mimicry quality.
    
    Args:
        df: DataFrame with analysis results
        top_n: Number of top authors to return
        criteria: "simple", "complex", or "both" (best overall)
        
    Returns:
        List of dicts with author info sorted by quality
    """
    if criteria == "simple":
        # Sort by dist_real_centroid_simple (lower = better)
        df_sorted = df.sort_values('dist_real_centroid_simple')
        metric = 'dist_real_centroid_simple'
    elif criteria == "complex":
        # Sort by dist_real_centroid_complex (lower = better)
        df_sorted = df.sort_values('dist_real_centroid_complex')
        metric = 'dist_real_centroid_complex'
    else:  # "both"
        # Sort by average of simple and complex (lower = better)
        df['avg_distance'] = (df['dist_real_centroid_simple'] + df['dist_real_centroid_complex']) / 2
        df_sorted = df.sort_values('avg_distance')
        metric = 'avg_distance'
    
    results = []
    for idx, row in df_sorted.head(top_n).iterrows():
        results.append({
            'author_id': row['author_id'],
            'dist_simple': row['dist_real_centroid_simple'],
            'dist_complex': row['dist_real_centroid_complex'],
            'intra_real': row['intra_real'],
            'intra_simple': row['intra_simple'],
            'intra_complex': row['intra_complex'],
            'simple_better': row['dist_real_centroid_simple'] < row['dist_real_centroid_complex'],
        })
    
    return results


def print_results(model_key: str, full_run: int, results: List[Dict], top_n: int):
    """Print formatted results."""
    print(f"\n{'='*80}")
    print(f"TOP {top_n} AUTHORS FOR MODEL: {model_key} (Full Run {full_run})")
    print(f"{'='*80}\n")
    
    print(f"{'Rank':<6} {'Author ID':<18} {'Simple':<10} {'Complex':<10} {'Better':<10} {'Intra Real':<12}")
    print(f"{'-'*80}")
    
    for i, result in enumerate(results, 1):
        better = "Simple" if result['simple_better'] else "Complex"
        print(f"{i:<6} {result['author_id']:<18} "
              f"{result['dist_simple']:<10.4f} {result['dist_complex']:<10.4f} "
              f"{better:<10} {result['intra_real']:<12.4f}")
    
    print(f"\n{'='*80}")
    print(f"INTERPRETATION:")
    print(f"  - Lower distance = Better mimicry (closer to real author style)")
    print(f"  - Intra Real = Natural variation in author's real documents")
    print(f"  - Look for: dist_simple/complex < intra_real (generated is more consistent than real!)")
    print(f"{'='*80}\n")
    
    # Print plot paths
    print(f"PLOT LOCATIONS:")
    for i, result in enumerate(results, 1):
        plot_path = PLOTS_DIR / model_key / f"fullrun{full_run}" / f"simple_vs_complex_{result['author_id']}.png"
        if plot_path.exists():
            print(f"  {i}. {plot_path}")
        else:
            print(f"  {i}. [NOT GENERATED YET] {plot_path}")
    
    print()


def save_to_file(model_key: str, full_run: int, results: List[Dict], top_n: int):
    """Save results to a text file."""
    output_file = CONSISTENCY_DIR / f"top_{top_n}_authors_{model_key}_fullrun{full_run}.txt"
    
    with open(output_file, 'w') as f:
        f.write(f"TOP {top_n} AUTHORS FOR MODEL: {model_key} (Full Run {full_run})\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"{'Rank':<6} {'Author ID':<18} {'Simple':<10} {'Complex':<10} {'Better':<10} {'Intra Real':<12}\n")
        f.write(f"{'-'*80}\n")
        
        for i, result in enumerate(results, 1):
            better = "Simple" if result['simple_better'] else "Complex"
            f.write(f"{i:<6} {result['author_id']:<18} "
                   f"{result['dist_simple']:<10.4f} {result['dist_complex']:<10.4f} "
                   f"{better:<10} {result['intra_real']:<12.4f}\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"PLOT PATHS:\n")
        for i, result in enumerate(results, 1):
            plot_path = PLOTS_DIR / model_key / f"fullrun{full_run}" / f"simple_vs_complex_{result['author_id']}.png"
            f.write(f"  {i}. {plot_path}\n")
    
    print(f"[SAVED] Results written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Identify best mimicry examples for presentation"
    )
    parser.add_argument(
        "--model-key",
        type=str,
        choices=STYLE_MODEL_KEYS,
        help="Style embedding model to analyze",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Analyze all models",
    )
    parser.add_argument(
        "--full-run",
        type=int,
        default=1,
        choices=[1, 2],
        help="Experimental run number (default: 1)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top authors to identify (default: 10)",
    )
    parser.add_argument(
        "--criteria",
        type=str,
        default="simple",
        choices=["simple", "complex", "both"],
        help="Ranking criteria: 'simple', 'complex', or 'both' (default: simple)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to text files",
    )
    
    args = parser.parse_args()
    
    if not args.model_key and not args.all_models:
        parser.error("Must specify either --model-key or --all-models")
    
    models = STYLE_MODEL_KEYS if args.all_models else [args.model_key]
    
    for model_key in models:
        df = load_analysis_results(model_key, args.full_run)
        
        if df is None:
            continue
        
        results = identify_best_authors(df, args.top_n, args.criteria)
        
        if not results:
            print(f"[WARNING] No results found for {model_key}")
            continue
        
        print_results(model_key, args.full_run, results, args.top_n)
        
        if args.save:
            save_to_file(model_key, args.full_run, results, args.top_n)


if __name__ == "__main__":
    main()
