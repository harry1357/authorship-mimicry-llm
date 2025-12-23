#!/usr/bin/env python3
"""
Identify Best Mimicry Examples for Presentation

This script analyzes the simple vs complex CSV results and identifies the top
authors for each model based on how well the generated texts matched their style.

By default it uses the HONEST metrics:
    - dist_to_training_simple
    - dist_to_training_complex

i.e., distance from generated texts to the *actual training documents*, not just
the centroid.

Usage:
    python src/identify_best_mimicry.py --model-key style_embedding --full-run 1 --top-n 10
    
    # Or analyze all models at once
    python src/identify_best_mimicry.py --all-models --full-run 1 --top-n 10
"""

import argparse
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

from generation_config import CONSISTENCY_DIR, STYLE_MODEL_KEYS
from model_configs import PLOTS_DIR


def load_analysis_results(model_key: str, llm_key: str, full_run: int) -> Optional[pd.DataFrame]:
    """Load the CSV analysis results for a model and LLM."""
    csv_path = CONSISTENCY_DIR / f"simple_vs_complex_{model_key}_{llm_key}_fullrun{full_run}.csv"
    
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        print(f"Run: python src/analyse_simple_vs_complex.py --model-key {model_key} --llm-key {llm_key} --full-run {full_run}")
        return None
    
    df = pd.read_csv(csv_path)
    
    required_cols = {
        "author_id",
        "dist_to_training_simple",
        "dist_to_training_complex",
        "dist_real_centroid_simple",
        "dist_real_centroid_complex",
        "intra_real",
        "intra_simple",
        "intra_complex",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[ERROR] CSV is missing expected columns: {missing}")
        return None
    
    return df


def identify_best_authors(
    df: pd.DataFrame, 
    top_n: int = 10,
    criteria: str = "simple"
) -> List[Dict]:
    """
    Identify top authors based on mimicry quality.

    Ranking uses HONEST metrics (distance to actual training docs):
      - criteria="simple":  dist_to_training_simple
      - criteria="complex": dist_to_training_complex
      - criteria="both":    average of both
    """
    df = df.copy()

    if criteria == "simple":
        df_sorted = df.sort_values("dist_to_training_simple")
    elif criteria == "complex":
        df_sorted = df.sort_values("dist_to_training_complex")
    else:  # "both"
        df["avg_dist_to_training"] = (
            df["dist_to_training_simple"] + df["dist_to_training_complex"]
        ) / 2.0
        df_sorted = df.sort_values("avg_dist_to_training")

    results: List[Dict] = []
    for _, row in df_sorted.head(top_n).iterrows():
        dist_simple_train = float(row["dist_to_training_simple"])
        dist_complex_train = float(row["dist_to_training_complex"])
        intra_real = float(row["intra_real"])
        
        results.append(
            {
                "author_id": row["author_id"],
                # HONEST metrics
                "dist_simple": dist_simple_train,
                "dist_complex": dist_complex_train,
                # Legacy centroid metrics (for reference)
                "dist_simple_centroid": float(row["dist_real_centroid_simple"]),
                "dist_complex_centroid": float(row["dist_real_centroid_complex"]),
                # Baselines / intra distances
                "intra_real": intra_real,
                "intra_simple": float(row["intra_simple"]),
                "intra_complex": float(row["intra_complex"]),
                # Who is better under honest metric?
                "simple_better": dist_simple_train < dist_complex_train,
                # Excess distance above intra-real baseline (smaller = better mimicry)
                "excess_simple": dist_simple_train - intra_real,
                "excess_complex": dist_complex_train - intra_real,
            }
        )
    
    return results


def print_results(model_key: str, llm_key: str, full_run: int, results: List[Dict], top_n: int):
    """Print formatted results."""
    print(f"\n{'='*80}")
    print(f"TOP {top_n} AUTHORS FOR MODEL: {model_key}, LLM: {llm_key} (Full Run {full_run})")
    print(f"{'='*80}\n")
    
    print(
        f"{'Rank':<6} {'Author ID':<18} "
        f"{'S→train':<10} {'C→train':<10} "
        f"{'Better':<8} {'IntraReal':<10}"
    )
    print(f"{'-'*80}")
    
    for i, r in enumerate(results, 1):
        better = "Simple" if r["simple_better"] else "Complex"
        print(
            f"{i:<6} {r['author_id']:<18} "
            f"{r['dist_simple']:<10.4f} {r['dist_complex']:<10.4f} "
            f"{better:<8} {r['intra_real']:<10.4f}"
        )
    
    print(f"\n{'='*80}")
    print("INTERPRETATION:")
    print("  - Distances shown are to REAL TRAINING DOCS (not centroids).")
    print("  - Lower distance = better mimicry (generated closer to author’s real texts).")
    print("  - IntraReal ≈ natural variation between the author's real documents.")
    print("  - Strong mimicry: dist_simple/complex is close to or below IntraReal.")
    print(f"{'='*80}\n")
    
    # Print plot paths
    print("PLOT LOCATIONS:")
    for i, r in enumerate(results, 1):
        plot_path = PLOTS_DIR / model_key / f"fullrun{full_run}" / f"simple_vs_complex_{r['author_id']}.png"
        if plot_path.exists():
            print(f"  {i}. {plot_path}")
        else:
            print(f"  {i}. [NOT GENERATED YET] {plot_path}")
    
    print()


def save_to_file(model_key: str, llm_key: str, full_run: int, results: List[Dict], top_n: int):
    """Save results to a text file."""
    output_file = CONSISTENCY_DIR / f"top_{top_n}_authors_{model_key}_{llm_key}_fullrun{full_run}.txt"
    
    with output_file.open("w", encoding="utf-8") as f:
        f.write(f"TOP {top_n} AUTHORS FOR MODEL: {model_key}, LLM: {llm_key} (Full Run {full_run})\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(
            f"{'Rank':<6} {'Author ID':<18} "
            f"{'S→train':<10} {'C→train':<10} "
            f"{'Better':<8} {'IntraReal':<10}\n"
        )
        f.write(f"{'-'*80}\n")
        
        for i, r in enumerate(results, 1):
            better = "Simple" if r["simple_better"] else "Complex"
            f.write(
                f"{i:<6} {r['author_id']:<18} "
                f"{r['dist_simple']:<10.4f} {r['dist_complex']:<10.4f} "
                f"{better:<8} {r['intra_real']:<10.4f}\n"
            )
        
        f.write(f"\n{'='*80}\n")
        f.write("PLOT PATHS:\n")
        for i, r in enumerate(results, 1):
            plot_path = PLOTS_DIR / model_key / f"fullrun{full_run}" / f"simple_vs_complex_{r['author_id']}.png"
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
        "--llm-key",
        type=str,
        required=True,
        help="LLM key (e.g., gpt-5.2-2025-12-11, gemini-3-pro-preview)",
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
        help="Ranking criteria using HONEST metrics: 'simple', 'complex', or 'both' (default: simple)",
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
        df = load_analysis_results(model_key, args.llm_key, args.full_run)
        if df is None:
            continue
        
        results = identify_best_authors(df, args.top_n, args.criteria)
        if not results:
            print(f"[WARNING] No results found for {model_key}")
            continue
        
        print_results(model_key, args.llm_key, args.full_run, results, args.top_n)
        
        if args.save:
            save_to_file(model_key, args.llm_key, args.full_run, results, args.top_n)


if __name__ == "__main__":
    main()