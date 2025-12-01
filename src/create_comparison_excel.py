# src/create_comparison_excel.py
"""
Create a comprehensive Excel workbook comparing all 6 models' top-100 consistency rankings.
Includes multiple sheets for different perspectives on the data.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .model_configs import CONSISTENCY_DIR, MODEL_CONFIGS


def load_top_k(model_key: str, k: int = 100) -> pd.DataFrame:
    """Load top-K authors for a given model."""
    path = CONSISTENCY_DIR / f"{model_key}_top100.csv"
    df = pd.read_csv(path)
    df = df.sort_values("mean_style_distance", ascending=True).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df[df["rank"] <= k].copy()


def create_wide_comparison_sheet(k: int = 100) -> pd.DataFrame:
    """
    Create a wide-format comparison with one row per author.
    Shows rank and distance for each model side-by-side.
    """
    combined = {}
    
    for model_key in MODEL_CONFIGS.keys():
        df = load_top_k(model_key, k=k)
        for _, row in df.iterrows():
            aid = str(row["author_id"])
            if aid not in combined:
                combined[aid] = {"author_id": aid}
            
            combined[aid][f"{model_key}_rank"] = int(row["rank"])
            combined[aid][f"{model_key}_distance"] = float(row["mean_style_distance"])
            combined[aid][f"{model_key}_num_reviews"] = int(row["num_reviews_total"])
    
    df = pd.DataFrame(list(combined.values()))
    
    # Add presence indicators and summary stats
    for model_key in MODEL_CONFIGS.keys():
        rank_col = f"{model_key}_rank"
        if rank_col not in df.columns:
            df[rank_col] = np.nan
        df[f"{model_key}_present"] = df[rank_col].notna()
    
    present_cols = [f"{model_key}_present" for model_key in MODEL_CONFIGS.keys()]
    df["num_models"] = df[present_cols].sum(axis=1)
    
    # Average rank across models where present
    rank_cols = [f"{model_key}_rank" for model_key in MODEL_CONFIGS.keys()]
    df["avg_rank"] = df[rank_cols].mean(axis=1, skipna=True)
    df["median_rank"] = df[rank_cols].median(axis=1, skipna=True)
    df["std_rank"] = df[rank_cols].std(axis=1, skipna=True)
    
    # Sort by num_models (desc), then avg_rank (asc)
    df = df.sort_values(["num_models", "avg_rank"], ascending=[False, True]).reset_index(drop=True)
    
    # Reorder columns nicely
    base_cols = ["author_id", "num_models", "avg_rank", "median_rank", "std_rank"]
    model_cols = []
    for model_key in MODEL_CONFIGS.keys():
        model_cols.extend([
            f"{model_key}_rank",
            f"{model_key}_distance",
            f"{model_key}_num_reviews",
            f"{model_key}_present"
        ])
    
    return df[base_cols + model_cols]


def create_long_comparison_sheet(k: int = 100) -> pd.DataFrame:
    """
    Create a long-format comparison with one row per (author, model) pair.
    Easier for filtering and pivot tables.
    """
    rows = []
    
    for model_key in MODEL_CONFIGS.keys():
        df = load_top_k(model_key, k=k)
        for _, row in df.iterrows():
            rows.append({
                "author_id": str(row["author_id"]),
                "model": model_key,
                "rank": int(row["rank"]),
                "mean_style_distance": float(row["mean_style_distance"]),
                "num_reviews_total": int(row["num_reviews_total"]),
                "num_reviews_used": int(row["num_reviews_used"])
            })
    
    df_long = pd.DataFrame(rows)
    df_long = df_long.sort_values(["author_id", "model"]).reset_index(drop=True)
    
    return df_long


def create_summary_sheet() -> pd.DataFrame:
    """
    Create a summary statistics sheet showing per-model statistics.
    """
    rows = []
    
    for model_key in MODEL_CONFIGS.keys():
        df = load_top_k(model_key, k=100)
        
        rows.append({
            "model": model_key,
            "num_authors_top100": len(df),
            "mean_distance": df["mean_style_distance"].mean(),
            "median_distance": df["mean_style_distance"].median(),
            "std_distance": df["mean_style_distance"].std(),
            "min_distance": df["mean_style_distance"].min(),
            "max_distance": df["mean_style_distance"].max(),
            "avg_reviews_per_author": df["num_reviews_total"].mean(),
            "median_reviews_per_author": df["num_reviews_total"].median()
        })
    
    return pd.DataFrame(rows)


def create_overlap_matrix() -> pd.DataFrame:
    """
    Create a matrix showing how many authors overlap between each pair of models' top-100.
    """
    model_keys = list(MODEL_CONFIGS.keys())
    matrix = pd.DataFrame(index=model_keys, columns=model_keys, dtype=int)
    
    author_sets = {}
    for model_key in model_keys:
        df = load_top_k(model_key, k=100)
        author_sets[model_key] = set(df["author_id"])
    
    for mk1 in model_keys:
        for mk2 in model_keys:
            overlap = len(author_sets[mk1].intersection(author_sets[mk2]))
            matrix.loc[mk1, mk2] = overlap
    
    return matrix


def create_consensus_sheet(min_models: int = 4) -> pd.DataFrame:
    """
    Create a sheet with only authors appearing in at least min_models.
    """
    df_wide = create_wide_comparison_sheet(k=100)
    df_consensus = df_wide[df_wide["num_models"] >= min_models].copy()
    
    return df_consensus


def main():
    parser = argparse.ArgumentParser(
        description="Create comprehensive Excel workbook comparing all 6 models' top-100 consistency rankings."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_comparison_analysis.xlsx",
        help="Output Excel filename (will be saved in data/consistency/)"
    )
    parser.add_argument(
        "--min_consensus",
        type=int,
        default=4,
        help="Minimum models for consensus sheet (default: 4)"
    )
    
    args = parser.parse_args()
    output_path = CONSISTENCY_DIR / args.output
    
    print("=" * 80)
    print("CREATING COMPREHENSIVE COMPARISON EXCEL WORKBOOK")
    print("=" * 80)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Wide format comparison (main sheet)
        print("\n[1/6] Creating 'All Authors (Wide)' sheet...")
        df_wide = create_wide_comparison_sheet(k=100)
        df_wide.to_excel(writer, sheet_name="All Authors (Wide)", index=False)
        print(f"   ✓ Added {len(df_wide)} unique authors")
        
        # Sheet 2: Long format (for pivot tables)
        print("\n[2/6] Creating 'All Authors (Long)' sheet...")
        df_long = create_long_comparison_sheet(k=100)
        df_long.to_excel(writer, sheet_name="All Authors (Long)", index=False)
        print(f"   ✓ Added {len(df_long)} rows (author-model pairs)")
        
        # Sheet 3: Summary statistics
        print("\n[3/6] Creating 'Model Summary' sheet...")
        df_summary = create_summary_sheet()
        df_summary.to_excel(writer, sheet_name="Model Summary", index=False)
        print(f"   ✓ Added statistics for {len(df_summary)} models")
        
        # Sheet 4: Overlap matrix
        print("\n[4/6] Creating 'Overlap Matrix' sheet...")
        df_overlap = create_overlap_matrix()
        df_overlap.to_excel(writer, sheet_name="Overlap Matrix")
        print(f"   ✓ Added {len(df_overlap)}x{len(df_overlap.columns)} overlap matrix")
        
        # Sheet 5: Consensus authors (appear in ≥4 models)
        print(f"\n[5/6] Creating 'Consensus (≥{args.min_consensus} models)' sheet...")
        df_consensus = create_consensus_sheet(min_models=args.min_consensus)
        df_consensus.to_excel(writer, sheet_name=f"Consensus (≥{args.min_consensus} models)", index=False)
        print(f"   ✓ Added {len(df_consensus)} consensus authors")
        
        # Sheet 6: Perfect agreement (all 6 models)
        print("\n[6/6] Creating 'All 6 Models' sheet...")
        df_all6 = df_wide[df_wide["num_models"] == 6].copy()
        df_all6.to_excel(writer, sheet_name="All 6 Models", index=False)
        print(f"   ✓ Added {len(df_all6)} authors appearing in all 6 models")
    
    print("\n" + "=" * 80)
    print("WORKBOOK CREATED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nLocation: {output_path}")
    print(f"\nSheets created:")
    print(f"  1. All Authors (Wide)     - One row per author, all models side-by-side")
    print(f"  2. All Authors (Long)     - One row per (author, model) pair")
    print(f"  3. Model Summary          - Per-model statistics")
    print(f"  4. Overlap Matrix         - Author overlap between model pairs")
    print(f"  5. Consensus (≥{args.min_consensus} models) - Authors in ≥{args.min_consensus} models")
    print(f"  6. All 6 Models           - Authors in all 6 models (perfect agreement)")
    
    print(f"\nKey findings:")
    print(f"  • Total unique authors: {len(df_wide)}")
    print(f"  • In all 6 models: {len(df_all6)}")
    print(f"  • In ≥{args.min_consensus} models: {len(df_consensus)}")
    
    if len(df_consensus) > 0:
        print(f"\n  RECOMMENDATION: Use the {len(df_consensus)} consensus authors from sheet 5")
        print(f"  for your next analysis step.")


if __name__ == "__main__":
    main()
