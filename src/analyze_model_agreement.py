# src/analyze_model_agreement.py
"""
Enhanced model comparison analysis to understand why models disagree
and identify the most reliable set of consistent authors.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

from .model_configs import CONSISTENCY_DIR, MODEL_CONFIGS


def load_top_k(model_key: str, k: int = 100) -> pd.DataFrame:
    """Load top-K authors for a given model."""
    path = CONSISTENCY_DIR / f"{model_key}_top100.csv"
    df = pd.read_csv(path)
    df = df.sort_values("mean_style_distance", ascending=True).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df[df["rank"] <= k].copy()


def compute_pairwise_rank_correlations(rank_threshold: int = 100):
    """
    Compute rank correlations between all pairs of models for their overlapping authors.
    
    Returns DataFrame with columns: model_1, model_2, overlap_size, spearman_r, spearman_p, kendall_tau, kendall_p
    """
    model_keys = list(MODEL_CONFIGS.keys())
    results = []
    
    for i, mk1 in enumerate(model_keys):
        df1 = load_top_k(mk1, k=rank_threshold)
        for mk2 in model_keys[i+1:]:
            df2 = load_top_k(mk2, k=rank_threshold)
            
            # Find overlapping authors
            overlap = set(df1["author_id"]).intersection(set(df2["author_id"]))
            
            if len(overlap) < 3:  # Need at least 3 for correlation
                continue
            
            # Get ranks for overlapping authors
            df1_overlap = df1[df1["author_id"].isin(overlap)].set_index("author_id")
            df2_overlap = df2[df2["author_id"].isin(overlap)].set_index("author_id")
            
            # Align by author_id
            common_authors = list(overlap)
            ranks1 = [df1_overlap.loc[aid, "rank"] for aid in common_authors]
            ranks2 = [df2_overlap.loc[aid, "rank"] for aid in common_authors]
            
            # Compute correlations
            spearman_r, spearman_p = spearmanr(ranks1, ranks2)
            kendall_tau, kendall_p = kendalltau(ranks1, ranks2)
            
            results.append({
                "model_1": mk1,
                "model_2": mk2,
                "overlap_size": len(overlap),
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "kendall_tau": kendall_tau,
                "kendall_p": kendall_p
            })
    
    return pd.DataFrame(results)


def analyze_agreement_by_threshold(max_k: int = 100):
    """
    For each K from 10 to max_k, compute how many authors appear in at least N models.
    
    Returns DataFrame with columns: k_threshold, in_all_6, in_5_plus, in_4_plus, in_3_plus
    """
    results = []
    
    for k in range(10, max_k + 1, 10):
        author_counts = {}
        
        for model_key in MODEL_CONFIGS.keys():
            df = load_top_k(model_key, k=k)
            for author_id in df["author_id"]:
                author_counts[author_id] = author_counts.get(author_id, 0) + 1
        
        in_all_6 = sum(1 for count in author_counts.values() if count == 6)
        in_5_plus = sum(1 for count in author_counts.values() if count >= 5)
        in_4_plus = sum(1 for count in author_counts.values() if count >= 4)
        in_3_plus = sum(1 for count in author_counts.values() if count >= 3)
        
        results.append({
            "k_threshold": k,
            "in_all_6": in_all_6,
            "in_5_plus": in_5_plus,
            "in_4_plus": in_4_plus,
            "in_3_plus": in_3_plus
        })
    
    return pd.DataFrame(results)


def get_consensus_authors(min_models: int = 4, rank_threshold: int = 100):
    """
    Get authors that appear in at least min_models, sorted by:
    1. Number of models they appear in (descending)
    2. Average rank across models they appear in (ascending)
    
    Returns DataFrame with author_id, num_models, avg_rank, median_rank, and per-model ranks/distances
    """
    author_data = {}
    
    for model_key in MODEL_CONFIGS.keys():
        df = load_top_k(model_key, k=rank_threshold)
        for _, row in df.iterrows():
            aid = str(row["author_id"])
            if aid not in author_data:
                author_data[aid] = {
                    "author_id": aid,
                    "models": [],
                    "ranks": [],
                    "distances": []
                }
            author_data[aid]["models"].append(model_key)
            author_data[aid]["ranks"].append(row["rank"])
            author_data[aid]["distances"].append(row["mean_style_distance"])
    
    # Filter by min_models and compute stats
    consensus = []
    for aid, data in author_data.items():
        if len(data["models"]) >= min_models:
            entry = {
                "author_id": aid,
                "num_models": len(data["models"]),
                "avg_rank": np.mean(data["ranks"]),
                "median_rank": np.median(data["ranks"]),
                "std_rank": np.std(data["ranks"]),
                "avg_distance": np.mean(data["distances"]),
                "models_list": ",".join(data["models"])
            }
            # Add per-model details
            for i, model_key in enumerate(MODEL_CONFIGS.keys()):
                if model_key in data["models"]:
                    idx = data["models"].index(model_key)
                    entry[f"rank_{model_key}"] = data["ranks"][idx]
                    entry[f"dist_{model_key}"] = data["distances"][idx]
                else:
                    entry[f"rank_{model_key}"] = None
                    entry[f"dist_{model_key}"] = None
            
            consensus.append(entry)
    
    df = pd.DataFrame(consensus)
    if not df.empty:
        df = df.sort_values(["num_models", "avg_rank"], ascending=[False, True]).reset_index(drop=True)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze agreement/disagreement between embedding models on author consistency rankings."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=CONSISTENCY_DIR,
        help="Directory to save analysis outputs (default: data/consistency/)"
    )
    parser.add_argument(
        "--min_models",
        type=int,
        default=4,
        help="Minimum number of models an author must appear in for consensus list (default: 4)"
    )
    parser.add_argument(
        "--rank_threshold",
        type=int,
        default=100,
        help="Consider top-K authors from each model (default: 100)"
    )
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("MODEL AGREEMENT ANALYSIS")
    print("=" * 80)
    
    # 1. Pairwise rank correlations
    print("\n[1/3] Computing pairwise rank correlations between models...")
    df_corr = compute_pairwise_rank_correlations(rank_threshold=args.rank_threshold)
    corr_path = args.output_dir / "model_rank_correlations.csv"
    df_corr.to_csv(corr_path, index=False)
    print(f"   Saved to: {corr_path}")
    print(f"\n   Average Spearman correlation: {df_corr['spearman_r'].mean():.3f}")
    print(f"   Range: {df_corr['spearman_r'].min():.3f} to {df_corr['spearman_r'].max():.3f}")
    print("\n   Top 3 most correlated model pairs:")
    print(df_corr.nlargest(3, "spearman_r")[["model_1", "model_2", "spearman_r", "overlap_size"]].to_string(index=False))
    print("\n   Top 3 least correlated model pairs:")
    print(df_corr.nsmallest(3, "spearman_r")[["model_1", "model_2", "spearman_r", "overlap_size"]].to_string(index=False))
    
    # 2. Agreement by threshold analysis
    print("\n[2/3] Analyzing agreement across different K thresholds...")
    df_agreement = analyze_agreement_by_threshold(max_k=args.rank_threshold)
    agreement_path = args.output_dir / "agreement_by_threshold.csv"
    df_agreement.to_csv(agreement_path, index=False)
    print(f"   Saved to: {agreement_path}")
    print("\n   Agreement summary (authors appearing in N+ models):")
    print(df_agreement.to_string(index=False))
    
    # 3. Consensus authors
    print(f"\n[3/3] Finding consensus authors (appearing in ≥{args.min_models} models)...")
    df_consensus = get_consensus_authors(
        min_models=args.min_models,
        rank_threshold=args.rank_threshold
    )
    consensus_path = args.output_dir / f"consensus_authors_min{args.min_models}models.csv"
    df_consensus.to_csv(consensus_path, index=False)
    print(f"   Saved to: {consensus_path}")
    print(f"   Found {len(df_consensus)} consensus authors")
    
    if not df_consensus.empty:
        print(f"\n   Distribution by number of models:")
        print(df_consensus["num_models"].value_counts().sort_index(ascending=False).to_string())
        print(f"\n   Top 10 consensus authors (by num_models, then avg_rank):")
        display_cols = ["author_id", "num_models", "avg_rank", "median_rank", "std_rank"]
        print(df_consensus[display_cols].head(10).to_string(index=False))
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nGenerated files in {args.output_dir}:")
    print(f"  - model_rank_correlations.csv")
    print(f"  - agreement_by_threshold.csv")
    print(f"  - consensus_authors_min{args.min_models}models.csv")
    print(f"\nRECOMMENDATION:")
    if not df_consensus.empty:
        print(f"  Use the {len(df_consensus)} consensus authors from consensus_authors_min{args.min_models}models.csv")
        print(f"  These authors appear in at least {args.min_models} out of 6 models' top-{args.rank_threshold}.")
    else:
        print(f"  Consider lowering --min_models threshold (current: {args.min_models})")


if __name__ == "__main__":
    main()
