# src/compare_models.py

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model_configs import CONSISTENCY_DIR, MODEL_CONFIGS


def load_top100_for_model(model_key: str, top_k: int = 100) -> pd.DataFrame:
    """
    Load <model_key>_top100.csv and ensure we have rank 1..N.
    """
    path = CONSISTENCY_DIR / f"{model_key}_top100.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing top-100 file for {model_key}: {path}")

    df = pd.read_csv(path)
    # Just in case it's not sorted
    df = df.sort_values("mean_style_distance", ascending=True).reset_index(drop=True)
    df["rank"] = df.index + 1  # 1-based rank
    # Keep only up to requested top_k if file is longer
    df = df[df["rank"] <= top_k].copy()
    return df


def build_combined_table(top_k_for_rank: int = 100) -> pd.DataFrame:
    """
    Combine all per-model top-100 lists into a single wide table.

    Columns:
        author_id,
        for each model_key:
            rank_<model_key>,
            mean_style_distance_<model_key>,
            num_reviews_total_<model_key>,
            num_reviews_used_<model_key>,
            present_<model_key> (bool),
        num_models_present
    """
    combined = {}

    for model_key in MODEL_CONFIGS.keys():
        df = load_top100_for_model(model_key, top_k=top_k_for_rank)
        for _, row in df.iterrows():
            aid = str(row["author_id"])
            if aid not in combined:
                combined[aid] = {"author_id": aid}
            entry = combined[aid]

            entry[f"rank_{model_key}"] = int(row["rank"])
            entry[f"mean_style_distance_{model_key}"] = float(row["mean_style_distance"])
            entry[f"num_reviews_total_{model_key}"] = int(row["num_reviews_total"])
            entry[f"num_reviews_used_{model_key}"] = int(row["num_reviews_used"])
            entry[f"present_{model_key}"] = True

    # Fill in missing present_* flags as False
    for aid, entry in combined.items():
        for model_key in MODEL_CONFIGS.keys():
            if f"present_{model_key}" not in entry:
                entry[f"present_{model_key}"] = False

    df_combined = pd.DataFrame(list(combined.values()))

    # num_models_present
    present_cols = [f"present_{mk}" for mk in MODEL_CONFIGS.keys()]
    df_combined["num_models_present"] = df_combined[present_cols].sum(axis=1)

    # Sort: by num_models_present desc, then by sum of ranks (ignoring missing)
    rank_cols = [f"rank_{mk}" for mk in MODEL_CONFIGS.keys()]
    # Replace missing ranks with a large value for sorting
    for rc in rank_cols:
        if rc not in df_combined.columns:
            df_combined[rc] = np.nan
    df_combined["_rank_sum"] = df_combined[rank_cols].fillna(1e9).sum(axis=1)
    df_combined = df_combined.sort_values(
        ["num_models_present", "_rank_sum"], ascending=[False, True]
    ).reset_index(drop=True)
    df_combined.drop(columns=["_rank_sum"], inplace=True)

    return df_combined


def compute_topk_intersection(k: int = 50):
    """
    Compute the intersection of top-k authors across all models.

    Returns:
        intersection_authors : sorted list of author_ids
        model_to_topk_sets   : dict model_key -> set(author_ids)
    """
    model_to_sets = {}
    for model_key in MODEL_CONFIGS.keys():
        df = load_top100_for_model(model_key, top_k=k)
        model_to_sets[model_key] = set(str(a) for a in df["author_id"].tolist())

    # Intersection across all models
    all_sets = list(model_to_sets.values())
    if not all_sets:
        return [], model_to_sets

    intersection = set.intersection(*all_sets)
    intersection_sorted = sorted(intersection)
    return intersection_sorted, model_to_sets


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Combine per-model top-100 consistency CSVs into one comparison table, "
            "and compute intersection of top-K authors across models."
        )
    )
    parser.add_argument(
        "--rank_k",
        type=int,
        default=100,
        help="Use ranks up to this K from each <model>_top100.csv when building the combined table (default 100).",
    )
    parser.add_argument(
        "--intersection_k",
        type=int,
        default=50,
        help="Top-K for intersection across models (default 50).",
    )
    parser.add_argument(
        "--output_combined",
        type=str,
        default="combined_top100_all_models.csv",
        help="Filename for combined comparison CSV (will be created under data/consistency/).",
    )
    parser.add_argument(
        "--output_intersection",
        type=str,
        default="intersection_topK_authors.txt",
        help="Filename for intersection author list (under data/consistency/).",
    )
    args = parser.parse_args()

    # 1) Build combined table
    df_combined = build_combined_table(top_k_for_rank=args.rank_k)
    combined_path = CONSISTENCY_DIR / args.output_combined
    df_combined.to_csv(combined_path, index=False)
    print(f"Wrote combined comparison table to: {combined_path}")

    # 2) Compute intersection of top-K authors
    intersection_authors, model_to_sets = compute_topk_intersection(k=args.intersection_k)
    print(f"\nTop-{args.intersection_k} intersection across all models:")
    print(f"- Number of authors in intersection: {len(intersection_authors)}")

    if intersection_authors:
        print("- Example authors:", ", ".join(intersection_authors[:10]), "...")
    else:
        print("- Intersection is empty for this K.")

    # Optional: print per-model top-k sizes (should all be <= K)
    for mk, s in model_to_sets.items():
        print(f"  {mk}: {len(s)} authors in top-{args.intersection_k}")

    # Save intersection to a simple text file (one author_id per line)
    out_intersection_path = CONSISTENCY_DIR / args.output_intersection
    with out_intersection_path.open("w", encoding="utf-8") as f:
        for aid in intersection_authors:
            f.write(aid + "\n")
    print(f"Saved intersection author list to: {out_intersection_path}")


if __name__ == "__main__":
    main()