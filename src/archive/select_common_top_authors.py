#!/usr/bin/env python
"""
select_common_top_authors.py

Given per-model style-consistency CSVs, this script:

  1. For each model, sorts by mean_style_distance ascending and keeps top K.
  2. Finds authors that appear in ALL models' top K lists.
  3. Computes average rank across models for those common authors.
  4. Writes out a CSV with the final top M authors, plus per-model ranks.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_and_rank(csv_path: Path, model_name: str, top_k: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.sort_values("mean_style_distance", ascending=True).reset_index(drop=True)
    df = df.head(top_k).copy()
    df[f"rank_{model_name}"] = np.arange(1, len(df) + 1)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anna_csv", type=str, required=True)
    parser.add_argument("--luar_st_csv", type=str, required=True)
    parser.add_argument("--luar_raw_csv", type=str, required=True)
    parser.add_argument("--star_csv", type=str, required=True)
    parser.add_argument(
        "--per_model_top_k",
        type=int,
        default=150,
        help="How many most-consistent authors to keep per model (default: 150).",
    )
    parser.add_argument(
        "--final_top_k",
        type=int,
        default=100,
        help="How many common authors to keep in the final list (default: 100).",
    )
    parser.add_argument(
        "--out_common_csv",
        type=str,
        default="data/author_style_consistent_common_top100.csv",
    )
    parser.add_argument(
        "--out_ids_txt",
        type=str,
        default="data/author_ids_common_top100.txt",
    )
    args = parser.parse_args()

    anna_df = load_and_rank(Path(args.anna_csv), "anna", args.per_model_top_k)
    luar_st_df = load_and_rank(Path(args.luar_st_csv), "luar_st", args.per_model_top_k)
    luar_raw_df = load_and_rank(Path(args.luar_raw_csv), "luar_raw", args.per_model_top_k)
    star_df = load_and_rank(Path(args.star_csv), "star", args.per_model_top_k)

    anna_df = anna_df[["author_id", "mean_style_distance", "rank_anna"]].rename(
        columns={"mean_style_distance": "mean_dist_anna"}
    )
    luar_st_df = luar_st_df[["author_id", "mean_style_distance", "rank_luar_st"]].rename(
        columns={"mean_style_distance": "mean_dist_luar_st"}
    )
    luar_raw_df = luar_raw_df[["author_id", "mean_style_distance", "rank_luar_raw"]].rename(
        columns={"mean_style_distance": "mean_dist_luar_raw"}
    )
    star_df = star_df[["author_id", "mean_style_distance", "rank_star"]].rename(
        columns={"mean_style_distance": "mean_dist_star"}
    )

    merged = anna_df.merge(luar_st_df, on="author_id", how="inner")
    merged = merged.merge(luar_raw_df, on="author_id", how="inner")
    merged = merged.merge(star_df, on="author_id", how="inner")

    print(
        f"Number of authors in top-{args.per_model_top_k} for ALL models: "
        f"{len(merged)}"
    )

    rank_cols = ["rank_anna", "rank_luar_st", "rank_luar_raw", "rank_star"]
    merged["avg_rank"] = merged[rank_cols].mean(axis=1)

    merged = merged.sort_values("avg_rank", ascending=True).reset_index(drop=True)

    final_top_k = min(args.final_top_k, len(merged))
    final = merged.head(final_top_k).copy()

    out_common_csv = Path(args.out_common_csv)
    out_common_csv.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_common_csv, index=False)

    out_ids_txt = Path(args.out_ids_txt)
    out_ids_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_ids_txt.open("w", encoding="utf-8") as f:
        for aid in final["author_id"]:
            f.write(str(aid) + "\n")

    print(f"Wrote final top-{final_top_k} common authors to {out_common_csv}")
    print(f"Wrote author ID list to {out_ids_txt}")


if __name__ == "__main__":
    main()