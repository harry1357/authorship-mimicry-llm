import os
import pandas as pd


def main():
    # repo_root = one level above src/
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(this_dir)

    in_path = os.path.join(repo_root, "data", "author_style_consistency.csv")
    out_table = os.path.join(repo_root, "data", "author_style_consistent_top100.csv")
    out_ids = os.path.join(repo_root, "data", "author_ids_top100.txt")

    if not os.path.isfile(in_path):
        raise FileNotFoundError(
            f"Input file not found: {in_path}\n"
            "Run compute_author_style_consistency.py first."
        )

    df = pd.read_csv(in_path)

    # Keep only authors with at least 6 reviews
    df = df[df["num_reviews"] == 6].copy()

    # Sort by mean_style_distance ascending (most consistent first)
    df = df.sort_values("mean_style_distance", na_position="last")

    # Take top 100
    top100 = df.head(100)

    os.makedirs(os.path.dirname(out_table), exist_ok=True)
    top100.to_csv(out_table, index=False)

    # Also save just the IDs (one per line)
    with open(out_ids, "w", encoding="utf-8") as f:
        for a in top100["author_id"]:
            f.write(str(a) + "\n")

    print(f"Saved top-100 author table to {out_table}")
    print(f"Saved top-100 author IDs to {out_ids}")
    print(top100.head(10))


if __name__ == "__main__":
    main()