import os
import glob
import argparse
from typing import List, Dict

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def load_author_ids(author_list_path: str) -> List[str]:
    """
    Read author IDs from author_ids_three_training_topics_x_two_two_generation_topics.txt.

    Assumes:
      - First non-empty line is a header with category names.
      - Each subsequent non-empty line starts with an author ID,
        followed by one or more category names.

    Example line:
      A1A1BM6N28X9J0 Automotive Baby ...

    We only care about the first token (author ID).
    """
    author_ids: List[str] = []

    with open(author_list_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"No non-empty lines found in {author_list_path}")

    # Skip header (first line)
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        author_id = parts[0]
        author_ids.append(author_id)

    return author_ids


def find_review_files_for_author(aavc_root: str, author_id: str) -> List[str]:
    """
    Find all .txt review files for a given author.

    Pattern:
      <aavc_root>/**/<author_id>_*.txt

    This is robust to nested directories under AAVC.
    Example:
      AAVC/A1A1BM6N28X9J0_Automotive.txt
      AAVC/some/folder/A1A1BM6N28X9J0_Electronics.txt
    """
    pattern = os.path.join(aavc_root, "**", f"{author_id}_*.txt")
    files = glob.glob(pattern, recursive=True)
    files = [f for f in files if f.lower().endswith(".txt")]
    return sorted(files)


def read_text_file(path: str) -> str:
    """
    Read a text file as UTF-8, fall back to latin-1 if needed.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


def build_corpus_for_author(
    aavc_root: str,
    author_id: str,
    max_reviews: int | None = None,
) -> Dict:
    """
    For one author:
      - Locate all their review files.
      - Read each file's text.
      - Optionally limit to first max_reviews files.

    Returns dict:
      {
        "author_id": author_id,
        "file_paths": [list of file paths],
        "texts": [list of review texts]
      }
    """
    review_files = find_review_files_for_author(aavc_root, author_id)
    if max_reviews is not None:
        review_files = review_files[:max_reviews]

    texts = [read_text_file(p) for p in review_files]

    return {
        "author_id": author_id,
        "file_paths": review_files,
        "texts": texts,
    }


def load_style_model(model_name: str = "AnnaWegmann/Style-Embedding") -> SentenceTransformer:
    """
    Load the style embedding model from Hugging Face via sentence-transformers.
    """
    print(f"Loading style model: {model_name}")
    model = SentenceTransformer(model_name)
    return model


def compute_style_embeddings(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """
    Encode a list of review texts into style embeddings.

    We normalise embeddings so cosine similarity becomes a dot product.
    """
    if not texts:
        # No texts: return an empty array with correct second dimension
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)

    embeddings = model.encode(
        texts,
        batch_size=4,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.astype(np.float32)


def save_author_embeddings(
    output_root: str,
    author_id: str,
    embeddings: np.ndarray,
    file_paths: List[str],
) -> None:
    """
    Save embeddings and metadata for one author to:

      data/style_embeddings/<AUTHOR_ID>.npz

    The .npz file contains:
      - "embeddings": (n_reviews, dim) float32
      - "file_paths": array of strings with original file paths
    """
    os.makedirs(output_root, exist_ok=True)
    out_path = os.path.join(output_root, f"{author_id}.npz")

    file_paths_arr = np.array(file_paths, dtype=object)

    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        file_paths=file_paths_arr,
    )


def main():
    # Determine repo root (one level up from src/)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(this_dir)

    parser = argparse.ArgumentParser(
        description="Vectorise AAVC reviews using AnnaWegmann/Style-Embedding."
    )
    parser.add_argument(
        "--aavc_root",
        type=str,
        default=os.path.join(repo_root, "amazon_product_data_corpus_mixed_topics_per_author_reformatted"),
        help="Path to the root folder containing the AAVC review .txt files.",
    )
    parser.add_argument(
        "--author_list",
        type=str,
        default=os.path.join(repo_root, "author_ids_three_training_topics_x_two_two_generation_topics.txt"),
        help="Path to the text file listing author IDs and their categories.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=os.path.join(repo_root, "data", "style_embeddings"),
        help="Folder where per-author embedding files will be saved.",
    )
    parser.add_argument(
        "--max_reviews",
        type=int,
        default=None,
        help="Optional limit on number of reviews per author (e.g., 6). Use None for all.",
    )
    parser.add_argument(
        "--limit_authors",
        type=int,
        default=None,
        help="Optional limit on number of authors to process (useful for quick tests).",
    )

    args = parser.parse_args()

    # Load author IDs
    print(f"Loading author IDs from: {args.author_list}")
    author_ids = load_author_ids(args.author_list)
    print(f"Total authors in list: {len(author_ids)}")

    if args.limit_authors is not None:
        author_ids = author_ids[: args.limit_authors]
        print(f"Limiting to first {len(author_ids)} authors for this run.")

    # Sanity check: AAVC root exists
    if not os.path.isdir(args.aavc_root):
        raise FileNotFoundError(
            f"AAVC root folder not found: {args.aavc_root}\n"
            "Please check the path or pass --aavc_root explicitly."
        )

    # Load model once
    model = load_style_model()

    # Process each author
    for author_id in tqdm(author_ids, desc="Authors"):
        corpus = build_corpus_for_author(
            aavc_root=args.aavc_root,
            author_id=author_id,
            max_reviews=args.max_reviews,
        )

        texts = corpus["texts"]
        file_paths = corpus["file_paths"]

        if not texts:
            print(f"[WARN] No review files found for author {author_id}. Skipping.")
            continue

        embeddings = compute_style_embeddings(model, texts)

        save_author_embeddings(
            output_root=args.output_root,
            author_id=author_id,
            embeddings=embeddings,
            file_paths=file_paths,
        )

    print("Done. Embeddings saved in:", args.output_root)


if __name__ == "__main__":
    main()