# src/compute_global_tsne.py
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from .model_configs import (
    EMBEDDINGS_DIR,
    CONSISTENCY_DIR,
    TSNE_RANDOM_STATE,
    TSNE_PERPLEXITY,
    TSNE_N_COMPONENTS,
    PCA_N_COMPONENTS,
    MODEL_CONFIGS,
)


def build_tsne_for_model(model_key: str):
    metrics_path = CONSISTENCY_DIR / f"{model_key}_all_authors.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing consistency file: {metrics_path}")

    df = pd.read_csv(metrics_path)
    selection_map = {}
    for _, row in df.iterrows():
        author_id = str(row["author_id"])
        try:
            sel = json.loads(row["selected_indices"])
        except Exception:
            sel = []
        selection_map[author_id] = [int(x) for x in sel]

    model_dir = EMBEDDINGS_DIR / model_key
    if not model_dir.is_dir():
        raise FileNotFoundError(f"No embeddings directory for model {model_key}: {model_dir}")

    all_embeddings = []
    all_author_ids = []
    all_doc_indices = []

    npz_files = sorted(model_dir.glob("*.npz"))

    for npz_path in tqdm(npz_files, desc=f"tsne-{model_key}"):
        data = np.load(npz_path, allow_pickle=True)
        author_id = str(data["author_id"])
        embeddings = data["embeddings"].astype(np.float32)
        n_docs = embeddings.shape[0]
        if n_docs == 0:
            continue

        sel_indices = selection_map.get(author_id, [])
        if n_docs >= 6 and len(sel_indices) == 6:
            use_idx = np.array(sel_indices, dtype=int)
        else:
            # Either fewer than 6 reviews or no selection info; use all.
            use_idx = np.arange(n_docs, dtype=int)

        selected = embeddings[use_idx]
        all_embeddings.append(selected)
        all_author_ids.extend([author_id] * selected.shape[0])
        all_doc_indices.extend(use_idx.tolist())

    X = np.vstack(all_embeddings).astype(np.float32)
    author_ids_arr = np.array(all_author_ids, dtype=object)
    doc_indices_arr = np.array(all_doc_indices, dtype=np.int32)

    # Optional PCA to 50 dims
    if PCA_N_COMPONENTS is not None and X.shape[1] > PCA_N_COMPONENTS:
        pca = PCA(n_components=PCA_N_COMPONENTS, random_state=TSNE_RANDOM_STATE)
        X_reduced = pca.fit_transform(X)
    else:
        X_reduced = X

    tsne = TSNE(
        n_components=TSNE_N_COMPONENTS,
        perplexity=TSNE_PERPLEXITY,
        init="pca",
        random_state=TSNE_RANDOM_STATE,
        learning_rate="auto",
        max_iter=1000,  # Changed from n_iter to max_iter for scikit-learn compatibility
        verbose=1,
    )
    coords = tsne.fit_transform(X_reduced).astype(np.float32)

    out_path = EMBEDDINGS_DIR / f"global_tsne_{model_key}.npz"
    np.savez_compressed(
        out_path,
        coords=coords,
        author_ids=author_ids_arr,
        doc_indices=doc_indices_arr,
        model_key=model_key,
    )
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute global 2D t-SNE per model using selected 6 reviews when possible."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Which model_keys to process; 'all' runs every model in MODEL_CONFIGS.",
    )
    args = parser.parse_args()

    if "all" in args.models:
        model_keys = list(MODEL_CONFIGS.keys())
    else:
        model_keys = args.models

    for mk in model_keys:
        build_tsne_for_model(mk)


if __name__ == "__main__":
    main()