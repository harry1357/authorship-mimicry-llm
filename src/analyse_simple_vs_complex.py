# src/analyse_simple_vs_complex.py
"""
Comparative Analysis Module for Simple vs Complex Prompt Variants

This module compares the stylistic consistency of texts generated using simple
versus complex prompts by analyzing their embeddings in relation to real author texts.

The analysis computes:
1. Distance from generated texts to real author centroid (style consistency)
2. Intra-variant distances (generation diversity within each prompt type)
3. Intra-real distances (baseline consistency of real author texts)

Additionally, the module can generate t-SNE visualizations for individual authors
to illustrate the clustering behavior of real vs generated texts in the embedding space.

Metrics:
    All distances use cosine distance (1 - cosine_similarity) for interpretability.

Output:
    - CSV file: data/consistency/simple_vs_complex_<model-key>_fullrun<N>.csv
    - Optional plots: data/plots/simple_vs_complex_<model-key>_<author-id>_fullrun<N>.png

Usage:
    # Generate analysis CSV
    python src/analyse_simple_vs_complex.py --model-key style_embedding --full-run 1

    # Generate t-SNE plot for specific author
    python src/analyse_simple_vs_complex.py --model-key style_embedding --full-run 1 \\
        --tsne-author-id A12345ABCDE
"""

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm

from generation_config import (
    EMBEDDINGS_DIR, 
    CONSISTENCY_DIR, 
    STYLE_MODEL_KEYS,
    REFERENCE_CONSISTENCY_CSV,
)
from model_configs import PLOTS_DIR, TSNE_RANDOM_STATE, TSNE_PERPLEXITY, PCA_N_COMPONENTS


def load_selected_indices() -> Dict[str, List[int]]:
    """
    Load pre-computed indices of the 6 most consistent reviews for each author.
    
    These are the training reviews used to generate the prompts.
    
    Returns:
        Dictionary mapping author IDs to lists of selected review indices
    """
    mapping: Dict[str, List[int]] = {}
    
    if not REFERENCE_CONSISTENCY_CSV.exists():
        print(f"[WARNING] Reference consistency CSV not found: {REFERENCE_CONSISTENCY_CSV}")
        return mapping
    
    with REFERENCE_CONSISTENCY_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            author_id = row["author_id"]
            raw = row.get("selected_indices", "").strip()
            if not raw:
                continue
            try:
                indices = ast.literal_eval(raw)
            except Exception:
                # Fallback parsing
                indices = [
                    int(x)
                    for x in raw.replace("[", "").replace("]", "").split(",")
                    if x.strip()
                ]
            indices = [int(i) for i in indices]
            if indices:
                mapping[author_id] = indices
    
    return mapping


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute cosine distance (1 - cosine_similarity) between vectors.
    
    Args:
        X: Array of shape (n_samples_X, n_features)
        Y: Array of shape (n_samples_Y, n_features)
        
    Returns:
        Array of shape (n_samples_X, n_samples_Y) containing cosine distances
    """
    sim = cosine_similarity(X, Y)
    return 1.0 - sim


def load_real_embeddings(
    model_key: str, 
    author_id: str,
    use_training_only: bool = False
) -> Optional[np.ndarray]:
    """
    Load real author embeddings from the standard location.
    
    Args:
        model_key: Style embedding model identifier
        author_id: Author identifier
        use_training_only: If True, load only the 6 training documents.
                          Uses selected_indices if available, otherwise first 6 files.
        
    Returns:
        2D array of embeddings, or None if file doesn't exist
    """
    emb_path = EMBEDDINGS_DIR / model_key / f"{author_id}.npz"
    
    if not emb_path.exists():
        return None
    
    try:
        data = np.load(emb_path, allow_pickle=True)
        embeddings = data["embeddings"]
        
        if use_training_only:
            # Load selected indices for this author
            selected_indices_map = load_selected_indices()
            
            if author_id in selected_indices_map:
                indices = selected_indices_map[author_id]
                # Validate indices
                if len(indices) >= 6 and max(indices) < len(embeddings) and min(indices) >= 0:
                    # Use the first 6 selected indices
                    indices_to_use = sorted(indices)[:6]
                    embeddings = embeddings[indices_to_use]
                    # print(f"[DEBUG] Using selected indices for {author_id}: {indices_to_use}")
                else:
                    # Fallback: use first 6 files
                    if len(embeddings) >= 6:
                        embeddings = embeddings[:6]
                        print(f"[WARNING] Selected indices invalid for {author_id}, using first 6 files")
                    else:
                        print(f"[WARNING] Author {author_id} has fewer than 6 embeddings ({len(embeddings)})")
            else:
                # Fallback: use first 6 files (matches build_generation_prompts.py behavior)
                if len(embeddings) >= 6:
                    embeddings = embeddings[:6]
                    # print(f"[DEBUG] No selected indices for {author_id}, using first 6 files")
                else:
                    print(f"[WARNING] Author {author_id} has fewer than 6 embeddings ({len(embeddings)})")
        
        return embeddings
    except Exception as e:
        print(f"[analyse] ERROR loading real embeddings for {author_id}: {e}")
        return None


def load_generated_embeddings(
    model_key: str,
    llm_key: str,
    prompt_variant: str,
    full_run: int,
    author_id: str,
) -> Optional[np.ndarray]:
    """
    Load generated text embeddings.
    
    Args:
        model_key: Style embedding model identifier
        llm_key: LLM identifier
        prompt_variant: 'simple' or 'complex'
        full_run: Experimental run number
        author_id: Author identifier
        
    Returns:
        2D array of embeddings, or None if file doesn't exist
    """
    emb_path = (
        EMBEDDINGS_DIR / "generated" / model_key / llm_key / 
        prompt_variant / f"fullrun{full_run}" / f"{author_id}.npz"
    )
    
    if not emb_path.exists():
        return None
    
    try:
        data = np.load(emb_path, allow_pickle=True)
        return data["embeddings"]
    except Exception as e:
        print(f"[analyse] ERROR loading generated embeddings for {author_id} "
              f"({prompt_variant}): {e}")
        return None


def compute_intra_distance(embeddings: np.ndarray) -> float:
    """
    Compute mean pairwise cosine distance within a set of embeddings.
    
    This measures the internal consistency/diversity of the embedding set.
    
    Args:
        embeddings: 2D array of shape (n_samples, n_features)
        
    Returns:
        Mean pairwise cosine distance
    """
    if len(embeddings) < 2:
        return 0.0
    
    # Compute pairwise distances
    dist_matrix = cosine_distance(embeddings, embeddings)
    
    # Extract upper triangle (excluding diagonal)
    n = len(embeddings)
    upper_triangle_indices = np.triu_indices(n, k=1)
    pairwise_dists = dist_matrix[upper_triangle_indices]
    
    return float(np.mean(pairwise_dists))


def compute_distance_to_centroid(
    embeddings: np.ndarray, 
    centroid: np.ndarray
) -> float:
    """
    Compute mean cosine distance from embeddings to a centroid.
    
    Args:
        embeddings: 2D array of shape (n_samples, n_features)
        centroid: 1D array of shape (n_features,)
        
    Returns:
        Mean distance to centroid
    """
    centroid_2d = centroid.reshape(1, -1)
    distances = cosine_distance(embeddings, centroid_2d)
    return float(np.mean(distances))


def compute_distance_to_training_docs(
    embeddings: np.ndarray,
    training_docs: np.ndarray
) -> float:
    """
    Compute mean cosine distance from embeddings to actual training documents.
    
    This is more honest than distance to centroid, which can be artificially
    optimistic since the centroid is an abstract average point.
    
    Args:
        embeddings: 2D array of shape (n_samples, n_features) - generated docs
        training_docs: 2D array of shape (n_training, n_features) - real docs
        
    Returns:
        Mean distance to all training documents
    """
    distances = cosine_distance(embeddings, training_docs)
    return float(np.mean(distances))


def analyze_author(
    author_id: str,
    model_key: str,
    llm_key: str,
    full_run: int,
) -> Optional[Dict[str, float]]:
    """
    Compute all metrics for a single author.
    
    Args:
        author_id: Author identifier
        model_key: Style embedding model
        llm_key: LLM identifier
        full_run: Experimental run number
        
    Returns:
        Dictionary of metrics, or None if data is missing
    """
    # Load all embeddings - use ONLY the 6 training documents for real
    # This ensures consistency with the t-SNE plots
    real_embs = load_real_embeddings(model_key, author_id, use_training_only=True)
    simple_embs = load_generated_embeddings(
        model_key, llm_key, "simple", full_run, author_id
    )
    complex_embs = load_generated_embeddings(
        model_key, llm_key, "complex", full_run, author_id
    )
    
    # Check if we have all necessary data
    if real_embs is None:
        return None
    
    if simple_embs is None or complex_embs is None:
        return None
    
    # Compute real author centroid
    real_centroid = np.mean(real_embs, axis=0)
    
    # Compute distances to real centroid (LEGACY - may be optimistic)
    dist_real_to_complex = compute_distance_to_centroid(complex_embs, real_centroid)
    dist_real_to_simple = compute_distance_to_centroid(simple_embs, real_centroid)
    
    # Compute distances to actual training documents (MORE HONEST METRIC)
    dist_to_training_complex = compute_distance_to_training_docs(complex_embs, real_embs)
    dist_to_training_simple = compute_distance_to_training_docs(simple_embs, real_embs)
    
    # Compute intra-distances
    intra_real = compute_intra_distance(real_embs)
    intra_complex = compute_intra_distance(complex_embs)
    intra_simple = compute_intra_distance(simple_embs)
    
    return {
        # Legacy metrics (distance to centroid - may be optimistic)
        "dist_real_centroid_complex": dist_real_to_complex,
        "dist_real_centroid_simple": dist_real_to_simple,
        # New honest metrics (distance to actual training docs)
        "dist_to_training_complex": dist_to_training_complex,
        "dist_to_training_simple": dist_to_training_simple,
        # Intra-distances (for reference)
        "intra_real": intra_real,
        "intra_complex": intra_complex,
        "intra_simple": intra_simple,
    }


def run_analysis(
    model_key: str,
    llm_key: str,
    full_run: int,
) -> Path:
    """
    Run complete analysis across all authors and save results to CSV.
    
    Args:
        model_key: Style embedding model identifier
        llm_key: LLM identifier
        full_run: Experimental run number
        
    Returns:
        Path to the generated CSV file
    """
    print(f"[analyse] Starting analysis: model={model_key}, llm={llm_key}, run={full_run}")
    
    # Get list of authors from GENERATED embeddings directory (not real!)
    # This ensures we only analyze authors who have generated texts
    gen_simple_dir = EMBEDDINGS_DIR / "generated" / model_key / llm_key / "simple" / f"fullrun{full_run}"
    gen_complex_dir = EMBEDDINGS_DIR / "generated" / model_key / llm_key / "complex" / f"fullrun{full_run}"
    
    if not gen_simple_dir.exists() or not gen_complex_dir.exists():
        raise FileNotFoundError(
            f"Generated embeddings not found.\n"
            f"Expected:\n  {gen_simple_dir}\n  {gen_complex_dir}\n"
            f"Please run:\n"
            f"  python src/embed_generated_texts.py --model-key {model_key} --full-run {full_run} --prompt-variant simple\n"
            f"  python src/embed_generated_texts.py --model-key {model_key} --full-run {full_run} --prompt-variant complex"
        )
    
    # Get intersection of authors who have both simple and complex generated texts
    simple_authors = {f.stem for f in gen_simple_dir.glob("*.npz")}
    complex_authors = {f.stem for f in gen_complex_dir.glob("*.npz")}
    author_ids = sorted(simple_authors & complex_authors)
    
    print(f"[analyse] Found {len(author_ids)} authors with complete data (simple + complex)")
    
    # Output CSV path - include LLM key to distinguish between different models
    CONSISTENCY_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CONSISTENCY_DIR / f"simple_vs_complex_{model_key}_{llm_key}_fullrun{full_run}.csv"
    
    # Analyze each author
    results = []
    missing_count = 0
    
    for author_id in tqdm(author_ids, desc="Analyzing authors"):
        metrics = analyze_author(author_id, model_key, llm_key, full_run)
        
        if metrics is None:
            missing_count += 1
            continue
        
        results.append({
            "author_id": author_id,
            "model_key": model_key,
            "llm_key": llm_key,
            "full_run": full_run,
            **metrics,
        })
    
    # Write CSV
    if not results:
        print(f"[analyse] ERROR: No valid results to write")
        return csv_path
    
    # Sort results by simple prompt performance (using HONEST metric - distance to training)
    results_sorted = sorted(results, key=lambda x: x["dist_to_training_simple"])
    
    fieldnames = [
        "author_id",
        "model_key",
        "llm_key",
        "full_run",
        # HONEST METRICS (distance to actual training docs) - USE THESE!
        "dist_to_training_complex",
        "dist_to_training_simple",
        # Legacy metrics (distance to centroid - may be optimistic)
        "dist_real_centroid_complex",
        "dist_real_centroid_simple",
        # Intra-distances for reference
        "intra_real",
        "intra_complex",
        "intra_simple",
    ]
    
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_sorted)
        
        # Add summary metrics at the bottom of CSV
        if results:
            complex_dists = [r["dist_to_training_complex"] for r in results]
            simple_dists = [r["dist_to_training_simple"] for r in results]
            avg_complex = np.mean(complex_dists)
            avg_simple = np.mean(simple_dists)
            avg_overall = (avg_complex + avg_simple) / 2
            
            # Write blank row separator
            f.write("\n")
            # Write summary header
            f.write("# MIMICRY PERFORMANCE METRICS (for LLM comparison)\n")
            f.write(f"# Average dist_to_training_complex,{avg_complex:.6f}\n")
            f.write(f"# Average dist_to_training_simple,{avg_simple:.6f}\n")
            f.write(f"# Overall average (both prompts),{avg_overall:.6f}\n")
            f.write(f"# Baseline (intra-author variation),{np.mean([r['intra_real'] for r in results]):.6f}\n")
            f.write(f"# Number of authors,{len(results)}\n")
    
    print(f"[analyse] Wrote {len(results_sorted)} rows to {csv_path}")
    print(f"[analyse] Sorted by HONEST metric: dist_to_training_simple (distance to actual training docs)")
    if missing_count > 0:
        print(f"[analyse] WARNING: {missing_count} authors skipped (missing real embeddings with training data)")
    
    # Print summary statistics
    if results:
        # New honest metrics
        complex_dists_honest = [r["dist_to_training_complex"] for r in results]
        simple_dists_honest = [r["dist_to_training_simple"] for r in results]
        # Legacy metrics for comparison
        complex_dists_legacy = [r["dist_real_centroid_complex"] for r in results]
        simple_dists_legacy = [r["dist_real_centroid_simple"] for r in results]
        # Intra-real for context
        intra_real_dists = [r["intra_real"] for r in results]
        
        avg_complex = np.mean(complex_dists_honest)
        avg_simple = np.mean(simple_dists_honest)
        
        print(f"\n[analyse] Summary Statistics:")
        print(f"=" * 80)
        print(f"HONEST METRICS (Distance to Actual Training Documents):")
        print(f"  Complex prompt: {avg_complex:.4f} ± {np.std(complex_dists_honest):.4f}")
        print(f"  Simple prompt:  {avg_simple:.4f} ± {np.std(simple_dists_honest):.4f}")
        print(f"  Difference:     {avg_simple - avg_complex:.4f}")
        print(f"\n⭐ MIMICRY PERFORMANCE METRICS (for LLM comparison):")
        print(f"  Average dist_to_training_complex: {avg_complex:.6f}")
        print(f"  Average dist_to_training_simple:  {avg_simple:.6f}")
        print(f"  Overall average (both prompts):   {(avg_complex + avg_simple) / 2:.6f}")
        print(f"\nLegacy Metrics (Distance to Centroid - may be optimistic):")
        print(f"  Complex prompt: {np.mean(complex_dists_legacy):.4f} ± {np.std(complex_dists_legacy):.4f}")
        print(f"  Simple prompt:  {np.mean(simple_dists_legacy):.4f} ± {np.std(simple_dists_legacy):.4f}")
        print(f"  Difference:     {np.mean(simple_dists_legacy) - np.mean(complex_dists_legacy):.4f}")
        print(f"\nCentroid Optimism (how much closer centroid appears vs actual docs):")
        print(f"  Complex: {np.mean(complex_dists_honest) - np.mean(complex_dists_legacy):.4f}")
        print(f"  Simple:  {np.mean(simple_dists_honest) - np.mean(simple_dists_legacy):.4f}")
        print(f"\nBaseline (Training Internal Distance): {np.mean(intra_real_dists):.4f}")
        print(f"=" * 80)
        print(f"\n💡 Use these metrics to compare LLM mimicry performance:")
        print(f"   Lower distance = Better mimicry")
        print(f"   Baseline (intra-author variation) = {np.mean(intra_real_dists):.4f}")
        print(f"=" * 80)
    
    return csv_path


def generate_tsne_plot(
    author_id: str,
    model_key: str,
    llm_key: str,
    full_run: int,
) -> Optional[Path]:
    """
    Generate t-SNE visualization for a specific author.
    
    Shows the 6 training documents (real) clustered with simple and complex generated texts.
    
    Args:
        author_id: Author identifier
        model_key: Style embedding model
        llm_key: LLM identifier
        full_run: Experimental run number
        
    Returns:
        Path to saved plot, or None if data is missing
    """
    print(f"[tsne] Generating plot for author {author_id}")
    
    # Load embeddings - use only the 6 training documents for real
    real_embs = load_real_embeddings(model_key, author_id, use_training_only=True)
    simple_embs = load_generated_embeddings(
        model_key, llm_key, "simple", full_run, author_id
    )
    complex_embs = load_generated_embeddings(
        model_key, llm_key, "complex", full_run, author_id
    )
    
    if real_embs is None or simple_embs is None or complex_embs is None:
        print(f"[tsne] ERROR: Missing embeddings for {author_id}")
        return None
    
    # Combine all embeddings
    all_embeddings = np.vstack([real_embs, simple_embs, complex_embs])
    
    # Create labels
    n_real = len(real_embs)
    n_simple = len(simple_embs)
    n_complex = len(complex_embs)
    
    labels = (
        ["Real (Training)"] * n_real +
        ["Simple"] * n_simple +
        ["Complex"] * n_complex
    )
    
    print(f"[tsne] Total embeddings: {len(all_embeddings)} "
          f"(real training={n_real}, simple={n_simple}, complex={n_complex})")
    
    # Optional PCA dimensionality reduction before t-SNE
    # PCA n_components must be <= min(n_samples, n_features)
    n_samples = all_embeddings.shape[0]
    n_features = all_embeddings.shape[1]
    
    if n_features > PCA_N_COMPONENTS and n_samples > PCA_N_COMPONENTS:
        # Only apply PCA if we have enough samples
        print(f"[tsne] Applying PCA: {n_features} -> {PCA_N_COMPONENTS} dims")
        pca = PCA(n_components=PCA_N_COMPONENTS, random_state=TSNE_RANDOM_STATE)
        all_embeddings = pca.fit_transform(all_embeddings)
    elif n_features > n_samples:
        # If features > samples but samples < PCA_N_COMPONENTS, reduce to n_samples - 1
        max_components = min(n_samples - 1, n_features)
        print(f"[tsne] Applying PCA: {n_features} -> {max_components} dims (limited by n_samples={n_samples})")
        pca = PCA(n_components=max_components, random_state=TSNE_RANDOM_STATE)
        all_embeddings = pca.fit_transform(all_embeddings)
    else:
        print(f"[tsne] Skipping PCA (n_features={n_features}, n_samples={n_samples})")
    
    # Run t-SNE
    # Perplexity must be less than n_samples
    perplexity = min(TSNE_PERPLEXITY, n_samples - 1)
    if perplexity < 5:
        perplexity = max(2, n_samples - 1)  # Minimum reasonable perplexity
    
    print(f"[tsne] Running t-SNE (perplexity={perplexity}, n_samples={n_samples})...")
    tsne = TSNE(
        n_components=2,
        random_state=TSNE_RANDOM_STATE,
        perplexity=perplexity,
        max_iter=1000,
    )
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define colors and markers
    colors = {"Real (Training)": "#1f77b4", "Simple": "#ff7f0e", "Complex": "#2ca02c"}
    markers = {"Real (Training)": "o", "Simple": "s", "Complex": "^"}
    hull_colors = {"Real (Training)": "#1f77b4", "Simple": "#ff7f0e", "Complex": "#2ca02c"}
    
    # Plot convex hulls first (behind the points)
    from scipy.spatial import ConvexHull
    for label in ["Real (Training)", "Simple", "Complex"]:
        mask = np.array(labels) == label
        points = embeddings_2d[mask]
        
        if len(points) >= 3:  # Need at least 3 points for a hull
            try:
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    ax.plot(points[simplex, 0], points[simplex, 1], 
                           color=hull_colors[label], alpha=0.3, linewidth=1.5)
                # Fill the hull
                hull_points = points[hull.vertices]
                ax.fill(hull_points[:, 0], hull_points[:, 1], 
                       color=hull_colors[label], alpha=0.1)
            except:
                pass  # Skip if hull can't be computed
    
    # Plot centroids
    for label in ["Real (Training)", "Simple", "Complex"]:
        mask = np.array(labels) == label
        points = embeddings_2d[mask]
        centroid = points.mean(axis=0)
        ax.scatter(centroid[0], centroid[1], 
                  c=colors[label], marker='X', s=300, 
                  edgecolors='black', linewidths=2, 
                  zorder=10, alpha=0.9)
    
    # Plot each group
    for label in ["Real (Training)", "Simple", "Complex"]:
        mask = np.array(labels) == label
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=colors[label],
            marker=markers[label],
            label=label,
            alpha=0.8,
            s=120,
            edgecolors='black',
            linewidths=0.8,
            zorder=5,
        )
    
    # Add legend entry for centroids
    ax.scatter([], [], c='gray', marker='X', s=300, 
              edgecolors='black', linewidths=2, 
              label='Centroids', alpha=0.9)
    
    ax.set_xlabel("t-SNE Dimension 1", fontsize=13, fontweight='bold')
    ax.set_ylabel("t-SNE Dimension 2", fontsize=13, fontweight='bold')
    ax.set_title(
        f"Author {author_id} - {model_key}\n"
        f"Training Documents vs Generated Texts (Run {full_run})",
        fontsize=15,
        fontweight='bold'
    )
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Make axes equal to preserve distances visually
    ax.set_aspect('equal', adjustable='box')
    
    # Save plot in model-specific subfolder
    model_plots_dir = PLOTS_DIR / model_key / f"fullrun{full_run}"
    model_plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = model_plots_dir / f"simple_vs_complex_{author_id}.png"
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[tsne] Saved plot to {plot_path}")
    
    return plot_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare simple vs complex prompt variants for authorship mimicry"
    )
    parser.add_argument(
        "--model-key",
        type=str,
        required=False,
        choices=STYLE_MODEL_KEYS,
        help="Style embedding model to use for analysis",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run analysis for all models",
    )
    parser.add_argument(
        "--llm-key",
        type=str,
        default="gpt-5.1",
        help="LLM identifier (default: gpt-5.1)",
    )
    parser.add_argument(
        "--full-run",
        type=int,
        default=1,
        choices=[1, 2],
        help="Experimental run number (default: 1)",
    )
    parser.add_argument(
        "--tsne-author-id",
        type=str,
        default=None,
        help="Generate t-SNE plot for specific author (optional)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all_models and not args.model_key:
        parser.error("Either --model-key or --all-models must be specified")
    
    if args.all_models and args.model_key:
        parser.error("Cannot specify both --model-key and --all-models")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Determine which models to run
    models = STYLE_MODEL_KEYS if args.all_models else [args.model_key]
    
    # Run main analysis
    if args.tsne_author_id is None:
        for model_key in models:
            print(f"\n{'='*80}")
            print(f"Running analysis for model: {model_key}")
            print(f"{'='*80}\n")
            
            run_analysis(
                model_key=model_key,
                llm_key=args.llm_key,
                full_run=args.full_run,
            )
    else:
        # Generate t-SNE plot for specific author (only first model if --all-models)
        model_key = models[0]
        generate_tsne_plot(
            author_id=args.tsne_author_id,
            model_key=model_key,
            llm_key=args.llm_key,
            full_run=args.full_run,
        )


if __name__ == "__main__":
    main()
