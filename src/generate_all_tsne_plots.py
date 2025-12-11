# src/generate_all_tsne_plots.py
"""
Batch t-SNE Plot Generation for All Authors

This script generates individual t-SNE visualizations for all authors with available
embeddings, showing how their real documents cluster with simple and complex generated texts.

Each plot shows:
- Real documents (blue circles)
- Simple-generated texts (orange squares)
- Complex-generated texts (green triangles)

Output:
    One plot per author saved to: data/plots/simple_vs_complex_<model>_<author>_fullrun<N>.png

Usage:
    # Generate plots for all authors with one model
    python src/generate_all_tsne_plots.py --model-key style_embedding --full-run 1
    
    # Generate for all models
    for model in luar_crud_orig luar_mud_orig luar_crud_st luar_mud_st style_embedding star; do
        python src/generate_all_tsne_plots.py --model-key $model --full-run 1
    done
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

# Import from analyse_simple_vs_complex with proper module path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from analyse_simple_vs_complex import generate_tsne_plot, load_real_embeddings, load_generated_embeddings
from generation_config import EMBEDDINGS_DIR, STYLE_MODEL_KEYS


def get_authors_with_complete_data(
    model_key: str,
    llm_key: str,
    full_run: int,
) -> list[str]:
    """
    Get list of authors that have all required embeddings (real, simple, complex).
    
    Args:
        model_key: Style embedding model identifier
        llm_key: LLM identifier
        full_run: Experimental run number
        
    Returns:
        List of author IDs with complete data
    """
    # Get all authors from real embeddings
    real_emb_dir = EMBEDDINGS_DIR / model_key
    
    if not real_emb_dir.exists():
        print(f"[ERROR] Real embeddings directory not found: {real_emb_dir}")
        print(f"Please run: python src/embed_authors.py --models {model_key}")
        return []
    
    author_files = sorted(real_emb_dir.glob("*.npz"))
    all_author_ids = [f.stem for f in author_files]
    
    # Filter to authors with complete data (real + simple + complex)
    complete_authors = []
    
    for author_id in all_author_ids:
        real_embs = load_real_embeddings(model_key, author_id, use_training_only=True)
        simple_embs = load_generated_embeddings(model_key, llm_key, "simple", full_run, author_id)
        complex_embs = load_generated_embeddings(model_key, llm_key, "complex", full_run, author_id)
        
        if real_embs is not None and simple_embs is not None and complex_embs is not None:
            complete_authors.append(author_id)
    
    return complete_authors


def generate_all_plots(
    model_key: str,
    llm_key: str,
    full_run: int,
    skip_existing: bool = True,
) -> None:
    """
    Generate t-SNE plots for all authors with complete embedding data.
    
    Args:
        model_key: Style embedding model identifier
        llm_key: LLM identifier
        full_run: Experimental run number
        skip_existing: If True, skip authors that already have plots
    """
    print(f"[batch_tsne] Starting batch plot generation")
    print(f"[batch_tsne] Model: {model_key}, LLM: {llm_key}, Run: {full_run}")
    
    # Get authors with complete data
    authors = get_authors_with_complete_data(model_key, llm_key, full_run)
    
    if not authors:
        print(f"[batch_tsne] ERROR: No authors found with complete embeddings")
        print(f"[batch_tsne] Make sure you've run:")
        print(f"  1. python src/embed_authors.py --models {model_key}")
        print(f"  2. python src/embed_generated_texts.py --model-key {model_key} --full-run {full_run} --prompt-variant simple")
        print(f"  3. python src/embed_generated_texts.py --model-key {model_key} --full-run {full_run} --prompt-variant complex")
        return
    
    print(f"[batch_tsne] Found {len(authors)} authors with complete data")
    
    # Generate plots
    skipped = 0
    generated = 0
    failed = 0
    
    for author_id in tqdm(authors, desc=f"Generating t-SNE plots"):
        # Check if plot already exists
        from model_configs import PLOTS_DIR
        model_plots_dir = PLOTS_DIR / model_key / f"fullrun{full_run}"
        plot_path = model_plots_dir / f"simple_vs_complex_{author_id}.png"
        
        if skip_existing and plot_path.exists():
            skipped += 1
            continue
        
        # Generate plot
        try:
            result = generate_tsne_plot(author_id, model_key, llm_key, full_run)
            if result is not None:
                generated += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[batch_tsne] ERROR generating plot for {author_id}: {e}")
            failed += 1
    
    print(f"\n[batch_tsne] Complete!")
    print(f"  Generated: {generated} plots")
    print(f"  Skipped: {skipped} (already existed)")
    print(f"  Failed: {failed}")
    
    if generated > 0:
        from model_configs import PLOTS_DIR
        print(f"\n[batch_tsne] Plots saved to: {PLOTS_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate t-SNE plots for all authors"
    )
    parser.add_argument(
        "--model-key",
        type=str,
        required=True,
        choices=STYLE_MODEL_KEYS,
        help="Style embedding model to use",
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
        "--overwrite",
        action="store_true",
        help="Regenerate plots even if they already exist",
    )
    
    args = parser.parse_args()
    
    generate_all_plots(
        model_key=args.model_key,
        llm_key=args.llm_key,
        full_run=args.full_run,
        skip_existing=not args.overwrite,
    )


if __name__ == "__main__":
    main()
