#!/usr/bin/env python3
"""
Compute INDEPENDENT topic/semantic similarity measures.

These measures are computed in embedding spaces/methods that are
DIFFERENT from the LUAR authorship embedding used for mimicry evaluation.

This allows us to test whether the strong correlation between
train-gen similarity and mimicry success is:
  1. A genuine topical effect (sports→sports works better than sports→beauty)
  2. An artifact of using the same embedding space for both measurements

Independent measures implemented:
  - Sentence-BERT (all-MiniLM-L6-v2): Generic semantic similarity
  - TF-IDF cosine similarity: Pure lexical/topic overlap
  - Jaccard similarity: Token overlap

Usage:
    python src/compute_independent_similarity.py --model-key luar_mud_orig --llm-key gpt-5.2-pro --full-run 1
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

# Try importing sentence-transformers (may need installation)
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("[WARNING] sentence-transformers not installed. Install with: pip install sentence-transformers")

from generation_config import CORPUS_DIR, GENERATED_DIR, EMBEDDINGS_DIR, REFERENCE_MODEL_KEY


def load_texts_from_npz(author_id: str, model_key: str, base_path: Path) -> List[str]:
    """Load training texts for an author from embeddings npz file."""
    emb_path = base_path / "data" / "embeddings" / model_key / f"{author_id}.npz"
    if not emb_path.exists():
        return []
    
    data = np.load(emb_path, allow_pickle=True)
    files = data.get("files")
    if files is None:
        return []
    
    training_files = files[:6]  # First 6 are training
    texts = []
    for file_rel in training_files:
        file_path = CORPUS_DIR / file_rel
        if file_path.exists():
            texts.append(file_path.read_text(encoding="utf-8"))
    return texts


def load_generated_texts_from_dir(
    author_id: str,
    llm_key: str,
    prompt_variant: str,
    full_run: int,
    base_path: Path
) -> List[str]:
    """Load generated texts for an author."""
    gen_dir = (base_path / "data" / "generated" / llm_key / "normalized" / 
              f"texts_{prompt_variant}_fullrun{full_run}" / author_id)
    
    if not gen_dir.exists():
        return []
    
    texts = []
    for file_path in sorted(gen_dir.glob("*.txt")):
        texts.append(file_path.read_text(encoding="utf-8"))
    return texts


def compute_sbert_similarity(training_texts: List[str], generated_texts: List[str]) -> Dict[str, float]:
    """
    Compute semantic similarity using Sentence-BERT (all-MiniLM-L6-v2).
    
    This is a GENERIC semantic embedding model trained on diverse datasets,
    NOT specialized for authorship. It captures topical/semantic similarity.
    
    Returns:
        Dict with mean, max, min similarities
    """
    if not SBERT_AVAILABLE:
        return {
            'mean_sbert_sim': np.nan,
            'max_sbert_sim': np.nan,
            'min_sbert_sim': np.nan
        }
    
    if not training_texts or not generated_texts:
        return {
            'mean_sbert_sim': np.nan,
            'max_sbert_sim': np.nan,
            'min_sbert_sim': np.nan
        }
    
    # Load model (cached after first load)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode texts
    train_embeddings = model.encode(training_texts, convert_to_numpy=True, show_progress_bar=False)
    gen_embeddings = model.encode(generated_texts, convert_to_numpy=True, show_progress_bar=False)
    
    # Compute pairwise similarities
    similarities = []
    for train_emb in train_embeddings:
        for gen_emb in gen_embeddings:
            # Cosine similarity = 1 - cosine distance
            sim = 1 - cosine(train_emb, gen_emb)
            similarities.append(sim)
    
    similarities = np.array(similarities)
    
    return {
        'mean_sbert_sim': np.mean(similarities),
        'max_sbert_sim': np.max(similarities),
        'min_sbert_sim': np.min(similarities)
    }


def compute_tfidf_similarity(training_texts: List[str], generated_texts: List[str]) -> Dict[str, float]:
    """
    Compute TF-IDF cosine similarity.
    
    This is a pure LEXICAL measure - it captures topic overlap through
    shared vocabulary, independent of any neural embeddings.
    
    Returns:
        Dict with mean, max, min similarities
    """
    if not training_texts or not generated_texts:
        return {
            'mean_tfidf_sim': np.nan,
            'max_tfidf_sim': np.nan,
            'min_tfidf_sim': np.nan
        }
    
    # Combine all texts for vectorization
    all_texts = training_texts + generated_texts
    
    # Compute TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    except ValueError:
        # Handle edge case where texts are too short or empty
        return {
            'mean_tfidf_sim': np.nan,
            'max_tfidf_sim': np.nan,
            'min_tfidf_sim': np.nan
        }
    
    # Split back into training and generated
    n_train = len(training_texts)
    train_vectors = tfidf_matrix[:n_train]
    gen_vectors = tfidf_matrix[n_train:]
    
    # Compute pairwise similarities
    similarities = []
    for i in range(n_train):
        for j in range(len(generated_texts)):
            sim = sklearn_cosine_similarity(train_vectors[i], gen_vectors[j])[0, 0]
            similarities.append(sim)
    
    similarities = np.array(similarities)
    
    return {
        'mean_tfidf_sim': np.mean(similarities),
        'max_tfidf_sim': np.max(similarities),
        'min_tfidf_sim': np.min(similarities)
    }


def compute_jaccard_similarity(training_texts: List[str], generated_texts: List[str]) -> Dict[str, float]:
    """
    Compute Jaccard similarity (token overlap).
    
    This is the simplest lexical overlap measure - just counts
    shared tokens between texts.
    
    Returns:
        Dict with mean, max, min similarities
    """
    if not training_texts or not generated_texts:
        return {
            'mean_jaccard_sim': np.nan,
            'max_jaccard_sim': np.nan,
            'min_jaccard_sim': np.nan
        }
    
    def get_tokens(text: str) -> set:
        """Tokenize and lowercase."""
        return set(text.lower().split())
    
    # Compute pairwise Jaccard similarities
    similarities = []
    for train_text in training_texts:
        train_tokens = get_tokens(train_text)
        for gen_text in generated_texts:
            gen_tokens = get_tokens(gen_text)
            
            if not train_tokens or not gen_tokens:
                similarities.append(0.0)
                continue
            
            intersection = len(train_tokens & gen_tokens)
            union = len(train_tokens | gen_tokens)
            jaccard = intersection / union if union > 0 else 0.0
            similarities.append(jaccard)
    
    similarities = np.array(similarities)
    
    return {
        'mean_jaccard_sim': np.mean(similarities),
        'max_jaccard_sim': np.max(similarities),
        'min_jaccard_sim': np.min(similarities)
    }


def compute_all_independent_similarities(
    training_texts: List[str],
    generated_texts: List[str]
) -> Dict[str, float]:
    """
    Compute all independent similarity measures.
    
    Returns a dict with all metrics combined.
    """
    result = {}
    
    # Sentence-BERT (generic semantic)
    sbert_sim = compute_sbert_similarity(training_texts, generated_texts)
    result.update(sbert_sim)
    
    # TF-IDF (lexical/topical)
    tfidf_sim = compute_tfidf_similarity(training_texts, generated_texts)
    result.update(tfidf_sim)
    
    # Jaccard (token overlap)
    jaccard_sim = compute_jaccard_similarity(training_texts, generated_texts)
    result.update(jaccard_sim)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute independent topic/semantic similarity measures"
    )
    parser.add_argument("--model-key", type=str, required=True,
                       help="Embedding model key for loading training texts")
    parser.add_argument("--llm-key", type=str, required=True,
                       help="LLM key for loading generated texts")
    parser.add_argument("--full-run", type=int, default=1,
                       help="Full run number")
    
    args = parser.parse_args()
    
    base_path = Path(__file__).parent.parent
    
    print(f"\n=== Computing Independent Similarity Measures ===")
    print(f"Model: {args.model_key}")
    print(f"LLM: {args.llm_key}")
    print(f"Full run: {args.full_run}\n")
    
    if not SBERT_AVAILABLE:
        print("[WARNING] Sentence-BERT not available - only TF-IDF and Jaccard will be computed")
        print("Install with: pip install sentence-transformers\n")
    
    # Load training embeddings to get author list
    embeddings_dir = base_path / "data" / "embeddings" / args.model_key
    author_files = sorted(embeddings_dir.glob("*.npz"))
    
    # Load generated texts for both prompts
    print("Loading generated texts...")
    gen_simple_dir = base_path / "data" / "generated" / args.llm_key / "normalized" / f"texts_simple_fullrun{args.full_run}"
    gen_complex_dir = base_path / "data" / "generated" / args.llm_key / "normalized" / f"texts_complex_fullrun{args.full_run}"
    
    if not gen_simple_dir.exists() or not gen_complex_dir.exists():
        print(f"[ERROR] Generated texts not found for {args.llm_key} run {args.full_run}")
        return
    
    # Get authors with both prompt variants
    authors_simple = {d.name for d in gen_simple_dir.iterdir() if d.is_dir()}
    authors_complex = {d.name for d in gen_complex_dir.iterdir() if d.is_dir()}
    common_authors = authors_simple & authors_complex
    
    print(f"Found {len(common_authors)} authors with both prompt variants")
    
    # Compute similarities for all authors
    results = []
    
    for i, author_id in enumerate(sorted(common_authors), 1):
        if i % 10 == 0:
            print(f"Processing {i}/{len(common_authors)} authors...")
        
        # Load training texts
        training_texts = load_texts_from_npz(author_id, args.model_key, base_path)
        if not training_texts:
            continue
        
        # Load generated texts (simple)
        gen_texts_simple = load_generated_texts_from_dir(
            author_id, args.llm_key, "simple", args.full_run, base_path
        )
        
        # Load generated texts (complex)
        gen_texts_complex = load_generated_texts_from_dir(
            author_id, args.llm_key, "complex", args.full_run, base_path
        )
        
        if not gen_texts_simple or not gen_texts_complex:
            continue
        
        # Compute similarities for simple prompt
        sim_simple = compute_all_independent_similarities(training_texts, gen_texts_simple)
        sim_simple_renamed = {f"{k}_simple": v for k, v in sim_simple.items()}
        
        # Compute similarities for complex prompt
        sim_complex = compute_all_independent_similarities(training_texts, gen_texts_complex)
        sim_complex_renamed = {f"{k}_complex": v for k, v in sim_complex.items()}
        
        # Compute similarities for pooled generations (all 4 texts)
        all_generated = gen_texts_simple + gen_texts_complex
        sim_avg = compute_all_independent_similarities(training_texts, all_generated)
        sim_avg_renamed = {f"{k}_avg": v for k, v in sim_avg.items()}
        
        # Combine all metrics
        result = {
            'author_id': author_id,
            **sim_simple_renamed,
            **sim_complex_renamed,
            **sim_avg_renamed
        }
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    print(f"\nComputed independent similarities for {len(df)} authors")
    
    # Save results
    output_dir = base_path / "data" / "independent_similarity" / args.model_key / args.llm_key
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"independent_similarity_fullrun{args.full_run}.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\nSaved results to: {output_file}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print("\nSimple Prompt:")
    for metric in ['mean_sbert_sim_simple', 'mean_tfidf_sim_simple', 'mean_jaccard_sim_simple']:
        if metric in df.columns:
            print(f"  {metric}: {df[metric].mean():.3f} ± {df[metric].std():.3f}")
    
    print("\nComplex Prompt:")
    for metric in ['mean_sbert_sim_complex', 'mean_tfidf_sim_complex', 'mean_jaccard_sim_complex']:
        if metric in df.columns:
            print(f"  {metric}: {df[metric].mean():.3f} ± {df[metric].std():.3f}")
    
    print("\nAverage (Pooled):")
    for metric in ['mean_sbert_sim_avg', 'mean_tfidf_sim_avg', 'mean_jaccard_sim_avg']:
        if metric in df.columns:
            print(f"  {metric}: {df[metric].mean():.3f} ± {df[metric].std():.3f}")
    
    print("\n=== Next Steps ===")
    print("1. Merge these independent similarities with your author_factors CSV")
    print("2. Rerun correlations to see if TF-IDF/SBERT/Jaccard also predict mimicry")
    print("3. Compare correlation strengths: LUAR vs independent measures")
    print("\nThis will help determine if the strong correlation is:")
    print("  - Topical (independent measures also correlate)")
    print("  - Authorship-space artifact (only LUAR correlates)")


if __name__ == "__main__":
    main()
