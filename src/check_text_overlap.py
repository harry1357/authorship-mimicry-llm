#!/usr/bin/env python3
"""
Detect Text Overlap Between Training and Generated Reviews

This script checks if generated texts contain copied sentences or phrases
from the EXACT training documents used in their generation prompts.

Key: For each author, we have:
  - Prompt 1 (training docs 1-3) → generates text g1
  - Prompt 2 (training docs 4-6) → generates text g2

We compare g1 ONLY against training docs 1-3, and g2 ONLY against 4-6.

Usage:
    python src/check_text_overlap.py --author-id A2PR5G1680ISEY --full-run 1 --prompt-variant simple
    
    # Check multiple authors
    python src/check_text_overlap.py --check-top-n 10 --full-run 1
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Set, Dict
from collections import defaultdict
import difflib

import numpy as np

from generation_config import (
    CORPUS_DIR,
    GENERATED_DIR,
    EMBEDDINGS_DIR,
    REFERENCE_MODEL_KEY,
    STYLE_MODEL_KEYS,
    CONSISTENCY_DIR,
)


def load_generation_mapping(
    author_id: str,
    llm_key: str,
    prompt_variant: str,
    full_run: int
) -> Dict:
    """
    Load the exact mapping of which training docs were used for each generated text.
    
    Returns:
        Dict with structure:
        {
            'prompt_1': {
                'training_texts': [text1, text2, text3],
                'generated_text': generated_text_1
            },
            'prompt_2': {
                'training_texts': [text4, text5, text6],
                'generated_text': generated_text_2
            }
        }
    """
    # Load the generations JSONL file
    # File naming: 
    #   - Complex: generations_fullrun{N}.jsonl
    #   - Simple: generations_simple_fullrun{N}.jsonl
    if prompt_variant == "simple":
        gen_file = GENERATED_DIR / llm_key / f"generations_simple_fullrun{full_run}.jsonl"
    else:  # complex
        gen_file = GENERATED_DIR / llm_key / f"generations_fullrun{full_run}.jsonl"
    
    if not gen_file.exists():
        print(f"[ERROR] Generations file not found: {gen_file}")
        return None
    
    mapping: Dict[str, Dict[str, List[str]]] = {}
    
    with gen_file.open('r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get('author_id') != author_id:
                continue
            
            prompt_idx = entry.get('prompt_index')
            if prompt_idx is None:
                continue
            
            # Extract training texts from this prompt
            training_reviews = entry.get('training_reviews', [])
            training_texts = [
                review.get('text', '')
                for review in training_reviews
                if isinstance(review, dict) and review.get('text')
            ]
            
            # Load the corresponding generated text from normalized directory
            gen_dir = GENERATED_DIR / llm_key / "normalized" / f"texts_{prompt_variant}_fullrun{full_run}" / author_id
            gen_file_pattern = f"*_p{prompt_idx}_*"
            gen_files = list(gen_dir.glob(gen_file_pattern)) if gen_dir.exists() else []
            
            if gen_files:
                generated_text = gen_files[0].read_text(encoding='utf-8')
            else:
                generated_text = None
                print(f"[WARNING] No generated text found for {author_id} prompt {prompt_idx}")
            
            mapping[f'prompt_{prompt_idx}'] = {
                'training_texts': training_texts,
                'generated_text': generated_text,
            }
    
    return mapping if mapping else None


def load_training_texts(author_id: str) -> List[str]:
    """
    Legacy helper: load 6 training docs for an author from the reference model.
    Kept as a fallback / utility, not used in the main pipeline.
    """
    emb_path = EMBEDDINGS_DIR / REFERENCE_MODEL_KEY / f"{author_id}.npz"
    if not emb_path.exists():
        return []
    
    data = np.load(emb_path, allow_pickle=True)
    files = data.get("files")
    if files is None:
        return []
    
    training_files = files[:6]
    texts = []
    for file_rel in training_files:
        file_path = CORPUS_DIR / file_rel
        if file_path.exists():
            texts.append(file_path.read_text(encoding="utf-8"))
    return texts


def load_generated_texts(
    author_id: str,
    llm_key: str,
    prompt_variant: str,
    full_run: int
) -> List[str]:
    """
    Legacy helper: load all generated texts for an author.
    Kept for potential debugging, the main path uses load_generation_mapping().
    """
    gen_dir = GENERATED_DIR / llm_key / "normalized" / f"texts_{prompt_variant}_fullrun{full_run}" / author_id
    if not gen_dir.exists():
        return []
    
    texts = []
    for file_path in sorted(gen_dir.glob("*.txt")):
        texts.append(file_path.read_text(encoding="utf-8"))
    return texts


def get_sentences(text: str) -> List[str]:
    """Split text into reasonably long sentences."""
    import re
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def jaccard_similarity(text1: str, text2: str, n: int = 3) -> float:
    """
    Compute Jaccard similarity between two texts using n-grams.
    
    Args:
        text1, text2: Texts to compare
        n: N-gram size (default 3 for trigrams)
    
    Returns:
        Jaccard coefficient [0.0, 1.0]
    """
    def get_ngrams(text: str, n: int) -> Set[str]:
        """Extract n-grams from text."""
        words = text.lower().split()
        return set(' '.join(words[i:i+n]) for i in range(len(words) - n + 1))
    
    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)
    
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2
    
    return len(intersection) / len(union) if union else 0.0


def analyze_overlap_for_text(
    training_texts: List[str],
    generated_text: str,
    exact_threshold: float = 1.0,
    very_high_threshold: float = 0.85,
    high_threshold: float = 0.7,
    med_threshold: float = 0.5,
    jaccard_threshold: float = 0.3
) -> Dict[str, List]:
    """
    For a SINGLE generated text, compare its sentences against all training sentences.
    
    Uses BOTH SequenceMatcher (for paraphrasing) AND Jaccard (for phrase copying).
    
    For each generated sentence, we:
      - compute similarity to every training sentence
      - take the BEST match
      - classify into one of: exact / 85%+ / 70–84% / 50–69% / <50% (ignored)
      - also compute Jaccard trigram overlap
    
    Returns:
        {
            'exact_sentences': [sent, ...],
            'very_high_pairs': [(gen, train, sim), ...],
            'high_pairs': [...],
            'med_pairs': [...],
            'jaccard_high': [(gen, train, jaccard_score), ...],
            'max_jaccard': float,
            'avg_jaccard': float,
        }
    """
    training_sentences: List[str] = []
    for t in training_texts:
        training_sentences.extend(get_sentences(t))
    
    if not training_sentences:
        return {
            'exact_sentences': [],
            'very_high_pairs': [],
            'high_pairs': [],
            'med_pairs': [],
            'jaccard_high': [],
            'max_jaccard': 0.0,
            'avg_jaccard': 0.0,
        }
    
    gen_sentences = get_sentences(generated_text)
    
    exact_sentences: List[str] = []
    very_high_pairs: List[Tuple[str, str, float]] = []
    high_pairs: List[Tuple[str, str, float]] = []
    med_pairs: List[Tuple[str, str, float]] = []
    jaccard_high: List[Tuple[str, str, float]] = []
    jaccard_scores: List[float] = []
    
    for gen_sent in gen_sentences:
        best_sim = 0.0
        best_train = None
        best_jaccard = 0.0
        best_train_jaccard = None
        
        for train_sent in training_sentences:
            # SequenceMatcher similarity (character-level LCS)
            sim = difflib.SequenceMatcher(
                None, gen_sent.lower(), train_sent.lower()
            ).ratio()
            if sim > best_sim:
                best_sim = sim
                best_train = train_sent
            
            # Jaccard similarity (trigram overlap)
            jac = jaccard_similarity(gen_sent, train_sent, n=3)
            if jac > best_jaccard:
                best_jaccard = jac
                best_train_jaccard = train_sent
        
        jaccard_scores.append(best_jaccard)
        
        # Track high Jaccard scores (phrase copying)
        if best_jaccard >= jaccard_threshold and best_train_jaccard:
            jaccard_high.append((gen_sent, best_train_jaccard, best_jaccard))
        
        if best_train is None:
            continue
        
        # Classify based on best SequenceMatcher similarity
        if best_sim >= exact_threshold:
            exact_sentences.append(gen_sent)
        elif best_sim >= very_high_threshold:
            very_high_pairs.append((gen_sent, best_train, best_sim))
        elif best_sim >= high_threshold:
            high_pairs.append((gen_sent, best_train, best_sim))
        elif best_sim >= med_threshold:
            med_pairs.append((gen_sent, best_train, best_sim))
        # else: < 0.5 similarity → treated as sufficiently original
    
    return {
        'exact_sentences': exact_sentences,
        'very_high_pairs': very_high_pairs,
        'high_pairs': high_pairs,
        'med_pairs': med_pairs,
        'jaccard_high': jaccard_high,
        'max_jaccard': max(jaccard_scores) if jaccard_scores else 0.0,
        'avg_jaccard': sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0,
    }


def check_author(
    author_id: str,
    llm_key: str,
    prompt_variant: str,
    full_run: int
) -> dict:
    """
    Check overlap for a single author using EXACT prompt-generation mapping.
    
    For each prompt:
      - use ONLY its 3 training docs
      - compare ONLY its generated text
    
    Returns aggregate statistics across all prompts for this author.
    """
    mapping = load_generation_mapping(author_id, llm_key, prompt_variant, full_run)
    if not mapping:
        print(f"[WARNING] No generation mapping for author {author_id}")
        return None
    
    all_exact: List[str] = []
    all_very_high: List[Tuple[str, str, float]] = []
    all_high: List[Tuple[str, str, float]] = []
    all_med: List[Tuple[str, str, float]] = []
    all_jaccard_high: List[Tuple[str, str, float]] = []
    all_max_jaccard: List[float] = []
    all_avg_jaccard: List[float] = []
    
    for prompt_key, data in mapping.items():
        training_texts = data.get('training_texts', [])
        generated_text = data.get('generated_text')
        
        if not generated_text or not training_texts:
            continue
        
        res = analyze_overlap_for_text(training_texts, generated_text)
        all_exact.extend(res['exact_sentences'])
        all_very_high.extend(res['very_high_pairs'])
        all_high.extend(res['high_pairs'])
        all_med.extend(res['med_pairs'])
        all_jaccard_high.extend(res['jaccard_high'])
        all_max_jaccard.append(res['max_jaccard'])
        all_avg_jaccard.append(res['avg_jaccard'])
    
    # Aggregate counts (already non-overlapping per generated sentence)
    exact_count = len(all_exact)
    very_high_count = len(all_very_high)
    high_count = len(all_high)
    med_count = len(all_med)
    jaccard_high_count = len(all_jaccard_high)
    max_jaccard_overall = max(all_max_jaccard) if all_max_jaccard else 0.0
    avg_jaccard_overall = sum(all_avg_jaccard) / len(all_avg_jaccard) if all_avg_jaccard else 0.0
    
    return {
        'author_id': author_id,
        'exact_copies': exact_count,
        'very_high': very_high_count,      # 85%+
        'high_similarity': high_count,     # 70-84%
        'med_similarity': med_count,       # 50-69%
        'jaccard_high_count': jaccard_high_count,  # Jaccard > 0.3
        'max_jaccard': max_jaccard_overall,
        'avg_jaccard': avg_jaccard_overall,
        'exact_sentences': all_exact[:3],
        'very_high_pairs': all_very_high[:3],
        'high_sim_pairs': all_high[:3],
        'med_sim_pairs': all_med[:3],
        'jaccard_high_pairs': all_jaccard_high[:3],
        # Approximate counts based on prompt mapping structure
        'num_training': len(mapping) * 3,  # Each prompt nominally has 3 training docs
        'num_generated': len(mapping),     # Number of prompts / generations
    }


def assess_overlap_severity(result: Dict) -> str:
    """
    Assess overlap severity with nuanced, research-appropriate thresholds.
    
    This uses contextual thresholds that consider BOTH exact copies AND
    other metrics to distinguish between:
    - Systematic plagiarism (many exact copies + high overlap)
    - Possible style mimicry (few exact copies, could be author's natural style)
    - Acceptable overlap (stylistic similarity without copying)
    """
    exact = result['exact_copies']
    very_high = result['very_high']
    high_sim = result['high_similarity']
    med_sim = result['med_similarity']
    jac_count = result['jaccard_high_count']
    max_jac = result['max_jaccard']
    
    # CRITICAL: Systematic plagiarism (multiple exact copies + high overlap)
    if exact > 5 and (max_jac > 0.7 or jac_count > 15):
        return "❌ SYSTEMATIC PLAGIARISM"
    
    # SEVERE: Many exact copies (even without high Jaccard)
    if exact > 10:
        return "❌ HEAVY PLAGIARISM"
    
    # SEVERE: Heavy phrase copying with some exact matches
    if exact > 2 and (max_jac > 0.8 or jac_count > 20):
        return "❌ PLAGIARISM + PHRASE COPYING"
    
    # CONCERNING: Few exact copies but pervasive phrase reuse
    if exact > 0 and (max_jac > 0.6 or jac_count > 15):
        return "⚠️  LIKELY PLAGIARISM (REVIEW)"
    
    # BORDERLINE: Few exact copies (could be natural style)
    if exact > 0 and exact <= 3:
        return "⚠️  POSSIBLE STYLE MIMICRY (CHECK BASELINE)"
    
    # CONCERNING: No exact copies but very high phrase overlap
    if max_jac > 0.7 or jac_count > 25:
        return "⚠️  HEAVY PHRASE COPYING"
    
    # MODERATE: High paraphrasing
    if very_high > 10:
        return "⚠️  HEAVY PARAPHRASING"
    
    # MODERATE: Moderate phrase reuse
    if max_jac > 0.5 or jac_count > 15:
        return "⚠️  PHRASE REUSE"
    
    # MINOR: Some copying but not systematic
    if very_high > 5 or high_sim > 15:
        return "⚡ SOME COPYING"
    
    # MINOR: Acceptable overlap
    if max_jac > 0.35 or med_sim > 25:
        return "⚡ MINOR OVERLAP"
    
    # ACCEPTABLE: Original content
    return "✓ MOSTLY ORIGINAL"


def choose_top_authors(
    model_key: str,
    llm_key: str,
    full_run: int,
    check_top_n: int,
    prompt_variant: str
) -> List[str]:
    """
    Choose top-N authors based on mimicry quality metrics.

    Prefers HONEST metrics (dist_to_training_simple/complex) if present,
    falls back to legacy centroid distances otherwise.
    """
    import pandas as pd

    # Updated CSV path to include LLM key
    csv_path = CONSISTENCY_DIR / f"simple_vs_complex_{model_key}_{llm_key}_fullrun{full_run}.csv"
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        print(f"Run: python src/analyse_simple_vs_complex.py --model-key {model_key} --llm-key {llm_key} --full-run {full_run}")
        return []

    df = pd.read_csv(csv_path)

    # Decide which columns to use
    if 'dist_to_training_simple' in df.columns:
        simple_col = 'dist_to_training_simple'
        complex_col = 'dist_to_training_complex'
        print("[INFO] Using HONEST metrics: distance to actual training docs")
    else:
        simple_col = 'dist_real_centroid_simple'
        complex_col = 'dist_real_centroid_complex'
        print("[WARNING] Using LEGACY centroid metrics - rerun analyse_simple_vs_complex.py for HONEST metrics")

    if prompt_variant == "simple":
        sort_col = simple_col
    else:
        sort_col = complex_col

    df_sorted = df.sort_values(sort_col)
    authors = df_sorted.head(check_top_n)['author_id'].tolist()

    print(f"[INFO] Using top {check_top_n} authors from {model_key} model")
    print(f"[INFO] Sorted by: {sort_col}\n")
    return authors


def save_overlap_report(
    results: List[Dict],
    model_key: str,
    llm_key: str,
    prompt_variant: str,
    full_run: int,
    total_exact: int,
    total_very_high: int,
    total_high: int,
    total_med: int,
    total_jaccard_high: int,
    n_authors: int,
    plagiarism_rate: float,
    heavy_copy_rate: float,
    phrase_copy_rate: float,
    avg_max_jaccard: float,
    avg_avg_jaccard: float
):
    """Save detailed overlap analysis report to file."""
    from datetime import datetime
    
    # Create output directory
    output_dir = CONSISTENCY_DIR / "overlap_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"overlap_report_{model_key}_{llm_key}_{prompt_variant}_run{full_run}_{timestamp}.txt"
    output_path = output_dir / filename
    
    with output_path.open('w', encoding='utf-8') as f:
        # Header
        f.write("="*95 + "\n")
        f.write("TEXT OVERLAP ANALYSIS REPORT\n")
        f.write("="*95 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_key}\n")
        f.write(f"LLM: {llm_key}\n")
        f.write(f"Prompt Variant: {prompt_variant}\n")
        f.write(f"Full Run: {full_run}\n")
        f.write(f"Number of Authors: {n_authors}\n\n")
        
        # Summary table
        f.write("="*95 + "\n")
        f.write("AUTHOR-LEVEL RESULTS\n")
        f.write("="*95 + "\n\n")
        f.write(f"{'Author ID':<18} {'Exact':<8} {'85%+':<8} {'70-84%':<8} {'Jac>0.3':<9} {'MaxJac':<8} {'AvgJac':<8} {'Assessment'}\n")
        f.write("-"*95 + "\n")
        
        for r in results:
            assessment = assess_overlap_severity(r)
            # Remove emoji for text file compatibility
            assessment_text = assessment.replace("❌", "").replace("⚠️", "").replace("⚡", "").replace("✓", "").strip()
            
            f.write(
                f"{r['author_id']:<18} {r['exact_copies']:<8} {r['very_high']:<8} "
                f"{r['high_similarity']:<8} {r['jaccard_high_count']:<9} "
                f"{r['max_jaccard']:<8.3f} {r['avg_jaccard']:<8.3f} {assessment_text}\n"
            )
        
        # High Jaccard examples
        f.write("\n" + "="*95 + "\n")
        f.write("EXAMPLES OF HIGH JACCARD OVERLAP (Phrase Copying, Jaccard > 0.3)\n")
        f.write("="*95 + "\n\n")
        
        for r in results:
            if r['jaccard_high_count'] > 0 and r['jaccard_high_pairs']:
                f.write(f"Author: {r['author_id']} (Max Jaccard: {r['max_jaccard']:.3f})\n")
                for i, (gen, train, jac) in enumerate(r['jaccard_high_pairs'], 1):
                    f.write(f"  Match {i} (Jaccard: {jac:.3f}):\n")
                    f.write(f"    Generated: \"{gen}\"\n")
                    f.write(f"    Training:  \"{train}\"\n")
                f.write("\n")
        
        # High paraphrase similarity examples
        f.write("="*95 + "\n")
        f.write("EXAMPLES OF VERY HIGH PARAPHRASE SIMILARITY (85%+, SequenceMatcher)\n")
        f.write("="*95 + "\n\n")
        
        for r in results:
            if r['very_high'] > 0 and r['very_high_pairs']:
                f.write(f"Author: {r['author_id']}\n")
                for i, (gen, train, sim) in enumerate(r['very_high_pairs'], 1):
                    f.write(f"  Match {i} (similarity: {sim:.1%}):\n")
                    f.write(f"    Generated: \"{gen}\"\n")
                    f.write(f"    Training:  \"{train}\"\n")
                f.write("\n")
        
        # Exact copies
        f.write("="*95 + "\n")
        f.write("EXAMPLES OF EXACT COPIED SENTENCES\n")
        f.write("="*95 + "\n\n")
        
        for r in results:
            if r['exact_copies'] > 0 and r['exact_sentences']:
                f.write(f"Author: {r['author_id']}\n")
                for i, sent in enumerate(r['exact_sentences'], 1):
                    f.write(f"  {i}. \"{sent}\"\n")
                f.write("\n")
        
        # Summary statistics
        f.write("="*95 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*95 + "\n\n")
        
        f.write("OVERLAP COUNTS:\n")
        f.write(f"  Total exact copies (100%): {total_exact}\n")
        f.write(f"  Total very high similarity (85%+ paraphrase): {total_very_high}\n")
        f.write(f"  Total high similarity (70-84% paraphrase): {total_high}\n")
        f.write(f"  Total medium similarity (50-69%): {total_med}\n")
        f.write(f"  Total high Jaccard (>0.3 phrase overlap): {total_jaccard_high}\n\n")
        
        f.write("AUTHOR-LEVEL RATES:\n")
        f.write(f"  Plagiarism rate (exact copies): {plagiarism_rate:.1f}% "
               f"({sum(1 for r in results if r['exact_copies'] > 0)}/{n_authors} authors)\n")
        f.write(f"  Heavy paraphrasing rate (85%+): {heavy_copy_rate:.1f}% "
               f"({sum(1 for r in results if r['very_high'] > 5)}/{n_authors} authors)\n")
        f.write(f"  Phrase copying rate (Jaccard>0.35): {phrase_copy_rate:.1f}% "
               f"({sum(1 for r in results if r['max_jaccard'] > 0.35)}/{n_authors} authors)\n\n")
        
        f.write("JACCARD STATISTICS (Trigram Overlap):\n")
        f.write(f"  Average MAX Jaccard per author: {avg_max_jaccard:.3f}\n")
        f.write(f"  Average AVG Jaccard per author: {avg_avg_jaccard:.3f}\n\n")
        
        f.write("ASSESSMENT CRITERIA (NUANCED THRESHOLDS):\n")
        f.write("  SYSTEMATIC PLAGIARISM: >5 exact copies + (MaxJac>0.7 OR Jac>0.3 count>15)\n")
        f.write("  HEAVY PLAGIARISM: >10 exact copies\n")
        f.write("  PLAGIARISM + PHRASE COPYING: >2 exact + (MaxJac>0.8 OR Jac>0.3 count>20)\n")
        f.write("  LIKELY PLAGIARISM: 1+ exact + (MaxJac>0.6 OR Jac>0.3 count>15)\n")
        f.write("  POSSIBLE STYLE MIMICRY: 1-3 exact copies (COMPARE WITH BASELINE)\n")
        f.write("  HEAVY PHRASE COPYING: MaxJac>0.7 OR Jac>0.3 count>25\n")
        f.write("  HEAVY PARAPHRASING: >10 sentences at 85%+ similarity\n")
        f.write("  PHRASE REUSE: MaxJac>0.5 OR Jac>0.3 count>15\n")
        f.write("  SOME COPYING: >5 at 85%+ OR >15 at 70-84%\n")
        f.write("  MINOR OVERLAP: MaxJac>0.35 OR >25 at 50-69%\n")
        f.write("  MOSTLY ORIGINAL: Below all thresholds\n\n")
        
        f.write("INTERPRETATION:\n")
        f.write("  SEQUENCE MATCHER (character-level LCS):\n")
        f.write("    - Exact (100%) = Word-for-word plagiarism (VERY BAD)\n")
        f.write("    - 85%+ = Close paraphrasing (CONCERNING)\n")
        f.write("    - 70-84% = Moderate paraphrasing (QUESTIONABLE)\n")
        f.write("    - 50-69% = Minor phrase reuse (ACCEPTABLE - stylistic overlap)\n\n")
        
        f.write("  JACCARD SIMILARITY (trigram overlap):\n")
        f.write("    - >0.5 = Heavy phrase copying (VERY BAD)\n")
        f.write("    - 0.35-0.5 = Moderate phrase reuse (CONCERNING)\n")
        f.write("    - 0.3-0.35 = Some phrase overlap (MONITOR)\n")
        f.write("    - <0.3 = Acceptable overlap (LIKELY ORIGINAL)\n\n")
        
        f.write("="*95 + "\n")
        f.write(f"Report saved: {output_path}\n")
        f.write("="*95 + "\n")
    
    print(f"\n✅ [SAVED] Detailed overlap report: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Check for text overlap between training and generated")
    parser.add_argument("--author-id", type=str, help="Specific author to check")
    parser.add_argument("--check-top-n", type=int, help="Check top N authors from best mimicry")
    parser.add_argument(
        "--model-key",
        type=str,
        default="style_embedding",
        choices=STYLE_MODEL_KEYS,
        help="Model to use for getting top authors (default: style_embedding)"
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Check overlap for all style models (author selection still based on per-model rankings)"
    )
    parser.add_argument("--llm-key", type=str, default="gpt-5.1")
    parser.add_argument("--prompt-variant", type=str, default="simple", choices=["simple", "complex"])
    parser.add_argument("--full-run", type=int, default=1, choices=[1, 2])
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save detailed overlap report to file"
    )
    
    args = parser.parse_args()
    
    # Determine which models to check (controls how we pick top-N authors)
    models_to_check = STYLE_MODEL_KEYS if args.all_models else [args.model_key]
    
    # Initial author loading (only for single-model mode)
    if args.author_id:
        base_authors = [args.author_id]
    elif args.check_top_n and not args.all_models:
        base_authors = choose_top_authors(args.model_key, args.llm_key, args.full_run, args.check_top_n, args.prompt_variant)
        if not base_authors:
            return
    elif args.check_top_n and args.all_models:
        base_authors = None  # will be computed per model
    else:
        parser.error("Must specify --author-id or --check-top-n")
    
    # Check all models or just one
    for model_key in models_to_check:
        if args.all_models:
            print(f"\n{'='*80}")
            print(f"MODEL: {model_key}")
            print(f"{'='*80}")
        
        # For all-models mode, pick top-N authors per model separately
        if args.check_top_n and (args.all_models or base_authors is None):
            authors = choose_top_authors(model_key, args.llm_key, args.full_run, args.check_top_n, args.prompt_variant)
            if not authors:
                print(f"[SKIP] No authors available for model {model_key}")
                continue
        else:
            authors = base_authors
        
        print(f"{'='*80}")
        print(f"CHECKING TEXT OVERLAP: {args.prompt_variant} prompts (Run {args.full_run})")
        print(f"Authors count: {len(authors)}")
        print(f"{'='*80}\n")
        
        results = []
        for author_id in authors:
            result = check_author(author_id, args.llm_key, args.prompt_variant, args.full_run)
            if result:
                results.append(result)
        
        if not results:
            print(f"[WARNING] No results for model {model_key}")
            continue
        
        # Print summary table with Jaccard scores
        print(f"{'Author ID':<18} {'Exact':<8} {'85%+':<8} {'70-84%':<8} {'Jac>0.3':<9} {'MaxJac':<8} {'AvgJac':<8} {'Assessment'}")
        print(f"{'-'*95}")
        
        for r in results:
            assessment = assess_overlap_severity(r)
            
            print(
                f"{r['author_id']:<18} {r['exact_copies']:<8} {r['very_high']:<8} "
                f"{r['high_similarity']:<8} {r['jaccard_high_count']:<9} "
                f"{r['max_jaccard']:<8.3f} {r['avg_jaccard']:<8.3f} {assessment}"
            )
        
        # Examples of high Jaccard overlap (phrase copying)
        if any(r['jaccard_high_count'] > 0 for r in results):
            print(f"\n{'='*95}")
            print("EXAMPLES OF HIGH JACCARD OVERLAP (Phrase Copying, Jaccard > 0.3):")
            print(f"{'='*95}\n")
            
            for r in results:
                if r['jaccard_high_count'] > 0 and r['jaccard_high_pairs']:
                    print(f"\nAuthor: {r['author_id']} (Max Jaccard: {r['max_jaccard']:.3f})")
                    for i, (gen, train, jac) in enumerate(r['jaccard_high_pairs'], 1):
                        print(f"  Match {i} (Jaccard: {jac:.3f}):")
                        print(f"    Generated: \"{gen[:80]}...\"")
                        print(f"    Training:  \"{train[:80]}...\"")
        
        # Examples of very high similarity matches
        if any(r['very_high'] > 0 for r in results):
            print(f"\n{'='*95}")
            print("EXAMPLES OF VERY HIGH PARAPHRASE SIMILARITY (85%+, SequenceMatcher):")
            print(f"{'='*95}\n")
            
            for r in results:
                if r['very_high'] > 0 and r['very_high_pairs']:
                    print(f"\nAuthor: {r['author_id']}")
                    for i, (gen, train, sim) in enumerate(r['very_high_pairs'], 1):
                        print(f"  Match {i} (similarity: {sim:.1%}):")
                        print(f"    Generated: \"{gen[:80]}...\"")
                        print(f"    Training:  \"{train[:80]}...\"")
        
        # Examples of exact copies
        if any(r['exact_copies'] > 0 for r in results):
            print(f"\n{'='*95}")
            print("EXAMPLES OF EXACT COPIED SENTENCES:")
            print(f"{'='*95}\n")
            
            for r in results:
                if r['exact_copies'] > 0 and r['exact_sentences']:
                    print(f"\nAuthor: {r['author_id']}")
                    for i, sent in enumerate(r['exact_sentences'], 1):
                        print(f"  {i}. \"{sent[:100]}...\"")
        
        # Aggregate statistics
        n_authors = len(results)
        total_exact = sum(r['exact_copies'] for r in results)
        total_very_high = sum(r['very_high'] for r in results)
        total_high = sum(r['high_similarity'] for r in results)
        total_med = sum(r['med_similarity'] for r in results)
        total_jaccard_high = sum(r['jaccard_high_count'] for r in results)
        avg_max_jaccard = sum(r['max_jaccard'] for r in results) / n_authors if n_authors else 0.0
        avg_avg_jaccard = sum(r['avg_jaccard'] for r in results) / n_authors if n_authors else 0.0
        
        plagiarism_rate = (sum(1 for r in results if r['exact_copies'] > 0) / n_authors * 100) if n_authors else 0.0
        heavy_copy_rate = (sum(1 for r in results if r['very_high'] > 5) / n_authors * 100) if n_authors else 0.0
        phrase_copy_rate = (sum(1 for r in results if r['max_jaccard'] > 0.35) / n_authors * 100) if n_authors else 0.0
        
        print(f"\n{'='*95}")
        print("SUMMARY STATISTICS:")
        print(f"{'='*95}")
        print("\n📊 OVERLAP COUNTS:")
        print(f"  Total exact copies (100%): {total_exact}")
        print(f"  Total very high similarity (85%+ paraphrase): {total_very_high}")
        print(f"  Total high similarity (70-84% paraphrase): {total_high}")
        print(f"  Total medium similarity (50-69%): {total_med}")
        print(f"  Total high Jaccard (>0.3 phrase overlap): {total_jaccard_high}")
        
        print(f"\n📈 AUTHOR-LEVEL RATES:")
        print(f"  Plagiarism rate (exact copies): {plagiarism_rate:.1f}% "
              f"({sum(1 for r in results if r['exact_copies'] > 0)}/{n_authors} authors)")
        print(f"  Heavy paraphrasing rate (85%+): {heavy_copy_rate:.1f}% "
              f"({sum(1 for r in results if r['very_high'] > 5)}/{n_authors} authors)")
        print(f"  Phrase copying rate (Jaccard>0.35): {phrase_copy_rate:.1f}% "
              f"({sum(1 for r in results if r['max_jaccard'] > 0.35)}/{n_authors} authors)")
        
        print(f"\n🔢 JACCARD STATISTICS (Trigram Overlap):")
        print(f"  Average MAX Jaccard per author: {avg_max_jaccard:.3f}")
        print(f"  Average AVG Jaccard per author: {avg_avg_jaccard:.3f}")
        
        print(f"\n📖 INTERPRETATION:")
        print("  SEQUENCE MATCHER (character-level LCS):")
        print("    • Exact (100%) = Word-for-word plagiarism ❌ VERY BAD")
        print("    • 85%+ = Close paraphrasing ⚠️  CONCERNING")  
        print("    • 70-84% = Moderate paraphrasing ⚠️  QUESTIONABLE")
        print("    • 50-69% = Minor phrase reuse ✓ ACCEPTABLE (stylistic overlap)")
        
        print(f"\n  JACCARD SIMILARITY (trigram overlap):")
        print("    • >0.5 = Heavy phrase copying ❌ VERY BAD")
        print("    • 0.35-0.5 = Moderate phrase reuse ⚠️  CONCERNING")
        print("    • 0.3-0.35 = Some phrase overlap ⚠️  MONITOR")
        print("    • <0.3 = Acceptable overlap ✓ LIKELY ORIGINAL")
        print(f"{'='*95}\n")
        
        # Save report if requested
        if args.save_report:
            save_overlap_report(
                results, model_key, args.llm_key, args.prompt_variant, args.full_run,
                total_exact, total_very_high, total_high, total_med, total_jaccard_high,
                n_authors, plagiarism_rate, heavy_copy_rate, phrase_copy_rate,
                avg_max_jaccard, avg_avg_jaccard
            )


if __name__ == "__main__":
    main()