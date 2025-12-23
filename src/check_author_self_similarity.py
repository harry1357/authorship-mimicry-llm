#!/usr/bin/env python3
"""
Check Author Self-Similarity (Control Baseline)

This script analyzes overlap WITHIN an author's 6 training documents to establish
a baseline for their natural writing style. This is a critical control to determine
if high overlap in generated texts is due to LLM plagiarism or just the author's
natural tendency to reuse phrases.

Methodology:
- For each author, take their 6 training documents
- Compare documents 1-3 against documents 4-6 (mimics the prompt structure)
- Calculate the same SequenceMatcher and Jaccard metrics
- Compare results with LLM-generated overlap

Usage:
    # Single author baseline
    python src/check_author_self_similarity.py --author-id A132ETQPMHQ585
    
    # Top N authors baseline
    python src/check_author_self_similarity.py --check-top-n 10 --model-key luar_mud_orig --full-run 1
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
    EMBEDDINGS_DIR,
    REFERENCE_MODEL_KEY,
    STYLE_MODEL_KEYS,
    CONSISTENCY_DIR,
)


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


def analyze_self_similarity(
    group1_texts: List[str],
    group2_texts: List[str],
    exact_threshold: float = 1.0,
    very_high_threshold: float = 0.85,
    high_threshold: float = 0.7,
    med_threshold: float = 0.5,
    jaccard_threshold: float = 0.3
) -> Dict[str, any]:
    """
    Compare two groups of texts from the SAME author to establish baseline.
    
    This mimics the LLM generation setup:
    - Group 1 (docs 1-3) compared against Group 2 (docs 4-6)
    - Each sentence in Group 2 is compared against ALL sentences in Group 1
    - Same metrics as LLM overlap analysis
    
    Returns metrics for natural author self-similarity.
    """
    # Get all sentences from both groups
    group1_sentences: List[str] = []
    for t in group1_texts:
        group1_sentences.extend(get_sentences(t))
    
    group2_sentences: List[str] = []
    for t in group2_texts:
        group2_sentences.extend(get_sentences(t))
    
    if not group1_sentences or not group2_sentences:
        return None
    
    exact_sentences: List[str] = []
    very_high_pairs: List[Tuple[str, str, float]] = []
    high_pairs: List[Tuple[str, str, float]] = []
    med_pairs: List[Tuple[str, str, float]] = []
    jaccard_high: List[Tuple[str, str, float]] = []
    jaccard_scores: List[float] = []
    
    # Compare each sentence in Group 2 against all sentences in Group 1
    for sent2 in group2_sentences:
        best_sim = 0.0
        best_sent1 = None
        best_jaccard = 0.0
        best_sent1_jaccard = None
        
        for sent1 in group1_sentences:
            # SequenceMatcher similarity
            sim = difflib.SequenceMatcher(
                None, sent2.lower(), sent1.lower()
            ).ratio()
            if sim > best_sim:
                best_sim = sim
                best_sent1 = sent1
            
            # Jaccard similarity
            jac = jaccard_similarity(sent2, sent1, n=3)
            if jac > best_jaccard:
                best_jaccard = jac
                best_sent1_jaccard = sent1
        
        jaccard_scores.append(best_jaccard)
        
        # Track high Jaccard scores
        if best_jaccard >= jaccard_threshold and best_sent1_jaccard:
            jaccard_high.append((sent2, best_sent1_jaccard, best_jaccard))
        
        if best_sent1 is None:
            continue
        
        # Classify by SequenceMatcher similarity
        if best_sim >= exact_threshold:
            exact_sentences.append(sent2)
        elif best_sim >= very_high_threshold:
            very_high_pairs.append((sent2, best_sent1, best_sim))
        elif best_sim >= high_threshold:
            high_pairs.append((sent2, best_sent1, best_sim))
        elif best_sim >= med_threshold:
            med_pairs.append((sent2, best_sent1, best_sim))
    
    return {
        'exact_sentences': exact_sentences,
        'very_high_pairs': very_high_pairs,
        'high_pairs': high_pairs,
        'med_pairs': med_pairs,
        'jaccard_high': jaccard_high,
        'max_jaccard': max(jaccard_scores) if jaccard_scores else 0.0,
        'avg_jaccard': sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0,
        'num_group1_sents': len(group1_sentences),
        'num_group2_sents': len(group2_sentences),
    }


def load_author_training_docs(author_id: str) -> Tuple[List[str], List[str]]:
    """
    Load 6 training documents for an author, split into two groups.
    
    Returns:
        (group1_texts [docs 1-3], group2_texts [docs 4-6])
    """
    emb_path = EMBEDDINGS_DIR / REFERENCE_MODEL_KEY / f"{author_id}.npz"
    if not emb_path.exists():
        return None, None
    
    data = np.load(emb_path, allow_pickle=True)
    files = data.get("files")
    if files is None or len(files) < 6:
        return None, None
    
    # Get first 6 files
    training_files = files[:6]
    
    # Load texts
    group1_texts = []  # docs 1-3
    group2_texts = []  # docs 4-6
    
    for i, file_rel in enumerate(training_files):
        file_path = CORPUS_DIR / file_rel
        if file_path.exists():
            text = file_path.read_text(encoding="utf-8")
            if i < 3:
                group1_texts.append(text)
            else:
                group2_texts.append(text)
    
    if len(group1_texts) < 3 or len(group2_texts) < 3:
        return None, None
    
    return group1_texts, group2_texts


def check_author_baseline(author_id: str) -> dict:
    """
    Check self-similarity baseline for a single author.
    
    Compares docs 1-3 against docs 4-6 to see how much the author
    naturally reuses phrases in their own writing.
    """
    group1, group2 = load_author_training_docs(author_id)
    
    if group1 is None or group2 is None:
        print(f"[WARNING] Could not load 6 training docs for {author_id}")
        return None
    
    result = analyze_self_similarity(group1, group2)
    
    if result is None:
        return None
    
    return {
        'author_id': author_id,
        'exact_copies': len(result['exact_sentences']),
        'very_high': len(result['very_high_pairs']),
        'high_similarity': len(result['high_pairs']),
        'med_similarity': len(result['med_pairs']),
        'jaccard_high_count': len(result['jaccard_high']),
        'max_jaccard': result['max_jaccard'],
        'avg_jaccard': result['avg_jaccard'],
        'exact_sentences': result['exact_sentences'][:3],
        'very_high_pairs': result['very_high_pairs'][:3],
        'high_sim_pairs': result['high_pairs'][:3],
        'jaccard_high_pairs': result['jaccard_high'][:3],
        'num_docs_group1': 3,
        'num_docs_group2': 3,
        'num_sents_group1': result['num_group1_sents'],
        'num_sents_group2': result['num_group2_sents'],
    }


def assess_overlap_level(r: dict) -> str:
    """
    Determine assessment level based on nuanced, research-appropriate thresholds.
    Same logic as check_text_overlap.py for consistency.
    """
    # Critical: Systematic plagiarism (multiple exact copies + high phrase overlap)
    if r['exact_copies'] > 5 and r['max_jaccard'] > 0.6:
        return "SYSTEMATIC COPYING"
    
    # Severe: Many exact copies OR very high phrase copying
    if r['exact_copies'] > 10:
        return "HEAVY EXACT COPYING"
    if r['max_jaccard'] > 0.7 and r['jaccard_high_count'] > 15:
        return "HEAVY PHRASE COPYING"
    
    # High concern: Multiple exact copies with supporting evidence
    if r['exact_copies'] > 3 and (r['very_high'] > 8 or r['max_jaccard'] > 0.5):
        return "HIGH COPYING"
    
    # Moderate concern: Some exact copies OR significant phrase reuse
    if r['exact_copies'] > 1 and r['max_jaccard'] > 0.4:
        return "MODERATE COPYING"
    if r['max_jaccard'] > 0.6 or r['jaccard_high_count'] > 20:
        return "MODERATE PHRASE REUSE"
    
    # Low concern: Isolated issues
    if r['exact_copies'] > 0:
        return "FEW EXACT MATCHES"
    if r['very_high'] > 10:
        return "HEAVY PARAPHRASING"
    if r['max_jaccard'] > 0.45 or r['jaccard_high_count'] > 10:
        return "SOME PHRASE REUSE"
    
    # Acceptable: Minor stylistic overlap
    if r['high_similarity'] > 15:
        return "MODERATE SIMILARITY"
    if r['med_similarity'] > 30:
        return "MINOR OVERLAP"
    
    return "LOW OVERLAP"


def choose_authors(model_key: str, llm_key: str, full_run: int, check_top_n: int) -> List[str]:
    """Choose top-N authors from consistency CSV (same as overlap check)."""
    import pandas as pd
    
    csv_path = CONSISTENCY_DIR / f"simple_vs_complex_{model_key}_{llm_key}_fullrun{full_run}.csv"
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        print(f"Run: python src/analyse_simple_vs_complex.py --model-key {model_key} --llm-key {llm_key} --full-run {full_run}")
        return []
    
    df = pd.read_csv(csv_path)
    
    if 'dist_to_training_simple' in df.columns:
        sort_col = 'dist_to_training_simple'
    else:
        sort_col = 'dist_real_centroid_simple'
    
    df_sorted = df.sort_values(sort_col)
    authors = df_sorted.head(check_top_n)['author_id'].tolist()
    
    print(f"[INFO] Checking baseline for top {check_top_n} authors from {model_key}")
    return authors


def save_baseline_report(
    results: List[Dict],
    model_key: str,
    llm_key: str,
    full_run: int
):
    """Save baseline self-similarity report."""
    from datetime import datetime
    
    output_dir = CONSISTENCY_DIR / "baseline_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"baseline_self_similarity_{model_key}_{llm_key}_run{full_run}_{timestamp}.txt"
    output_path = output_dir / filename
    
    n_authors = len(results)
    
    # Calculate aggregate stats
    total_exact = sum(r['exact_copies'] for r in results)
    total_very_high = sum(r['very_high'] for r in results)
    total_high = sum(r['high_similarity'] for r in results)
    total_med = sum(r['med_similarity'] for r in results)
    total_jaccard_high = sum(r['jaccard_high_count'] for r in results)
    avg_max_jaccard = sum(r['max_jaccard'] for r in results) / n_authors if n_authors else 0.0
    avg_avg_jaccard = sum(r['avg_jaccard'] for r in results) / n_authors if n_authors else 0.0
    
    with output_path.open('w', encoding='utf-8') as f:
        f.write("="*95 + "\n")
        f.write("AUTHOR SELF-SIMILARITY BASELINE REPORT\n")
        f.write("(Control: Natural phrase reuse within author's own writing)\n")
        f.write("="*95 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_key}\n")
        f.write(f"LLM: {llm_key}\n")
        f.write(f"Full Run: {full_run}\n")
        f.write(f"Number of Authors: {n_authors}\n\n")
        f.write("METHODOLOGY:\n")
        f.write("  - For each author, compare their 6 training documents\n")
        f.write("  - Documents 1-3 (Group 1) vs Documents 4-6 (Group 2)\n")
        f.write("  - Mimics the LLM generation setup\n")
        f.write("  - Establishes baseline for natural phrase reuse\n\n")
        
        # Summary table
        f.write("="*95 + "\n")
        f.write("AUTHOR-LEVEL BASELINE (Natural Self-Similarity)\n")
        f.write("="*95 + "\n\n")
        f.write(f"{'Author ID':<18} {'Exact':<8} {'85%+':<8} {'70-84%':<8} {'Jac>0.3':<9} {'MaxJac':<8} {'AvgJac':<8} {'Assessment'}\n")
        f.write("-"*95 + "\n")
        
        for r in results:
            assessment = assess_overlap_level(r)
            
            f.write(
                f"{r['author_id']:<18} {r['exact_copies']:<8} {r['very_high']:<8} "
                f"{r['high_similarity']:<8} {r['jaccard_high_count']:<9} "
                f"{r['max_jaccard']:<8.3f} {r['avg_jaccard']:<8.3f} {assessment}\n"
            )
        
        # Examples
        has_examples = any(r['jaccard_high_count'] > 0 for r in results)
        if has_examples:
            f.write("\n" + "="*95 + "\n")
            f.write("EXAMPLES OF NATURAL PHRASE REUSE (Within Author's Own Writing)\n")
            f.write("="*95 + "\n\n")
            
            for r in results:
                if r['jaccard_high_count'] > 0 and r['jaccard_high_pairs']:
                    f.write(f"Author: {r['author_id']} (Max Jaccard: {r['max_jaccard']:.3f})\n")
                    for i, (sent2, sent1, jac) in enumerate(r['jaccard_high_pairs'], 1):
                        f.write(f"  Match {i} (Jaccard: {jac:.3f}):\n")
                        f.write(f"    Doc 4-6: \"{sent2}\"\n")
                        f.write(f"    Doc 1-3: \"{sent1}\"\n")
                    f.write("\n")
        
        # Summary statistics
        f.write("="*95 + "\n")
        f.write("SUMMARY STATISTICS (BASELINE)\n")
        f.write("="*95 + "\n\n")
        
        f.write("OVERLAP COUNTS (Natural author self-reuse):\n")
        f.write(f"  Total exact copies: {total_exact}\n")
        f.write(f"  Total very high similarity (85%+): {total_very_high}\n")
        f.write(f"  Total high similarity (70-84%): {total_high}\n")
        f.write(f"  Total medium similarity (50-69%): {total_med}\n")
        f.write(f"  Total high Jaccard (>0.3): {total_jaccard_high}\n\n")
        
        f.write("JACCARD STATISTICS (Natural baseline):\n")
        f.write(f"  Average MAX Jaccard per author: {avg_max_jaccard:.3f}\n")
        f.write(f"  Average AVG Jaccard per author: {avg_avg_jaccard:.3f}\n\n")
        
        f.write("INTERPRETATION:\n")
        f.write("  This baseline shows how much an author naturally reuses phrases\n")
        f.write("  in their own writing (across different documents/topics).\n\n")
        f.write("  COMPARISON WITH LLM RESULTS:\n")
        f.write("  - If LLM overlap >> baseline: LLM is plagiarizing\n")
        f.write("  - If LLM overlap ≈ baseline: LLM captured natural style\n")
        f.write("  - If LLM overlap < baseline: LLM is more original than author!\n\n")
        
        f.write("="*95 + "\n")
    
    print(f"\n✅ [SAVED] Baseline report: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Check author self-similarity baseline (control for natural phrase reuse)"
    )
    parser.add_argument("--author-id", type=str, help="Specific author to check")
    parser.add_argument("--check-top-n", type=int, help="Check top N authors")
    parser.add_argument(
        "--model-key",
        type=str,
        default="luar_mud_orig",
        choices=STYLE_MODEL_KEYS,
        help="Model for author selection"
    )
    parser.add_argument("--llm-key", type=str, help="LLM key (for author selection from CSV)")
    parser.add_argument("--full-run", type=int, default=1, choices=[1, 2])
    parser.add_argument("--save-report", action="store_true", help="Save baseline report")
    
    args = parser.parse_args()
    
    # Determine which authors to check
    if args.author_id:
        authors = [args.author_id]
    elif args.check_top_n:
        if not args.llm_key:
            parser.error("--llm-key required when using --check-top-n")
        authors = choose_authors(args.model_key, args.llm_key, args.full_run, args.check_top_n)
        if not authors:
            return
    else:
        parser.error("Must specify --author-id or --check-top-n")
    
    print(f"="*95)
    print(f"AUTHOR SELF-SIMILARITY BASELINE ANALYSIS")
    print(f"Comparing docs 1-3 vs docs 4-6 for each author")
    print(f"Authors count: {len(authors)}")
    print(f"="*95 + "\n")
    
    results = []
    for author_id in authors:
        result = check_author_baseline(author_id)
        if result:
            results.append(result)
    
    if not results:
        print("[ERROR] No results obtained")
        return
    
    # Print summary table
    print(f"{'Author ID':<18} {'Exact':<8} {'85%+':<8} {'70-84%':<8} {'Jac>0.3':<9} {'MaxJac':<8} {'AvgJac':<8} {'Assessment'}")
    print("-"*95)
    
    for r in results:
        assessment = assess_overlap_level(r)
        
        print(
            f"{r['author_id']:<18} {r['exact_copies']:<8} {r['very_high']:<8} "
            f"{r['high_similarity']:<8} {r['jaccard_high_count']:<9} "
            f"{r['max_jaccard']:<8.3f} {r['avg_jaccard']:<8.3f} {assessment}"
        )
    
    # Summary statistics
    n_authors = len(results)
    avg_max_jaccard = sum(r['max_jaccard'] for r in results) / n_authors
    avg_avg_jaccard = sum(r['avg_jaccard'] for r in results) / n_authors
    
    print(f"\n{'='*95}")
    print("BASELINE SUMMARY:")
    print(f"  Average MAX Jaccard (natural): {avg_max_jaccard:.3f}")
    print(f"  Average AVG Jaccard (natural): {avg_avg_jaccard:.3f}")
    print(f"\n  INTERPRETATION:")
    print(f"  This is how much authors NATURALLY reuse phrases in their own writing.")
    print(f"  Compare these numbers with LLM-generated overlap to assess plagiarism.")
    print(f"{'='*95}\n")
    
    # Save report if requested
    if args.save_report and args.llm_key:
        save_baseline_report(results, args.model_key, args.llm_key, args.full_run)


if __name__ == "__main__":
    main()
