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
import difflib
from collections import defaultdict

from generation_config import CORPUS_DIR, GENERATED_DIR, EMBEDDINGS_DIR, REFERENCE_MODEL_KEY, STYLE_MODEL_KEYS
import numpy as np


def load_generation_mapping(author_id: str, llm_key: str, prompt_variant: str, full_run: int) -> Dict:
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
    
    mapping = {}
    
    with open(gen_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            
            if entry['author_id'] != author_id:
                continue
            
            prompt_idx = entry['prompt_index']
            
            # Extract training texts from this prompt
            training_texts = [review['text'] for review in entry['training_reviews']]
            
            # Load the corresponding generated text from normalized directory
            gen_dir = GENERATED_DIR / llm_key / "normalized" / f"texts_{prompt_variant}_fullrun{full_run}" / author_id
            gen_file_pattern = f"*_p{prompt_idx}_*"
            gen_files = list(gen_dir.glob(gen_file_pattern))
            
            if gen_files:
                generated_text = gen_files[0].read_text(encoding='utf-8')
            else:
                generated_text = None
                print(f"[WARNING] No generated text found for {author_id} prompt {prompt_idx}")
            
            mapping[f'prompt_{prompt_idx}'] = {
                'training_texts': training_texts,
                'generated_text': generated_text
            }
    
    return mapping if mapping else None


def load_training_texts(author_id: str) -> List[str]:
    """Load the 6 training documents for an author."""
    # Load from embeddings to get the same order
    emb_path = EMBEDDINGS_DIR / REFERENCE_MODEL_KEY / f"{author_id}.npz"
    
    if not emb_path.exists():
        return []
    
    data = np.load(emb_path, allow_pickle=True)
    files = data["files"]
    
    # Get first 6 (fallback if no selected indices)
    training_files = files[:6]
    
    texts = []
    for file_rel in training_files:
        file_path = CORPUS_DIR / file_rel
        if file_path.exists():
            texts.append(file_path.read_text(encoding="utf-8"))
    
    return texts


def load_generated_texts(author_id: str, llm_key: str, prompt_variant: str, full_run: int) -> List[str]:
    """Load generated texts for an author."""
    gen_dir = GENERATED_DIR / llm_key / "normalized" / f"texts_{prompt_variant}_fullrun{full_run}" / author_id
    
    if not gen_dir.exists():
        return []
    
    texts = []
    for file_path in sorted(gen_dir.glob("*.txt")):
        texts.append(file_path.read_text(encoding="utf-8"))
    
    return texts


def get_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitting
    import re
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def find_exact_matches(training_texts: List[str], generated_texts: List[str]) -> Tuple[int, List[str]]:
    """Find sentences that appear verbatim in both training and generated."""
    training_sentences = set()
    for text in training_texts:
        training_sentences.update(get_sentences(text))
    
    copied_sentences = []
    for gen_text in generated_texts:
        gen_sentences = get_sentences(gen_text)
        for sent in gen_sentences:
            if sent in training_sentences:
                copied_sentences.append(sent)
    
    return len(copied_sentences), copied_sentences


def find_similar_matches(training_texts: List[str], generated_texts: List[str], threshold: float = 0.3) -> Tuple[int, List[Tuple[str, str, float]]]:
    """Find sentences that are similar (paraphrased or partially copied)."""
    training_sentences = []
    for text in training_texts:
        training_sentences.extend(get_sentences(text))
    
    similar_pairs = []
    for gen_text in generated_texts:
        gen_sentences = get_sentences(gen_text)
        for gen_sent in gen_sentences:
            for train_sent in training_sentences:
                similarity = difflib.SequenceMatcher(None, gen_sent.lower(), train_sent.lower()).ratio()
                if similarity >= threshold and similarity < 1.0:  # Not exact but similar
                    similar_pairs.append((gen_sent, train_sent, similarity))
    
    return len(similar_pairs), similar_pairs


def check_author(author_id: str, llm_key: str, prompt_variant: str, full_run: int) -> dict:
    """Check overlap for a single author using EXACT prompt-generation mapping."""
    
    # Load the precise mapping from JSONL
    mapping = load_generation_mapping(author_id, llm_key, prompt_variant, full_run)
    
    if not mapping:
        return None
    
    # Check each generated text against its specific training docs
    all_exact = []
    all_very_high = []
    all_high = []
    all_med = []
    
    for prompt_key, data in mapping.items():
        training_texts = data['training_texts']
        generated_text = data['generated_text']
        
        if not generated_text:
            continue
        
        # Check this generated text against ONLY its training docs
        exact_count, exact_sentences = find_exact_matches(training_texts, [generated_text])
        very_high_count, very_high_pairs = find_similar_matches(training_texts, [generated_text], threshold=0.85)
        high_sim_count, high_sim_pairs = find_similar_matches(training_texts, [generated_text], threshold=0.7)
        med_sim_count, med_sim_pairs = find_similar_matches(training_texts, [generated_text], threshold=0.5)
        
        all_exact.extend(exact_sentences)
        all_very_high.extend(very_high_pairs)
        all_high.extend(high_sim_pairs)
        all_med.extend(med_sim_pairs)
    
    # Calculate non-overlapping buckets
    very_high_only = len(all_very_high)  # 85-99%
    high_only = len(all_high) - very_high_only  # 70-84%
    med_only = len(all_med) - len(all_high)  # 50-69%
    
    return {
        'author_id': author_id,
        'exact_copies': len(all_exact),
        'very_high': very_high_only,  # 85%+
        'high_similarity': high_only,  # 70-84%
        'med_similarity': med_only,    # 50-69%
        'exact_sentences': all_exact[:3],
        'very_high_pairs': all_very_high[:3],
        'high_sim_pairs': all_high[:3],
        'med_sim_pairs': all_med[:3],
        'num_training': len(mapping) * 3,  # Each prompt has 3 training docs
        'num_generated': len(mapping),      # Number of prompts
    }


def main():
    parser = argparse.ArgumentParser(description="Check for text overlap between training and generated")
    parser.add_argument("--author-id", type=str, help="Specific author to check")
    parser.add_argument("--check-top-n", type=int, help="Check top N authors from best mimicry")
    parser.add_argument("--model-key", type=str, default="style_embedding", 
                       help="Model to use for getting top authors (default: style_embedding)")
    parser.add_argument("--all-models", action="store_true",
                       help="Check overlap for all models")
    parser.add_argument("--llm-key", type=str, default="gpt-5.1")
    parser.add_argument("--prompt-variant", type=str, default="simple", choices=["simple", "complex"])
    parser.add_argument("--full-run", type=int, default=1, choices=[1, 2])
    
    args = parser.parse_args()
    
    # Determine which models to check
    models_to_check = STYLE_MODEL_KEYS if args.all_models else [args.model_key]
    
    # Initial author loading (only if not checking all models)
    if args.author_id:
        authors = [args.author_id]
    elif args.check_top_n and not args.all_models:
        # Load top authors from analysis (only for single model mode)
        from generation_config import CONSISTENCY_DIR
        import pandas as pd
        
        csv_path = CONSISTENCY_DIR / f"simple_vs_complex_{args.model_key}_fullrun{args.full_run}.csv"
        if not csv_path.exists():
            print(f"ERROR: CSV not found: {csv_path}")
            print(f"Run: python src/analyse_simple_vs_complex.py --model-key {args.model_key} --full-run {args.full_run}")
            return
        
        df = pd.read_csv(csv_path)
        
        # Sort by the prompt variant we're checking
        if args.prompt_variant == "simple":
            df_sorted = df.sort_values('dist_real_centroid_simple')
        else:
            df_sorted = df.sort_values('dist_real_centroid_complex')
        
        authors = df_sorted.head(args.check_top_n)['author_id'].tolist()
        
        print(f"[INFO] Using top {args.check_top_n} authors from {args.model_key} model")
        print(f"[INFO] Sorted by: dist_real_centroid_{args.prompt_variant}")
    elif args.check_top_n and args.all_models:
        # Will load authors inside the loop for each model
        authors = None
    else:
        parser.error("Must specify --author-id or --check-top-n")
    
    # Check all models or just one
    for model_key in models_to_check:
        if args.all_models:
            print(f"\n{'='*80}")
            print(f"MODEL: {model_key}")
            print(f"{'='*80}")
        
        # Load authors for this model (if checking all models)
        if args.check_top_n and (args.all_models or authors is None):
            from generation_config import CONSISTENCY_DIR
            import pandas as pd
            
            csv_path = CONSISTENCY_DIR / f"simple_vs_complex_{model_key}_fullrun{args.full_run}.csv"
            if not csv_path.exists():
                print(f"[SKIP] CSV not found: {csv_path}")
                continue
            
            df = pd.read_csv(csv_path)
            
            if args.prompt_variant == "simple":
                df_sorted = df.sort_values('dist_real_centroid_simple')
            else:
                df_sorted = df.sort_values('dist_real_centroid_complex')
            
            authors = df_sorted.head(args.check_top_n)['author_id'].tolist()
            
            print(f"[INFO] Using top {args.check_top_n} authors from {model_key} model")
            print(f"[INFO] Sorted by: dist_real_centroid_{args.prompt_variant}\n")
        
        print(f"{'='*80}")
        print(f"CHECKING TEXT OVERLAP: {args.prompt_variant} prompts (Run {args.full_run})")
        print(f"{'='*80}\n")
        
        results = []
        for author_id in authors:
            result = check_author(author_id, args.llm_key, args.prompt_variant, args.full_run)
            if result:
                results.append(result)
        
        if not results:
            print(f"[WARNING] No results for {model_key}")
            continue
        
        # Print summary
        print(f"{'Author ID':<18} {'Exact':<8} {'85%+':<8} {'70-84%':<8} {'50-69%':<8} {'Assessment'}")
        print(f"{'-'*80}")
        
        for r in results:
            if r['exact_copies'] > 0:
                assessment = "❌ PLAGIARISM"
            elif r['very_high'] > 5:
                assessment = "⚠️  HEAVY COPYING"
            elif r['high_similarity'] > 10:
                assessment = "⚠️  SOME COPYING"
            elif r['med_similarity'] > 20:
                assessment = "⚡ MINOR OVERLAP"
            else:
                assessment = "✓ Original"
            
            print(f"{r['author_id']:<18} {r['exact_copies']:<8} {r['very_high']:<8} "
                  f"{r['high_similarity']:<8} {r['med_similarity']:<8} {assessment}")
        
        # Print examples of highest similarity matches
        if any(r['very_high'] > 0 for r in results):
            print(f"\n{'='*80}")
            print("EXAMPLES OF VERY HIGH SIMILARITY MATCHES (85%+):")
            print(f"{'='*80}\n")
            
            for r in results:
                if r['very_high'] > 0 and r['very_high_pairs']:
                    print(f"\nAuthor: {r['author_id']}")
                    for i, (gen, train, sim) in enumerate(r['very_high_pairs'], 1):
                        print(f"  Match {i} (similarity: {sim:.1%}):")
                        print(f"    Generated: \"{gen[:80]}...\"")
                        print(f"    Training:  \"{train[:80]}...\"")
        
        # Print examples
        if any(r['exact_copies'] > 0 for r in results):
            print(f"\n{'='*80}")
            print("EXAMPLES OF EXACT COPIED SENTENCES:")
            print(f"{'='*80}\n")
            
            for r in results:
                if r['exact_copies'] > 0 and r['exact_sentences']:
                    print(f"\nAuthor: {r['author_id']}")
                    for i, sent in enumerate(r['exact_sentences'], 1):
                        print(f"  {i}. \"{sent[:100]}...\"")
        
        # Statistics
        total_exact = sum(r['exact_copies'] for r in results)
        total_very_high = sum(r['very_high'] for r in results)
        total_high = sum(r['high_similarity'] for r in results)
        total_med = sum(r['med_similarity'] for r in results)
        plagiarism_rate = sum(1 for r in results if r['exact_copies'] > 0) / len(results) * 100 if results else 0
        heavy_copy_rate = sum(1 for r in results if r['very_high'] > 5) / len(results) * 100 if results else 0
        
        print(f"\n{'='*80}")
        print(f"SUMMARY:")
        print(f"  Total exact copies: {total_exact}")
        print(f"  Total very high similarity (85%+): {total_very_high}")
        print(f"  Total high similarity (70-84%): {total_high}")
        print(f"  Total medium similarity (50-69%): {total_med}")
        print(f"  Plagiarism rate (exact): {plagiarism_rate:.1f}% ({sum(1 for r in results if r['exact_copies'] > 0)}/{len(results)} authors)")
        print(f"  Heavy copying rate (85%+): {heavy_copy_rate:.1f}% ({sum(1 for r in results if r['very_high'] > 5)}/{len(results)} authors)")
        print(f"\n  INTERPRETATION:")
        print(f"  - Exact = Word-for-word plagiarism (VERY BAD)")
        print(f"  - 85%+ = Close paraphrasing (CONCERNING)")  
        print(f"  - 70-84% = Moderate paraphrasing (QUESTIONABLE)")
        print(f"  - 50-69% = Minor phrase reuse (ACCEPTABLE - stylistic overlap)")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
