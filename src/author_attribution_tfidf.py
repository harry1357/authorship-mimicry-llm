#!/usr/bin/env python3
"""
Author Attribution with Interpretable Features (TF-IDF / Bag of Words)

Compares three approaches for LLM vs Human attribution:
1. Bag of Words (raw counts) - unweighted word frequencies
2. TF-IDF (weighted) - importance-weighted word frequencies
3. LUAR embeddings (baseline) - neural embeddings

All three use Random Forest + SHAP for feature importance analysis.
The word-based approaches (1 & 2) provide interpretable feature names.

Usage:
    python src/author_attribution_tfidf.py --model-key luar_mud_orig --full-run 1
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict, GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, List, Tuple
import json
import string


def load_raw_texts(model_key: str, full_run: int, base_path: Path, 
                   prompt_variant: str = "both") -> Tuple[List[str], List[str], List[str]]:
    """
    Load raw text data for all LLMs and human texts.
    
    Returns:
        texts: List of text strings
        labels: List of source labels
        author_ids: List of author IDs
    """
    llm_models = [
        "claude-opus-4-5-20251101",
        "deepseek-reasoner",
        "gemini-3-pro-preview",
        "gpt-5.2-2025-12-11",
        "gpt-5.2-pro",
        "grok-4-1-fast-reasoning"
    ]
    
    display_names = {
        "claude-opus-4-5-20251101": "Claude Opus 4.5",
        "deepseek-reasoner": "DeepSeek R1",
        "gemini-3-pro-preview": "Gemini 3 Pro",
        "gpt-5.2-2025-12-11": "GPT-5.2",
        "gpt-5.2-pro": "GPT-5.2 Pro",
        "grok-4-1-fast-reasoning": "Grok 4.1",
    }
    
    texts = []
    labels = []
    author_ids = []
    
    # Determine which prompts to load
    if prompt_variant == "both":
        prompts_to_load = ["simple", "complex"]
    else:
        prompts_to_load = [prompt_variant]
    
    print(f"Loading raw texts (prompt: {prompt_variant})...")
    
    # Load LLM generated texts
    for llm_key in llm_models:
        n_docs = 0
        for prompt in prompts_to_load:
            text_dir = (base_path / "data" / "generated" / llm_key / 
                       "normalized" / f"texts_{prompt}_fullrun{full_run}")
            
            if not text_dir.exists():
                print(f"  WARNING: Directory not found: {text_dir}")
                continue
            
            for author_dir in sorted(text_dir.glob("*")):
                if not author_dir.is_dir():
                    continue
                    
                author_id = author_dir.name
                
                for text_file in sorted(author_dir.glob("*.txt")):
                    try:
                        text = text_file.read_text(encoding='utf-8', errors='ignore').strip()
                        if text:
                            texts.append(text)
                            labels.append(display_names[llm_key])
                            author_ids.append(author_id)
                            n_docs += 1
                    except Exception as e:
                        print(f"  ERROR loading {text_file}: {e}")
        
        print(f"  ✓ {display_names[llm_key]}: {n_docs} documents")
    
    # Load human training texts
    print("\nLoading human training texts...")
    
    # Load selected indices CSV
    indices_csv_path = base_path / "data" / "consistency" / f"{model_key}_top100.csv"
    author_indices = {}
    if indices_csv_path.exists():
        import ast
        df_indices = pd.read_csv(indices_csv_path)
        for _, row in df_indices.iterrows():
            author_id = row['author_id']
            selected_indices = ast.literal_eval(row['selected_indices'])
            author_indices[author_id] = selected_indices
        print(f"  Loaded selected indices for {len(author_indices)} authors")
    
    # Get authors with generated texts
    authors_with_generated = set(author_ids)
    
    # Load human texts from corpus
    corpus_dir = base_path / "amazon_product_data_corpus_mixed_topics_per_author_reformatted"
    n_human = 0
    
    if corpus_dir.exists():
        for author_dir in sorted(corpus_dir.glob("*")):
            if not author_dir.is_dir():
                continue
                
            author_id = author_dir.name
            
            if author_id not in authors_with_generated:
                continue
            
            # Get all text files for this author
            text_files = sorted(author_dir.glob("*.txt"))
            
            # Use selected indices if available
            if author_id in author_indices:
                selected_idx = author_indices[author_id]
                selected_files = [text_files[i] for i in selected_idx if i < len(text_files)]
            else:
                # Fallback: use first 6
                selected_files = text_files[:6]
            
            for text_file in selected_files:
                try:
                    text = text_file.read_text(encoding='utf-8', errors='ignore').strip()
                    if text:
                        texts.append(text)
                        labels.append("Human (training reviews)")
                        author_ids.append(author_id)
                        n_human += 1
                except Exception as e:
                    print(f"  ERROR loading {text_file}: {e}")
    
    print(f"  ✓ Human (training reviews): {n_human} documents")
    
    print(f"\nTotal texts loaded: {len(texts)}")
    
    return texts, labels, author_ids


def strip_punctuation_preprocessor(text: str) -> str:
    """
    Preprocessor that removes punctuation from text EXCEPT apostrophes.
    Applied before tokenization.
    This preserves contractions like "don't", "I've" while removing commas, periods, etc.
    """
    # Remove all punctuation EXCEPT apostrophes
    punctuation_to_remove = string.punctuation.replace("'", "")
    translator = str.maketrans('', '', punctuation_to_remove)
    return text.translate(translator)


def create_bow_features(texts: List[str], max_features: int = 5000, lowercase: bool = True,
                       ngram_range: Tuple[int, int] = (1, 2)) -> Tuple[np.ndarray, List[str]]:
    """
    Create Bag of Words features (raw counts).
    
    Args:
        texts: List of text documents
        max_features: Maximum number of features to keep
        lowercase: Whether to convert text to lowercase (True = case-insensitive)
        ngram_range: Range of n-grams to extract (default: (1,2) for unigrams+bigrams)
    """
    ngram_label = f"{ngram_range[0]}-{ngram_range[1]}grams" if ngram_range[0] != ngram_range[1] else f"{ngram_range[0]}grams"
    print(f"\nCreating Bag of Words features ({ngram_label}, max_features={max_features}, lowercase={lowercase})...")
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,  # ignore terms that appear in fewer than 2 documents
        max_df=0.95,  # ignore terms that appear in more than 95% of documents
        stop_words='english',
        lowercase=lowercase,  # Explicit case sensitivity control
        preprocessor=strip_punctuation_preprocessor,  # Remove punctuation before tokenization
        token_pattern=r'\S+'  # Whitespace tokenizer: splits only on whitespace
    )
    
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Count unigrams vs bigrams
    unigrams = sum(1 for f in feature_names if ' ' not in f)
    bigrams = len(feature_names) - unigrams
    
    print(f"  ✓ Created {X.shape[1]} features ({unigrams} unigrams, {bigrams} bigrams)")
    print(f"  ✓ Feature matrix shape: {X.shape}")
    
    return X.toarray(), feature_names.tolist()


def create_tfidf_features(texts: List[str], max_features: int = 5000, lowercase: bool = True,
                         ngram_range: Tuple[int, int] = (1, 2)) -> Tuple[np.ndarray, List[str]]:
    """
    Create TF-IDF features (weighted).
    
    Args:
        texts: List of text documents
        max_features: Maximum number of features to keep
        lowercase: Whether to convert text to lowercase (True = case-insensitive)
        ngram_range: Range of n-grams to extract (default: (1,2) for unigrams+bigrams)
    """
    ngram_label = f"{ngram_range[0]}-{ngram_range[1]}grams" if ngram_range[0] != ngram_range[1] else f"{ngram_range[0]}grams"
    print(f"\nCreating TF-IDF features ({ngram_label}, max_features={max_features}, lowercase={lowercase})...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        stop_words='english',
        sublinear_tf=True,  # Use log scaling for term frequency
        lowercase=lowercase,  # Explicit case sensitivity control
        preprocessor=strip_punctuation_preprocessor,  # Remove punctuation before tokenization
        token_pattern=r'\S+'  # Whitespace tokenizer: splits only on whitespace
    )
    
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Count unigrams vs bigrams
    unigrams = sum(1 for f in feature_names if ' ' not in f)
    bigrams = len(feature_names) - unigrams
    
    print(f"  ✓ Created {X.shape[1]} features ({unigrams} unigrams, {bigrams} bigrams)")
    print(f"  ✓ Feature matrix shape: {X.shape}")
    
    return X.toarray(), feature_names.tolist()


def load_luar_embeddings(model_key: str, full_run: int, base_path: Path,
                         prompt_variant: str = "both") -> Tuple[np.ndarray, List[str], List[str]]:
    """Load LUAR embeddings (baseline comparison)."""
    # Reuse logic from author_attribution_test.py
    from author_attribution_test import load_all_embeddings_for_attribution
    
    print(f"\nLoading LUAR embeddings for baseline comparison...")
    data = load_all_embeddings_for_attribution(model_key, prompt_variant, full_run, base_path)
    
    return data['X'], data['y'].tolist(), data['author_ids']


def train_and_evaluate(X: np.ndarray, y: np.ndarray, author_ids: List[str],
                      source_names: List[str], feature_names: List[str],
                      method_name: str, output_dir: Path, 
                      n_estimators: int = 100) -> Dict:
    """Train Random Forest and evaluate with GroupKFold."""
    
    print(f"\n{'='*80}")
    print(f"Training and Evaluating: {method_name}")
    print(f"{'='*80}")
    
    # Convert author_ids to numeric groups
    unique_authors = sorted(set(author_ids))
    author_to_group = {author: i for i, author in enumerate(unique_authors)}
    groups = np.array([author_to_group[author] for author in author_ids])
    
    print(f"  Total authors: {len(unique_authors)}")
    print(f"  Total documents: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    
    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        n_jobs=-1
    )
    
    # Cross-validation with GroupKFold
    gkf = GroupKFold(n_splits=5)
    cv_scores = cross_val_score(clf, X, y, cv=gkf, groups=groups, n_jobs=-1)
    y_pred = cross_val_predict(clf, X, y, cv=gkf, groups=groups, n_jobs=-1)
    
    print(f"\n  Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train on full data for feature importance
    clf.fit(X, y)
    
    # Compute per-class accuracies
    per_class_acc = {}
    for source in source_names:
        mask = np.array(y) == source
        if mask.sum() > 0:
            acc = (np.array(y_pred)[mask] == source).sum() / mask.sum()
            per_class_acc[source] = acc
    
    # Get top features by importance
    if feature_names is not None and len(feature_names) == X.shape[1]:
        importances = clf.feature_importances_
        top_indices = np.argsort(importances)[::-1][:20]
        top_features = [(feature_names[i], importances[i]) for i in top_indices]
    else:
        top_features = None
    
    results = {
        'method': method_name,
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'per_class_accuracy': per_class_acc,
        'top_features': top_features,
        'y_true': y,
        'y_pred': y_pred,
        'classifier': clf,
        'feature_names': feature_names
    }
    
    return results


def comprehensive_shap_analysis(clf: RandomForestClassifier, X, y: np.ndarray,
                                feature_names: List[str], source_names: List[str],
                                method_name: str, plots_dir: Path, data_dir: Path, full_run: int,
                                max_samples: int = 500, max_display: int = 20):
    """
    Comprehensive SHAP analysis with multiple visualization types.
    
    Creates:
    1. SHAP bar plots (top features per class)
    2. SHAP beeswarm plots (feature value impact)
    3. SHAP dependence plots (feature interactions)
    4. Per-class summary plots
    """
    print(f"\n[SHAP] Computing SHAP values for {method_name}...")
    print(f"[SHAP] This may take a few minutes for {len(feature_names)} features...")
    
    # Sample data if too large (SHAP is expensive)
    if len(X) > max_samples:
        indices = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        if hasattr(X, 'toarray'):  # sparse matrix
            X_sample = X[indices].toarray()
        else:
            X_sample = X[indices]
        y_sample = y[indices]
    else:
        if hasattr(X, 'toarray'):
            X_sample = X.toarray()
        else:
            X_sample = X
        y_sample = y
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)
    
    print(f"[SHAP] ✓ SHAP values computed for {len(source_names)} classes")
    
    # Handle both old SHAP (list of arrays) and new SHAP (3D array)
    if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        # New SHAP format: (n_samples, n_features, n_classes)
        # Convert to list of arrays: [(n_samples, n_features) for each class]
        shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    
    # === 1. SHAP Bar Plots (Top Features Per Class) ===
    print(f"[SHAP] Creating bar plots...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, source_name in enumerate(source_names):
        ax = axes[i]
        
        # Get mean absolute SHAP values for this class
        mean_abs_shap = np.abs(shap_values[i]).mean(axis=0)
        
        # Get top features
        top_indices = np.argsort(mean_abs_shap)[-max_display:][::-1]
        top_values = mean_abs_shap[top_indices]
        top_feature_names = [feature_names[idx] for idx in top_indices]
        
        # Plot
        y_pos = np.arange(len(top_feature_names))
        ax.barh(y_pos, top_values, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_feature_names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP value|', fontsize=10, fontweight='bold')
        ax.set_title(f'{source_name}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    # Hide extra subplot
    if len(source_names) < len(axes):
        axes[-1].axis('off')
    
    plt.suptitle(f'SHAP Feature Importance by Source ({method_name})\n'
                 f'Top {max_display} Features Per Class',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    bar_path = plots_dir / f"shap_bar_plots_{method_name}_fullrun{full_run}.png"
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SHAP] ✓ Saved bar plots to: {bar_path}")
    
    # === 2. SHAP Beeswarm Plots - SKIPPED (too computationally expensive for 5000 features) ===
    # Note: Beeswarm plots are better suited for smaller feature sets (<100 features)
    print(f"[SHAP] Skipping beeswarm plots (too expensive for {len(feature_names)} features)")
    
    # === 3. Overall Feature Importance (Combined) ===
    print(f"[SHAP] Creating overall feature importance plot...")
    
    # Calculate mean absolute SHAP across all classes
    # shap_values is a list of arrays, one per class
    # Each array should be shape (n_samples, n_features)
    print(f"[SHAP] DEBUG: shap_values type: {type(shap_values)}, length: {len(shap_values)}")
    if len(shap_values) > 0:
        print(f"[SHAP] DEBUG: First class shape: {shap_values[0].shape}")
    
    # Initialize with correct shape
    if isinstance(shap_values, list) and len(shap_values) > 0:
        n_features = shap_values[0].shape[1] if len(shap_values[0].shape) > 1 else len(feature_names)
    else:
        n_features = len(feature_names)
    
    overall_importance = np.zeros(n_features)
    
    for i, shap_class in enumerate(shap_values):
        if len(shap_class.shape) > 1:
            overall_importance += np.abs(shap_class).mean(axis=0)
        else:
            print(f"[SHAP] WARNING: Unexpected shape for class {i}: {shap_class.shape}")
    
    overall_importance /= len(shap_values)
    
    # Get top features
    top_indices = np.argsort(overall_importance)[-30:][::-1]
    top_values = overall_importance[top_indices]
    top_feature_names = [feature_names[idx] for idx in top_indices]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    y_pos = np.arange(len(top_feature_names))
    ax.barh(y_pos, top_values, color='darkblue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_feature_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Absolute SHAP Value (Across All Classes)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top 30 Most Important Features Overall ({method_name})',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    overall_path = plots_dir / f"shap_overall_importance_{method_name}_fullrun{full_run}.png"
    plt.savefig(overall_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SHAP] ✓ Saved overall importance to: {overall_path}")
    
    # === 4. Save Top Features to JSON ===
    print(f"[SHAP] Saving detailed feature rankings...")
    
    detailed_rankings = {}
    for i, source_name in enumerate(source_names):
        mean_abs_shap = np.abs(shap_values[i]).mean(axis=0)
        top_50_indices = np.argsort(mean_abs_shap)[-50:][::-1]
        
        detailed_rankings[source_name] = [
            {
                'rank': rank + 1,
                'feature': feature_names[idx],
                'importance': float(mean_abs_shap[idx])
            }
            for rank, idx in enumerate(top_50_indices)
        ]
    
    json_path = data_dir / f"shap_detailed_rankings_{method_name}_fullrun{full_run}.json"
    with open(json_path, 'w') as f:
        json.dump(detailed_rankings, f, indent=2)
    print(f"[SHAP] ✓ Saved detailed rankings to: {json_path}")
    
    print(f"[SHAP] ✓ Complete SHAP analysis finished for {method_name}")
    
    return {
        'shap_values': shap_values,
        'feature_importance': overall_importance,
        'top_features': top_feature_names[:30]
    }


def plot_comparison(all_results: List[Dict], output_dir: Path, full_run: int):
    """Create comparison plots across all methods."""
    
    print("\nCreating comparison plots...")
    
    # Overall accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = [r['method'] for r in all_results]
    accuracies = [r['cv_accuracy_mean'] for r in all_results]
    errors = [r['cv_accuracy_std'] for r in all_results]
    
    bars = ax.bar(range(len(methods)), accuracies, yerr=errors, capsize=5,
                  color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=0)
    ax.set_ylabel('Cross-Validation Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Attribution Performance: Interpretable vs Neural Features',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=1/7, color='red', linestyle='--', label='Random Baseline', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # Add values on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plot_path = output_dir / f"method_comparison_fullrun{full_run}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved comparison plot: {plot_path}")
    plt.close()


def save_top_words_report(results: Dict, output_dir: Path, full_run: int):
    """Save a report of top discriminative words/features."""
    
    method = results['method']
    top_features = results['top_features']
    
    if top_features is None:
        return
    
    report_path = output_dir / f"top_words_{method.lower().replace(' ', '_')}_fullrun{full_run}.md"
    
    with open(report_path, 'w') as f:
        f.write(f"# Top Discriminative Features: {method}\n\n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}  \n\n")
        f.write(f"## Top 20 Most Important Features\n\n")
        f.write(f"| Rank | Feature | Importance |\n")
        f.write(f"|------|---------|------------|\n")
        
        for i, (feature, importance) in enumerate(top_features, 1):
            f.write(f"| {i} | {feature} | {importance:.6f} |\n")
        
        f.write(f"\n## Per-Source Accuracy\n\n")
        f.write(f"| Source | Accuracy |\n")
        f.write(f"|--------|----------|\n")
        
        for source, acc in sorted(results['per_class_accuracy'].items(), 
                                 key=lambda x: x[1], reverse=True):
            f.write(f"| {source} | {acc:.4f} |\n")
    
    print(f"  ✓ Saved word report: {report_path}")


def create_confusion_matrices(all_results: List[Dict], source_names: List[str],
                              output_dir: Path, full_run: int):
    """Create confusion matrices for all methods."""
    
    print("\nCreating confusion matrices...")
    
    n_methods = len(all_results)
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
    
    if n_methods == 1:
        axes = [axes]
    
    for ax, results in zip(axes, all_results):
        y_true = np.array(results['y_true'])
        y_pred = np.array(results['y_pred'])
        
        cm = confusion_matrix(y_true, y_pred, labels=source_names)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=source_names, yticklabels=source_names,
                   ax=ax, cbar_kws={'label': 'Proportion'})
        
        ax.set_xlabel('Predicted', fontsize=10, fontweight='bold')
        ax.set_ylabel('True', fontsize=10, fontweight='bold')
        ax.set_title(f"{results['method']}\nAcc: {results['cv_accuracy_mean']:.3f}",
                    fontsize=11, fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout()
    cm_path = output_dir / f"confusion_matrices_comparison_fullrun{full_run}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved confusion matrices: {cm_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Author attribution with interpretable features (TF-IDF/BoW)"
    )
    parser.add_argument("--model-key", type=str, default="luar_mud_orig",
                       help="Embedding model key (for LUAR baseline)")
    parser.add_argument("--full-run", type=int, default=1,
                       help="Full run number")
    parser.add_argument("--max-features", type=int, default=5000,
                       help="Maximum number of features for BoW/TF-IDF")
    parser.add_argument("--n-estimators", type=int, default=100,
                       help="Number of trees in Random Forest")
    parser.add_argument("--case-insensitive", action="store_true",
                       help="Use case-insensitive features (default: case-sensitive)")
    parser.add_argument("--skip-luar", action="store_true",
                       help="Skip LUAR baseline (faster)")
    
    args = parser.parse_args()
    
    base_path = Path(__file__).parent.parent
    prompt_variant = "both"
    
    # Determine case sensitivity (default is case-sensitive)
    lowercase = args.case_insensitive
    case_label = "case-insensitive" if args.case_insensitive else "case-sensitive"
    
    print(f"\n{'='*80}")
    print(f"Author Attribution: Interpretable Features Analysis")
    print(f"{'='*80}")
    print(f"Model key: {args.model_key}")
    print(f"Full run: {args.full_run}")
    print(f"Case sensitivity: {case_label}")
    print(f"Tokenization: Whitespace (splits on spaces)")
    print(f"Preprocessing: Punctuation removed (apostrophes kept)")
    print(f"{'='*80}\n")
    print(f"Prompt variant: {prompt_variant}")
    print(f"Max features: {args.max_features}\n")
    
    # Create output directories
    data_output_dir = base_path / "data" / "author_attribution_tfidf" / args.model_key / f"fullrun{args.full_run}"
    data_output_dir.mkdir(parents=True, exist_ok=True)
    
    plots_output_dir = base_path / "data" / "plots" / args.model_key / "attribution" / f"fullrun{args.full_run}"
    plots_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Data output: {data_output_dir}")
    print(f"Plots output: {plots_output_dir}\n")
    
    # Load raw texts
    texts, labels, author_ids = load_raw_texts(args.model_key, args.full_run, base_path, prompt_variant)
    
    source_names = sorted(set(labels))
    labels_array = np.array(labels)
    
    all_results = []
    
    # Define ngram configurations
    ngram_configs = [
        ((1, 1), "unigrams"),
        ((2, 2), "bigrams"),
        ((1, 2), "unigrams_bigrams")
    ]
    
    # Run Bag of Words for each ngram configuration
    for ngram_range, ngram_name in ngram_configs:
        print("\n" + "="*80)
        print(f"BAG OF WORDS: {ngram_name.upper().replace('_', ' + ')}")
        print("="*80)
        
        # Create subdirectories
        bow_data_dir = data_output_dir / "bag_of_words" / ngram_name
        bow_plots_dir = plots_output_dir / "bag_of_words" / ngram_name
        bow_data_dir.mkdir(parents=True, exist_ok=True)
        bow_plots_dir.mkdir(parents=True, exist_ok=True)
        
        X_bow, bow_features = create_bow_features(texts, max_features=args.max_features, 
                                                  lowercase=lowercase, ngram_range=ngram_range)
        bow_results = train_and_evaluate(
            X_bow, labels_array, author_ids, source_names, bow_features,
            f"BoW_{ngram_name}", bow_data_dir, args.n_estimators
        )
        all_results.append(bow_results)
        save_top_words_report(bow_results, bow_data_dir, args.full_run)
        
        # SHAP analysis
        comprehensive_shap_analysis(
            bow_results['classifier'], X_bow, labels_array,
            bow_features, source_names, f"BoW_{ngram_name}",
            bow_plots_dir, bow_data_dir, args.full_run
        )
    
    # Run TF-IDF for each ngram configuration
    for ngram_range, ngram_name in ngram_configs:
        print("\n" + "="*80)
        print(f"TF-IDF: {ngram_name.upper().replace('_', ' + ')}")
        print("="*80)
        
        # Create subdirectories
        tfidf_data_dir = data_output_dir / "tfidf" / ngram_name
        tfidf_plots_dir = plots_output_dir / "tfidf" / ngram_name
        tfidf_data_dir.mkdir(parents=True, exist_ok=True)
        tfidf_plots_dir.mkdir(parents=True, exist_ok=True)
        
        X_tfidf, tfidf_features = create_tfidf_features(texts, max_features=args.max_features,
                                                        lowercase=lowercase, ngram_range=ngram_range)
        tfidf_results = train_and_evaluate(
            X_tfidf, labels_array, author_ids, source_names, tfidf_features,
            f"TF-IDF_{ngram_name}", tfidf_data_dir, args.n_estimators
        )
        all_results.append(tfidf_results)
        save_top_words_report(tfidf_results, tfidf_data_dir, args.full_run)
        
        # SHAP analysis
        comprehensive_shap_analysis(
            tfidf_results['classifier'], X_tfidf, labels_array,
            tfidf_features, source_names, f"TF-IDF_{ngram_name}",
            tfidf_plots_dir, tfidf_data_dir, args.full_run
        )
    
    # Method 3: LUAR embeddings (baseline)
    if not args.skip_luar:
        print("\n" + "="*80)
        print("METHOD 3: LUAR EMBEDDINGS (Baseline)")
        print("="*80)
        X_luar, labels_luar, author_ids_luar = load_luar_embeddings(
            args.model_key, args.full_run, base_path, prompt_variant
        )
        luar_results = train_and_evaluate(
            X_luar, np.array(labels_luar), author_ids_luar, source_names, None,
            "LUAR_Embeddings", data_output_dir, args.n_estimators
        )
        all_results.append(luar_results)
    
    # Create comparison visualizations
    print("\n" + "="*80)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*80)
    plot_comparison(all_results, plots_output_dir, args.full_run)
    create_confusion_matrices(all_results, source_names, plots_output_dir, args.full_run)
    
    # Save summary CSV
    summary_data = []
    for results in all_results:
        summary_data.append({
            'method': results['method'],
            'cv_accuracy_mean': results['cv_accuracy_mean'],
            'cv_accuracy_std': results['cv_accuracy_std'],
            **{f'acc_{source}': results['per_class_accuracy'][source] 
               for source in source_names}
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = data_output_dir / f"summary_comparison_fullrun{args.full_run}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Saved summary CSV: {summary_path}")
    
    print(f"\n{'='*80}")
    print(f"Analysis Complete!")
    print(f"Data saved to: {data_output_dir}")
    print(f"Plots saved to: {plots_output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
