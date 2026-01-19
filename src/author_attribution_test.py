#!/usr/bin/env python3
"""
Author Attribution Test: Can we distinguish between LLMs and human-written text?

Uses Random Forest classifier with SHAP analysis to test if we can reliably
attribute texts to their source (6 LLMs vs human-written).

Usage:
    python src/author_attribution_test.py --model-key luar_mud_orig --full-run 1
    python src/author_attribution_test.py --model-key luar_mud_orig --full-run 1 --n-estimators 200
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, List


def load_all_embeddings_for_attribution(model_key: str, prompt_variant: str, 
                                        full_run: int, base_path: Path) -> Dict:
    """
    Load embeddings from all sources for attribution task.
    
    Returns:
        Dictionary with:
        - X: embeddings array (n_samples, embedding_dim)
        - y: source labels (n_samples,)
        - source_names: list of unique sources
        - author_ids: author ID for each sample
    """
    llm_models = [
        "claude-opus-4-5-20251101",
        "deepseek-reasoner",
        "gemini-3-pro-preview",
        "gpt-5.2-2025-12-11",
        "gpt-5.2-pro",
        "grok-4-1-fast-reasoning"
    ]
    
    # Map to shorter display names
    display_names = {
        "claude-opus-4-5-20251101": "Claude Opus 4.5",
        "deepseek-reasoner": "DeepSeek R1",
        "gemini-3-pro-preview": "Gemini 3 Pro",
        "gpt-5.2-2025-12-11": "GPT-5.2",
        "gpt-5.2-pro": "GPT-5.2 Pro",
        "grok-4-1-fast-reasoning": "Grok 4.1",
        "Human (training)": "Human (training reviews)"
    }
    
    # Determine which prompts to load
    if prompt_variant == "both":
        prompts_to_load = ["simple", "complex"]
    else:
        prompts_to_load = [prompt_variant]
    
    all_embeddings = []
    all_labels = []
    all_author_ids = []
    
    print(f"Loading embeddings for attribution test (prompt: {prompt_variant})...")
    
    # Load LLM generations
    for llm_key in llm_models:
        n_docs = 0
        for prompt in prompts_to_load:
            embeddings_dir = (base_path / "data" / "embeddings" / "generated" / 
                             model_key / llm_key / prompt / f"fullrun{full_run}")
            
            if not embeddings_dir.exists():
                continue
            
            for author_file in sorted(embeddings_dir.glob("*.npz")):
                author_id = author_file.stem
                data = np.load(author_file)
                embeddings = data['embeddings']  # Shape: (2, embedding_dim)
                
                for emb in embeddings:
                    all_embeddings.append(emb)
                    all_labels.append(display_names[llm_key])
                    all_author_ids.append(author_id)
                    n_docs += 1
        
        print(f"  ✓ {display_names[llm_key]}: {n_docs} documents")
    
    # Load human-written training texts (only for authors with generated texts)
    print("\nLoading human-written training texts...")
    human_embeddings_dir = base_path / "data" / "embeddings" / model_key
    
    # Load the CSV with selected indices for each author
    indices_csv_path = base_path / "data" / "consistency" / f"{model_key}_top100.csv"
    author_indices = {}
    if indices_csv_path.exists():
        import ast
        df_indices = pd.read_csv(indices_csv_path)
        for _, row in df_indices.iterrows():
            author_id = row['author_id']
            # Parse the string representation of list
            selected_indices = ast.literal_eval(row['selected_indices'])
            author_indices[author_id] = selected_indices
        print(f"  Loaded selected indices for {len(author_indices)} authors from CSV")
    else:
        print(f"  WARNING: Indices CSV not found at {indices_csv_path}, using first 6 docs")
    
    # Get authors with generated texts
    authors_with_generated = set(all_author_ids)
    
    if human_embeddings_dir.exists():
        n_human = 0
        for author_file in sorted(human_embeddings_dir.glob("*.npz")):
            author_id = author_file.stem
            
            if author_id not in authors_with_generated:
                continue
            
            data = np.load(author_file)
            embeddings = data['embeddings']  # Shape: (6+, embedding_dim)
            
            # Use selected indices from CSV if available, otherwise take first 6
            if author_id in author_indices:
                selected_idx = author_indices[author_id]
                selected_embeddings = embeddings[selected_idx]
            else:
                # Fallback: take first 6
                selected_embeddings = embeddings[:6]
            
            for emb in selected_embeddings:
                all_embeddings.append(emb)
                all_labels.append("Human (training reviews)")
                all_author_ids.append(author_id)
                n_human += 1
        
        print(f"  ✓ Human (training reviews): {n_human} documents")
    
    X = np.array(all_embeddings)
    y = np.array(all_labels)
    author_ids_array = np.array(all_author_ids)
    
    source_names = sorted(set(y))
    unique_authors = sorted(set(author_ids_array))
    
    # Print dataset statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    print(f"{'Source':<25} {'Documents':<12} {'Authors':<10}")
    print("-" * 80)
    for source in source_names:
        mask = y == source
        n_docs = mask.sum()
        n_authors = len(set(author_ids_array[mask]))
        print(f"{source:<25} {n_docs:<12} {n_authors:<10}")
    print("-" * 80)
    print(f"{'TOTAL':<25} {len(X):<12} {len(unique_authors):<10}")
    print("="*80 + "\n")
    
    return {
        'X': X,
        'y': y,
        'source_names': source_names,
        'author_ids': author_ids_array.tolist()
    }


def train_random_forest(X: np.ndarray, y: np.ndarray, n_estimators: int = 100, 
                       random_state: int = 42) -> RandomForestClassifier:
    """Train Random Forest classifier with balanced class weights."""
    print(f"\nTraining Random Forest (n_estimators={n_estimators})...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',  # Adjust for class imbalance (Human has 3x more samples)
        n_jobs=-1
    )
    clf.fit(X, y)
    print("✓ Training complete")
    return clf


def evaluate_classifier(clf: RandomForestClassifier, X: np.ndarray, y: np.ndarray,
                       author_ids: List[str], source_names: List[str], 
                       data_output_dir: Path, plots_output_dir: Path,
                       prompt_variant: str, full_run: int):
    """Evaluate classifier with grouped cross-validation (by author) and generate reports."""
    
    print("\nEvaluating with 5-fold grouped cross-validation (by author)...")
    print("Note: Each fold holds out complete authors, preventing author leakage")
    
    # Convert author_ids to numeric groups for GroupKFold
    unique_authors = sorted(set(author_ids))
    author_to_group = {author: i for i, author in enumerate(unique_authors)}
    groups = np.array([author_to_group[author] for author in author_ids])
    
    print(f"  Total authors: {len(unique_authors)}")
    print(f"  Documents per fold: ~{len(X) // 5}")
    print(f"  Authors per fold: ~{len(unique_authors) // 5}")
    
    # Use GroupKFold to ensure authors don't leak across folds
    gkf = GroupKFold(n_splits=5)
    
    # Cross-validation scores
    cv_scores = cross_val_score(clf, X, y, cv=gkf, groups=groups, n_jobs=-1)
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Get predictions for confusion matrix (also using GroupKFold)
    y_pred = cross_val_predict(clf, X, y, cv=gkf, groups=groups, n_jobs=-1)
    
    # Classification report
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    report = classification_report(y, y_pred, target_names=source_names, digits=4)
    print(report)
    
    # Save report to data directory
    report_path = data_output_dir / f"classification_report_{prompt_variant}_fullrun{full_run}.txt"
    with open(report_path, 'w') as f:
        f.write("Author Attribution Test - Classification Report\n")
        f.write("="*80 + "\n\n")
        f.write(f"Prompt variant: {prompt_variant}\n")
        f.write(f"Evaluation method: 5-fold GroupKFold (by author)\n")
        f.write(f"Total authors: {len(unique_authors)}\n")
        f.write(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
        f.write(report)
    print(f"✓ Saved classification report to: {report_path}")
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred, labels=source_names)
    
    # Plot confusion matrix to plots directory
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Normalize confusion matrix for better visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=source_names, yticklabels=source_names,
                ax=ax, cbar_kws={'label': 'Proportion'})
    
    ax.set_xlabel('Predicted Source', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Source', fontsize=13, fontweight='bold')
    ax.set_title(f'Confusion Matrix: Author Attribution Test ({prompt_variant} prompt)\n' +
                f'Overall Accuracy: {accuracy_score(y, y_pred):.4f} (GroupKFold by author)',
                fontsize=14, fontweight='bold', pad=15)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = plots_output_dir / f"confusion_matrix_{prompt_variant}_fullrun{full_run}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved confusion matrix to: {cm_path}")
    
    return cv_scores, y_pred


def analyze_with_shap(clf: RandomForestClassifier, X: np.ndarray, y: np.ndarray,
                     source_names: List[str], output_dir: Path, prompt_variant: str,
                     full_run: int, max_samples: int = 500):
    """
    Analyze feature importance using SHAP values.
    
    NOTE: SHAP is computed on the model trained on ALL data (descriptive analysis).
    This shows which features the model uses but does not guarantee these features
    generalize to unseen authors (use GroupKFold CV results for generalization claims).
    """
    
    print(f"\nComputing SHAP values (sampling {max_samples} documents)...")
    print("NOTE: SHAP is descriptive (trained on full data), not a generalization guarantee")
    
    # Sample for SHAP (it's computationally expensive)
    if len(X) > max_samples:
        indices = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)
    
    print("✓ SHAP values computed")
    
    # Plot SHAP feature importance (bar plot) for each class
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    # For multi-class, shap_values is a list of arrays (one per class)
    for i, source_name in enumerate(source_names):
        ax = axes[i]
        
        # Get mean absolute SHAP values for this class
        mean_abs_shap = np.abs(shap_values[i]).mean(axis=0)
        
        # Get top 10 features
        top_features = np.argsort(mean_abs_shap)[-10:][::-1]
        top_values = mean_abs_shap[top_features]
        
        # Plot horizontal bar chart
        ax.barh(range(len(top_features)), top_values)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels([f'Dim {idx}' for idx in top_features])
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP value|', fontsize=9)
        ax.set_title(source_name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    # Hide the extra subplot (we have 7 sources but 8 subplots)
    if len(source_names) < len(axes):
        axes[-1].axis('off')
    
    plt.suptitle(f'SHAP Feature Importance by Source ({prompt_variant} prompt)\n' +
                '(Descriptive: trained on full data)',
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    shap_path = output_dir / f"shap_importance_{prompt_variant}_fullrun{full_run}.png"
    plt.savefig(shap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved SHAP analysis to: {shap_path}")


def plot_feature_importance(clf: RandomForestClassifier, output_dir: Path,
                           prompt_variant: str, full_run: int, top_n: int = 20):
    """Plot Random Forest feature importance."""
    
    print(f"\nPlotting top {top_n} feature importances...")
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.barh(range(top_n), importances[indices])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f'Dim {i}' for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Embedding Dimension', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Important Features ({prompt_variant} prompt)',
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    importance_path = output_dir / f"feature_importance_{prompt_variant}_fullrun{full_run}.png"
    plt.savefig(importance_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved feature importance to: {importance_path}")


def generate_summary_report(cv_scores: np.ndarray, source_names: List[str],
                           y_true: np.ndarray, y_pred: np.ndarray,
                           output_dir: Path, prompt_variant: str, full_run: int):
    """Generate a markdown summary of the attribution test."""
    
    summary_path = output_dir / f"attribution_summary_{prompt_variant}_fullrun{full_run}.md"
    
    # Compute per-class accuracies
    per_class_acc = {}
    for source in source_names:
        mask = y_true == source
        if mask.sum() > 0:
            acc = (y_pred[mask] == source).sum() / mask.sum()
            per_class_acc[source] = acc
    
    with open(summary_path, 'w') as f:
        f.write(f"# Author Attribution Test Summary\n\n")
        f.write(f"**Prompt Variant**: {prompt_variant}  \n")
        f.write(f"**Full Run**: {full_run}  \n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}  \n\n")
        
        f.write(f"---\n\n")
        
        f.write(f"## Task\n\n")
        f.write(f"Can we distinguish between text generated by different LLMs vs human-written text?\n\n")
        f.write(f"**Important Note**: The 'Human' class consists of training reviews (natural Amazon product reviews), ")
        f.write(f"while LLM classes are prompted generations (simple + complex prompts). ")
        f.write(f"This measures separability of 'prompted LLM generations vs human training reviews' ")
        f.write(f"in LUAR embedding space, not a direct comparison of LLM vs human under identical task conditions.\n\n")
        f.write(f"**Evaluation Method**: 5-fold GroupKFold cross-validation by author_id (no author leakage between train/test).\n\n")
        
        f.write(f"## Dataset\n\n")
        f.write(f"- **Sources**: {len(source_names)} (6 LLMs + Human)\n")
        f.write(f"- **Total documents**: {len(y_true)}\n")
        f.write(f"- **Documents per source**: \n")
        for source in source_names:
            count = (y_true == source).sum()
            f.write(f"  - {source}: {count}\n")
        f.write(f"\n")
        
        f.write(f"## Results\n\n")
        f.write(f"### Overall Performance\n\n")
        f.write(f"- **Cross-validation accuracy**: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
        f.write(f"- **Baseline (random guessing)**: {1/len(source_names):.4f}\n\n")
        
        f.write(f"### Per-Source Accuracy\n\n")
        f.write(f"| Source | Accuracy | Interpretation |\n")
        f.write(f"|--------|----------|----------------|\n")
        
        # Sort by accuracy
        sorted_sources = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
        for source, acc in sorted_sources:
            if acc > 0.8:
                interp = "Highly distinguishable"
            elif acc > 0.6:
                interp = "Moderately distinguishable"
            elif acc > 0.4:
                interp = "Somewhat distinguishable"
            else:
                interp = "Hard to distinguish"
            
            f.write(f"| {source} | {acc:.4f} | {interp} |\n")
        
        f.write(f"\n")
        
        f.write(f"## Interpretation\n\n")
        
        if cv_scores.mean() > 0.8:
            f.write(f"✅ **Strong Attribution**: The classifier can reliably distinguish between sources. ")
            f.write(f"Different LLMs and human-written text have distinctive stylistic signatures in the embedding space.\n\n")
        elif cv_scores.mean() > 0.6:
            f.write(f"⚠️ **Moderate Attribution**: The classifier can distinguish between sources better than random, ")
            f.write(f"but there is overlap. Some sources are more distinctive than others.\n\n")
        else:
            f.write(f"❌ **Weak Attribution**: The classifier struggles to distinguish between sources. ")
            f.write(f"The embedding space does not capture strong stylistic differences.\n\n")
        
        # Identify most/least distinguishable
        best_source = sorted_sources[0][0]
        worst_source = sorted_sources[-1][0]
        
        f.write(f"- **Most distinguishable**: {best_source} ({sorted_sources[0][1]:.4f} accuracy)\n")
        f.write(f"- **Least distinguishable**: {worst_source} ({sorted_sources[-1][1]:.4f} accuracy)\n\n")
        
        f.write(f"---\n\n")
        f.write(f"*Generated by `author_attribution_test.py`*\n")
    
    print(f"✓ Saved summary report to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Author attribution test: Can we distinguish LLMs from human text?"
    )
    parser.add_argument("--model-key", type=str, default="luar_mud_orig",
                       help="Embedding model key")
    parser.add_argument("--full-run", type=int, default=1,
                       help="Full run number")
    parser.add_argument("--n-estimators", type=int, default=100,
                       help="Number of trees in Random Forest")
    
    args = parser.parse_args()
    
    base_path = Path(__file__).parent.parent
    
    # Always use both prompts combined
    prompt_variant = "both"
    
    print(f"\n{'='*80}")
    print(f"Author Attribution Test")
    print(f"{'='*80}")
    print(f"Model: {args.model_key}")
    print(f"Prompt: {prompt_variant} (simple + complex combined)")
    print(f"Full run: {args.full_run}\n")
    
    # Load data
    data = load_all_embeddings_for_attribution(
        args.model_key, prompt_variant, args.full_run, base_path
    )
    X = data['X']
    y = data['y']
    source_names = data['source_names']
    author_ids = data['author_ids']
    
    # Create output directories
    data_output_dir = base_path / "data" / "author_attribution" / args.model_key / f"fullrun{args.full_run}"
    data_output_dir.mkdir(parents=True, exist_ok=True)
    
    plots_output_dir = base_path / "data" / "plots" / args.model_key / "attribution" / f"fullrun{args.full_run}"
    plots_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train classifier
    clf = train_random_forest(X, y, n_estimators=args.n_estimators)
    
    # Evaluate (saves classification report and confusion matrix)
    # Uses GroupKFold to prevent author leakage across train/test splits
    cv_scores, y_pred = evaluate_classifier(
        clf, X, y, author_ids, source_names, data_output_dir, plots_output_dir, prompt_variant, args.full_run
    )
    
    # Feature importance (plot)
    plot_feature_importance(clf, plots_output_dir, prompt_variant, args.full_run)
    
    # SHAP analysis (plot)
    analyze_with_shap(clf, X, y, source_names, plots_output_dir, prompt_variant, args.full_run)
    
    # Generate summary (data file)
    generate_summary_report(cv_scores, source_names, y, y_pred, 
                           data_output_dir, prompt_variant, args.full_run)
    
    print(f"\n{'='*80}")
    print(f"Attribution test complete!")
    print(f"Data files saved to: {data_output_dir}")
    print(f"Plots saved to: {plots_output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
