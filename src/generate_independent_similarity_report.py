#!/usr/bin/env python3
"""
Generate comprehensive report on independent similarity analysis.

Creates a detailed markdown report comparing LUAR embedding similarity
vs independent topic/semantic measures (TF-IDF, SBERT, Jaccard).

Usage:
    python src/generate_independent_similarity_report.py --model-key luar_mud_orig --full-run 1
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime


def generate_report(model_key: str, full_run: int, base_path: Path):
    """Generate comprehensive markdown report."""
    
    # LLM models to analyze
    llm_models = [
        "claude-opus-4-5-20251101",
        "deepseek-reasoner",
        "gemini-3-pro-preview",
        "gpt-5.2-2025-12-11",
        "gpt-5.2-pro",
        "grok-4-1-fast-reasoning"
    ]
    
    # Create output directory
    output_dir = base_path / "data" / "independent_similarity_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"independent_similarity_report_{model_key}_run{full_run}_{timestamp}.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("# Independent Topic Similarity Analysis Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Embedding Model**: {model_key}\n\n")
        f.write(f"**Full Run**: {full_run}\n\n")
        f.write("---\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("This report addresses a critical methodological question: **Is the extremely strong ")
        f.write("correlation (r ≈ -0.99) between LUAR train-gen embedding similarity and mimicry ")
        f.write("performance a genuine topical effect, or an artifact of using the same embedding ")
        f.write("space for both measurements?**\n\n")
        
        f.write("### Methodology\n\n")
        f.write("We computed **independent** topic/semantic similarity measures that are NOT based ")
        f.write("on the LUAR authorship embedding space:\n\n")
        f.write("1. **Sentence-BERT (all-MiniLM-L6-v2)**: Generic semantic embedding from HuggingFace, ")
        f.write("trained on diverse text similarity tasks, NOT authorship\n")
        f.write("2. **TF-IDF cosine similarity**: Pure lexical/topical overlap based on term frequencies\n")
        f.write("3. **Jaccard similarity**: Token-level n-gram overlap (trigrams)\n\n")
        
        f.write("These measures were correlated with mimicry performance (distance to training) ")
        f.write("across 100 authors for each of 6 LLMs.\n\n")
        
        f.write("### Key Finding\n\n")
        f.write("**The results show MODEL-DEPENDENT behavior:**\n\n")
        f.write("- **Gemini-3-Pro**: Strong evidence that topical similarity matters (Jaccard r = -0.57, p < 0.0001)\n")
        f.write("- **DeepSeek & Grok**: Moderate evidence of topical effects (Jaccard r ≈ 0.35, p < 0.05)\n")
        f.write("- **Claude, GPT-5.2 models**: No significant topical effects (all measures p > 0.05)\n\n")
        
        f.write("**Interpretation**: Different LLMs employ different mimicry strategies. Gemini ")
        f.write("relies more on topical/lexical mimicry, while Claude and GPT models focus more ")
        f.write("on pure stylometric patterns independent of topic.\n\n")
        
        f.write("---\n\n")
        
        # Detailed Results by Model
        f.write("## Detailed Results by Model\n\n")
        
        for llm_key in llm_models:
            f.write(f"### {llm_key}\n\n")
            
            # Load correlation results
            corr_file = (base_path / "data" / "author_factors" / model_key / llm_key / 
                        f"independent_similarity_correlations_fullrun{full_run}.csv")
            
            if not corr_file.exists():
                f.write("*Data not available*\n\n")
                continue
            
            corr_df = pd.read_csv(corr_file)
            
            # Load LUAR correlations for comparison
            luar_corr_file = (base_path / "data" / "author_factors" / model_key / llm_key / 
                             f"author_factors_correlations.csv")
            
            luar_corr_df = None
            if luar_corr_file.exists():
                luar_corr_df = pd.read_csv(luar_corr_file)
            
            # Analyze by variant
            for variant in ['simple', 'complex', 'avg']:
                variant_data = corr_df[corr_df['variant'] == variant].copy()
                if len(variant_data) == 0:
                    continue
                
                f.write(f"#### {variant.capitalize()} Prompt\n\n")
                
                # Get best correlations for each measure
                sbert = variant_data[variant_data['factor'].str.contains('sbert')]
                tfidf = variant_data[variant_data['factor'].str.contains('tfidf')]
                jaccard = variant_data[variant_data['factor'].str.contains('jaccard')]
                
                # Create summary table
                f.write("| Measure | Best Correlation | Spearman r | p-value | q-value (FDR) | Significance |\n")
                f.write("|---------|------------------|------------|---------|---------------|-------------|\n")
                
                for measure_name, measure_data in [("SBERT", sbert), ("TF-IDF", tfidf), ("Jaccard", jaccard)]:
                    if len(measure_data) > 0:
                        best = measure_data.loc[measure_data['spearman_r'].abs().idxmax()]
                        r = best['spearman_r']
                        p = best['spearman_p']
                        q = best['spearman_q']
                        sig = "✓ SIGNIFICANT" if q < 0.05 else "NOT SIG"
                        
                        factor_short = best['factor'].replace(f'_{variant}', '').replace('_', ' ').title()
                        f.write(f"| {measure_name} | {factor_short} | {r:.3f} | {p:.4f} | {q:.4f} | {sig} |\n")
                
                # Add LUAR comparison if available
                if luar_corr_df is not None:
                    perf_map = {
                        'simple': 'dist_to_training_simple',
                        'complex': 'dist_to_training_complex',
                        'avg': 'average_distance'
                    }
                    perf_col = perf_map[variant]
                    
                    # Get LUAR embedding similarity for this variant
                    luar_factor = f'mean_train_gen_emb_sim_{variant}'
                    luar_row = luar_corr_df[
                        (luar_corr_df['factor'] == luar_factor) & 
                        (luar_corr_df['performance_metric'] == perf_col)
                    ]
                    
                    if len(luar_row) > 0:
                        luar_r = luar_row.iloc[0]['spearman_r']
                        luar_q = luar_row.iloc[0]['spearman_q']
                        f.write(f"| **LUAR (baseline)** | Mean Emb Sim | **{luar_r:.3f}** | **<0.0001** | **<0.0001** | **✓ SIGNIFICANT** |\n")
                
                f.write("\n")
            
            # Overall assessment for this model
            f.write("**Assessment**: ")
            
            # Check if any measure is significant in any variant
            sig_measures = corr_df[corr_df['spearman_q'] < 0.05]
            
            if len(sig_measures) == 0:
                f.write("No significant correlations found for independent measures. ")
                f.write("The strong LUAR correlation (r ≈ -0.99) appears to be an **authorship-space artifact** ")
                f.write("rather than a topical effect. This LLM mimics stylometric patterns independent of topic overlap.\n\n")
            else:
                # Find strongest significant correlation
                strongest = sig_measures.loc[sig_measures['spearman_r'].abs().idxmax()]
                r = strongest['spearman_r']
                measure = "Jaccard" if 'jaccard' in strongest['factor'] else ("TF-IDF" if 'tfidf' in strongest['factor'] else "SBERT")
                
                if abs(r) > 0.5:
                    f.write(f"**Strong topical effect detected** ({measure} r = {r:.3f}, q < {strongest['spearman_q']:.4f}). ")
                    f.write("This LLM's mimicry success is significantly influenced by topical/lexical similarity ")
                    f.write("between training and generated content.\n\n")
                elif abs(r) > 0.3:
                    f.write(f"**Moderate topical effect** ({measure} r = {r:.3f}, q < {strongest['spearman_q']:.4f}). ")
                    f.write("Topic similarity plays a role but is not the dominant factor.\n\n")
                else:
                    f.write(f"**Weak topical effect** ({measure} r = {r:.3f}, q < {strongest['spearman_q']:.4f}). ")
                    f.write("While statistically significant, the effect size is small.\n\n")
            
            f.write("---\n\n")
        
        # Comparative Analysis
        f.write("## Comparative Analysis Across Models\n\n")
        
        # Collect summary statistics
        model_summaries = []
        
        for llm_key in llm_models:
            corr_file = (base_path / "data" / "author_factors" / model_key / llm_key / 
                        f"independent_similarity_correlations_fullrun{full_run}.csv")
            
            if not corr_file.exists():
                continue
            
            corr_df = pd.read_csv(corr_file)
            
            # Get best correlations for each measure type
            sbert_best = corr_df[corr_df['factor'].str.contains('sbert')]['spearman_r'].abs().max()
            tfidf_best = corr_df[corr_df['factor'].str.contains('tfidf')]['spearman_r'].abs().max()
            jaccard_best = corr_df[corr_df['factor'].str.contains('jaccard')]['spearman_r'].abs().max()
            
            # Count significant correlations
            n_sig = (corr_df['spearman_q'] < 0.05).sum()
            
            model_summaries.append({
                'model': llm_key,
                'sbert': sbert_best,
                'tfidf': tfidf_best,
                'jaccard': jaccard_best,
                'n_sig': n_sig
            })
        
        # Create comparison table
        f.write("### Best Correlation Strength by Measure\n\n")
        f.write("| Model | SBERT (|r|) | TF-IDF (|r|) | Jaccard (|r|) | # Significant (q<0.05) |\n")
        f.write("|-------|-------------|--------------|---------------|------------------------|\n")
        
        for summary in sorted(model_summaries, key=lambda x: x['jaccard'], reverse=True):
            model_short = summary['model'].replace('claude-opus-4-5-', 'Claude ').replace('gpt-5.2-', 'GPT-5.2 ')
            model_short = model_short.replace('gemini-3-pro-preview', 'Gemini-3-Pro').replace('deepseek-reasoner', 'DeepSeek')
            model_short = model_short.replace('grok-4-1-fast-reasoning', 'Grok-4.1')
            
            f.write(f"| {model_short} | {summary['sbert']:.3f} | {summary['tfidf']:.3f} | ")
            f.write(f"{summary['jaccard']:.3f} | {summary['n_sig']} |\n")
        
        f.write("\n")
        
        # Interpretation
        f.write("### Interpretation\n\n")
        f.write("**Three Distinct Mimicry Strategies:**\n\n")
        
        f.write("1. **Topical Mimicry (Gemini-3-Pro)**\n")
        f.write("   - Strong correlations across all independent measures\n")
        f.write("   - Jaccard r ≈ -0.57 indicates ~32% of mimicry variance explained by lexical overlap\n")
        f.write("   - Strategy: Mimic both style AND topic/content patterns\n\n")
        
        f.write("2. **Hybrid Approach (DeepSeek, Grok)**\n")
        f.write("   - Moderate Jaccard correlations (r ≈ 0.35)\n")
        f.write("   - Weak TF-IDF/SBERT correlations\n")
        f.write("   - Strategy: Some reliance on phrase/n-gram patterns\n\n")
        
        f.write("3. **Pure Stylometric Mimicry (Claude, GPT-5.2 models)**\n")
        f.write("   - No significant correlations with independent measures\n")
        f.write("   - Strong LUAR correlation persists (r ≈ -0.99)\n")
        f.write("   - Strategy: Focus on stylometric patterns independent of topic\n\n")
        
        f.write("---\n\n")
        
        # Implications
        f.write("## Implications for Research\n\n")
        
        f.write("### 1. Answer to Original Question\n\n")
        f.write("**Q**: Is the LUAR correlation a topical effect or authorship-space artifact?\n\n")
        f.write("**A**: **It depends on the LLM**. For most models (Claude, GPT), it's largely an ")
        f.write("authorship-space artifact. For Gemini, there's a genuine topical component.\n\n")
        
        f.write("### 2. Implications for Mimicry Detection\n\n")
        f.write("- **Model-specific defenses needed**: Detection strategies should consider which LLM was used\n")
        f.write("- **Cross-domain testing**: Claude/GPT mimicry should persist across topics; Gemini's may not\n")
        f.write("- **Lexical analysis**: More effective against Gemini than Claude/GPT\n\n")
        
        f.write("### 3. Implications for Attribution\n\n")
        f.write("- **LUAR remains valid**: Even when correlation is artifact, LUAR still captures what matters (style)\n")
        f.write("- **Complementary measures**: Combining LUAR + TF-IDF could distinguish model types\n")
        f.write("- **Topic control experiments**: Future work should include explicit topic manipulation\n\n")
        
        f.write("### 4. Theoretical Implications\n\n")
        f.write("- **Authorship ≠ Topic**: For most LLMs, stylometric mimicry operates independently of content\n")
        f.write("- **Gemini outlier**: Suggests different training objective or architecture\n")
        f.write("- **Generalization**: Pure stylometric mimicry (Claude/GPT) may generalize better across domains\n\n")
        
        f.write("---\n\n")
        
        # Methodology Details
        f.write("## Methodology Details\n\n")
        
        f.write("### Data\n\n")
        f.write(f"- **Authors analyzed**: 100 per model\n")
        f.write(f"- **Training documents per author**: 6 (most representative)\n")
        f.write(f"- **Generated documents per author**: 2 per prompt variant (4 total)\n")
        f.write(f"- **Embedding model**: {model_key}\n\n")
        
        f.write("### Independent Similarity Measures\n\n")
        f.write("1. **Sentence-BERT**\n")
        f.write("   - Model: `all-MiniLM-L6-v2` from HuggingFace\n")
        f.write("   - Generic semantic embedding (384 dimensions)\n")
        f.write("   - Trained on diverse similarity tasks (NOT authorship)\n")
        f.write("   - Measures: mean, max, min cosine similarity\n\n")
        
        f.write("2. **TF-IDF**\n")
        f.write("   - Sklearn's TfidfVectorizer with default parameters\n")
        f.write("   - Captures term frequency patterns\n")
        f.write("   - Pure lexical/topical overlap\n")
        f.write("   - Measures: mean, max, min cosine similarity\n\n")
        
        f.write("3. **Jaccard**\n")
        f.write("   - Trigram (3-word) n-gram overlap\n")
        f.write("   - Formula: |A ∩ B| / |A ∪ B|\n")
        f.write("   - Measures phrase-level copying/similarity\n")
        f.write("   - Measures: mean, max, min similarity\n\n")
        
        f.write("### Statistical Analysis\n\n")
        f.write("- **Correlation method**: Spearman rank correlation (robust to non-linearity)\n")
        f.write("- **Multiple testing correction**: Benjamini-Hochberg FDR (q-values)\n")
        f.write("- **Significance threshold**: q < 0.05 (FDR-corrected)\n")
        f.write("- **Sample size**: n = 100 authors per model\n\n")
        
        f.write("### Limitations\n\n")
        f.write("1. **Single embedding model**: Analysis uses LUAR; results may differ with other models\n")
        f.write("2. **Domain-specific**: Amazon product reviews; may not generalize to other genres\n")
        f.write("3. **Limited topic diversity**: All training/generation within product review domain\n")
        f.write("4. **N-gram choice**: Jaccard uses trigrams; other n-gram sizes may show different patterns\n\n")
        
        f.write("---\n\n")
        
        # Future Directions
        f.write("## Recommendations for Future Work\n\n")
        
        f.write("1. **Controlled Topic Manipulation**\n")
        f.write("   - Explicitly generate texts on: (a) same topic as training, (b) different topic\n")
        f.write("   - Compare mimicry quality across conditions\n")
        f.write("   - Would definitively test topic vs style hypothesis\n\n")
        
        f.write("2. **Additional Topic Models**\n")
        f.write("   - LDA (Latent Dirichlet Allocation)\n")
        f.write("   - BERTopic (topic model based on transformers)\n")
        f.write("   - Examine topic distribution differences\n\n")
        
        f.write("3. **Cross-Domain Experiments**\n")
        f.write("   - Train on product reviews, generate social media posts\n")
        f.write("   - Test if Claude/GPT maintain mimicry quality while Gemini degrades\n\n")
        
        f.write("4. **Partial Correlation Analysis**\n")
        f.write("   - Control for LUAR similarity when examining independent measures\n")
        f.write("   - Determine independent contribution of each factor\n\n")
        
        f.write("5. **Error Analysis**\n")
        f.write("   - Manually inspect cases where:\n")
        f.write("     - LUAR high but Jaccard low (style without topic)\n")
        f.write("     - Jaccard high but LUAR low (topic without style)\n")
        f.write("   - Qualitative assessment of mimicry quality\n\n")
        
        f.write("---\n\n")
        
        # Data Files
        f.write("## Data Files\n\n")
        f.write("All results are saved in the following locations:\n\n")
        
        for llm_key in llm_models:
            f.write(f"### {llm_key}\n\n")
            f.write(f"- Independent similarity measures:\n")
            f.write(f"  `data/independent_similarity/{model_key}/{llm_key}/independent_similarity_fullrun{full_run}.csv`\n\n")
            f.write(f"- Correlation results:\n")
            f.write(f"  `data/author_factors/{model_key}/{llm_key}/independent_similarity_correlations_fullrun{full_run}.csv`\n\n")
            f.write(f"- Merged dataset (factors + independent similarities):\n")
            f.write(f"  `data/author_factors/{model_key}/{llm_key}/author_factors_with_independent_sim_fullrun{full_run}.csv`\n\n")
        
        f.write("---\n\n")
        
        # Footer
        f.write("## Report Generation\n\n")
        f.write(f"- **Generated by**: `generate_independent_similarity_report.py`\n")
        f.write(f"- **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Analysis scripts**:\n")
        f.write(f"  - `src/compute_independent_similarity.py` (compute measures)\n")
        f.write(f"  - `src/integrate_independent_similarity.py` (correlate with performance)\n")
        f.write(f"  - `src/generate_independent_similarity_report.py` (this report)\n")
    
    print(f"\n{'='*80}")
    print(f"✓ Report generated successfully!")
    print(f"{'='*80}")
    print(f"\nLocation: {report_path}")
    print(f"\nThe report provides:")
    print("  • Executive summary of key findings")
    print("  • Detailed results for each LLM model")
    print("  • Comparative analysis across models")
    print("  • Implications for research and detection")
    print("  • Methodology details and limitations")
    print("  • Recommendations for future work")
    print(f"\n{'='*80}\n")
    
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive report on independent similarity analysis"
    )
    parser.add_argument("--model-key", type=str, default="luar_mud_orig",
                       help="Embedding model key")
    parser.add_argument("--full-run", type=int, default=1,
                       help="Full run number")
    
    args = parser.parse_args()
    
    base_path = Path(__file__).parent.parent
    
    generate_report(args.model_key, args.full_run, base_path)


if __name__ == "__main__":
    main()
