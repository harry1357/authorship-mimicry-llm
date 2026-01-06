# Quick Start: LLM Cheat Sheet

## Setup All LLMs

```bash
# 1. Install packages
pip install openai google-genai anthropic

# 2. Add API keys to .env file
cat >> .env << EOF
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
ANTHROPIC_API_KEY=your_anthropic_key
XAI_API_KEY=your_xai_key
PERPLEXITY_API_KEY=your_perplexity_key
DEEPSEEK_API_KEY=your_deepseek_key
EOF
```

## One-Command Generation (All LLMs)

```bash
# Generate with recommended LLMs (skip Gemini/Sonar Reasoning due to issues)
for LLM in deepseek-reasoner grok-4-1-fast-reasoning sonar-pro gpt-5.2-2025-12-11 claude-opus-4-5-20251101; do
    echo "=== Generating with $LLM ==="
    python src/run_generation.py --llm-key $LLM --full-run 1 --prompt-variant simple
done
```

## Full Pipeline (Single LLM)

```bash
# Replace "grok-4-1-fast-reasoning" with your chosen LLM
LLM="grok-4-1-fast-reasoning"

# 1. Generate
python src/run_generation.py --llm-key $LLM --full-run 1 --prompt-variant simple

# 2. Extract & Normalize
python src/export_generated_texts.py --llm-key $LLM --full-run 1 --prompt-variant simple
python src/normalize_generated_texts.py --llm-key $LLM --full-run 1 --prompt-variant simple

# 3. Embed
python src/embed_generated_texts.py --model-key luar_mud_orig --llm-key $LLM --full-run 1 --prompt-variant simple

# 4. Analyze
python src/analyse_simple_vs_complex.py --model-key luar_mud_orig --llm-key $LLM --full-run 1

# 5. Check overlap
python src/check_text_overlap.py --check-top-n 100 --model-key luar_mud_orig --llm-key $LLM --prompt-variant simple --full-run 1 --save-report

# 6. Baseline
python src/check_author_self_similarity.py --check-top-n 100 --model-key luar_mud_orig --llm-key $LLM --full-run 1 --save-report

# 7. Visualize
python src/plot_true_distances.py --model-key luar_mud_orig --llm-key $LLM --full-run 1 --top-n 10 --rank-by average --grid-view
python src/plot_author_training_vs_generated_all.py --model-key luar_mud_orig --llm-key $LLM --full-run 1 --rank-by average
```

## LLM Keys Quick Reference

| Provider | Full Key | Short Aliases |
|----------|----------|---------------|
| **DeepSeek** | `deepseek-reasoner` | `deepseek-v3.2`, `deepseek` |
| **OpenAI** | `gpt-5.2-2025-12-11` | `gpt-5.2`, `gpt5.2` |
| **Google** | `gemini-3-pro-preview` | `gemini-3-pro`, `gemini3pro` |
| **Anthropic** | `claude-opus-4-5-20251101` | `claude-opus-4-5`, `opus-4-5` |
| **xAI** | `grok-4-1-fast-reasoning` | `grok-4.1-fast`, `grok41-fast` |
| **Perplexity** | `sonar-pro` | `sonarpro`, `sonar_pro` |
| **Perplexity** | `sonar-reasoning-pro` | `sonar-reasoning`, `sonareasoningpro` |


