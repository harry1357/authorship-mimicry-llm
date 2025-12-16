# Authorship Mimicry Using Generative AI

## Research Overview

This project investigates the capability of large language models (LLMs) to replicate individual writing styles in the domain of product reviews. The research examines whether contemporary generative AI systems can effectively mimic authorship characteristics when provided with a limited set of training examples from target authors.

### Research Questions

1. To what extent can LLMs generate text that mimics the stylistic characteristics of individual authors?
2. How does prompt complexity and the number of training examples affect mimicry quality?
3. Can automated stylometric analysis successfully distinguish between genuine author texts and LLM-generated mimicry attempts?

## **NEW: Phase 1 - Model Selection (Authorship Verification)**

Before the main mimicry experiment, we conduct a preliminary model selection study:

1. **Generate embeddings** with split-and-average strategy (no truncation)
2. **Run authorship verification** (same-author vs different-author discrimination)
3. **Compare ROC curves** across all models at their optimal configurations
4. **Select best model** for main experiment based on AUC

**Scripts:**
- `src/generate_split_embeddings.py` - Generate embeddings without truncation
- `src/authorship_verification_experiment.py` - SA vs DA experiment with ROC curves

## Project Structure

```
.
├── src/                              # Source code modules
│   ├── build_generation_prompts.py   # Prompt construction pipeline
│   ├── run_generation.py             # Text generation orchestration
│   ├── export_generated_texts.py     # Output formatting utilities
│   ├── llm_client.py                 # LLM API client implementations
│   └── generation_config.py          # Configuration parameters
├── data/                             # Experimental data files
├── amazon_product_data_corpus_.../   # Author-specific review corpora
└── readings/                         # Reference literature

```

## Methodology

### Data Collection

The research utilizes the Amazon Product Review dataset, focusing on authors with substantial review histories across multiple product categories. Each author's corpus is segmented by topic to enable controlled experiments in cross-domain style transfer.

### Experimental Design

**Full Run 1**: Initial generation using baseline prompt configurations
**Full Run 2**: Replication study to assess generation consistency

For each experimental run, the system:
1. Selects high-consistency training reviews based on embedding similarity
2. Constructs prompts with author-specific examples
3. Generates synthetic reviews using various LLM models
4. Exports results for stylometric analysis

### Prompt Variants

- **Simple**: Minimal instruction
- **Complex**: Detailed stylistic guidance

---

## **Quick Start: Phase 1 Model Selection**

### Step 1: Generate Split-Average Embeddings

```bash
# Generate for all models at their optimal max_length
for MODEL in luar_crud_orig luar_mud_orig luar_crud_st luar_mud_st style_embedding star; do
    python src/generate_split_embeddings.py --model-key $MODEL --corpus all
done
```

### Step 2: Run Authorship Verification Experiment

```bash
# Single model
python src/authorship_verification_experiment.py --model-key star --use-split-embeddings --seed 42

# All models
python src/authorship_verification_experiment.py --all-models --use-split-embeddings --seed 42

# With consistency check (5 repeats)
python src/authorship_verification_experiment.py --all-models --use-split-embeddings --n-repeats 5
```

### Step 3: Compare Results

Results saved to `data/plots/{model}/authorship_verification/`:
- ROC curves with AUC scores
- Distance distributions (SA vs DA)
- Statistics (mean, std, Cohen's d)

**Select the model with highest AUC for main experiment!**

---

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- API access for various LLM models
- Sufficient storage for review corpora and embeddings (~5GB)


### Data Format

All intermediate and output files use JSONL (JSON Lines) format for efficient processing and storage. Each record contains:
- Author identifier
- Prompt configuration
- Generated text
- Usage statistics and metadata

## Research Ethics and Limitations

### Ethical Considerations

- Review data is anonymized and used solely for academic research purposes
- Generated content is clearly marked as synthetic to prevent misrepresentation
- The research aims to improve detection of AI-generated deceptive content

### Limitations

- Findings are specific to the product review domain and may not generalize to other writing contexts
- The study focuses on English-language reviews from North American authors
- LLM capabilities continue to evolve, potentially affecting reproducibility

## Project Timeline

- **Data Preparation**: November 2025
- **Prompt Development**: November-December 2025
- **Generation Experiments**: December 2025
- **Analysis and Reporting**: December 2025 - January 2026

## References

Please refer to `research_description_hgupta.docx` for detailed methodology and theoretical framework.



---

*This project is conducted as part of a research internship investigating computational approaches to authorship analysis and detection of AI-generated text.*
