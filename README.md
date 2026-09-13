# AmazonHelp AI Customer Support Agent & Evaluation Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end AI Customer Support Agent for **Amazon (`AmazonHelp`)** on Twitter, built for the **Hiver SDE Intern Take-Home Assignment**. 

The system classifies incoming customer support tweets into 8 operational intents, retrieves top-$k$ historical brand resolution context using dense vector embeddings, generates grounded, policy-compliant replies, makes pure LLM human-escalation decisions with explicit rationale, and evaluates performance using an automated evaluation harness with an LLM-as-a-Judge rubric and human agreement checks.

---

## Key Features

- **Pure Live Multi-Provider LLM Engine (`llm_client.py`)**: Built-in support for Groq (`openai/gpt-oss-120b`), Google Gemini, and OpenAI with persistent disk caching (`data/.llm_cache.json`) and exponential backoff retry window scaling to prevent rate limits. Zero hardcoded fallback helpers or dummy strings.
- **RAG Grounded Reply Generation (`task7_reply_generator.py`)**: Dense vector retrieval using `all-MiniLM-L6-v2` over 8,000 historical AmazonHelp conversation pairs to ground replies in actual brand resolution patterns.
- **Pure LLM Escalation Decision Logic (`task8_escalation.py`)**: Evaluates incoming complaints 100% via live LLM prompts for auto-handling vs. human escalation with stated reasons.
- **Comprehensive Evaluation Harness (`task9_eval_harness.py`)**: Computes intent classification accuracy vs. two baselines (Trivial Majority Class & Keyword Heuristics), ROUGE-1 and Cosine Semantic Similarity for replies, 4-axis LLM-as-a-Judge rubric scores, and Cohen's Kappa ($\kappa$) human-LLM agreement.
- **<15 Minute One-Command Reproduction (`run_pipeline.py`)**: Single entry point script to reproduce headline evaluation results end-to-end.

---

## Project Architecture & Directory Structure

```
d:\AmazonHelp\
├── data/
│   ├── cleaned_pairs.csv            # 8,000 cleaned customer_message -> brand_reply pairs
│   ├── history_embeddings.npy       # Dense MiniLM vector embeddings of past customer messages
│   ├── golden_eval_set.csv          # 200 hand-labeled golden evaluation samples
│   ├── golden_eval_notes.md         # Notes on golden set sampling and hand-labeling methodology
│   ├── intent_definitions.md        # Taxonomy of the 8 predefined customer support intents
│   ├── full_pipeline_results.csv    # Combined pipeline outputs across all 200 golden samples
│   └── eval_summary.json            # Final quantitative metrics summary JSON
├── llm_client.py                    # Multi-provider LLM client (Groq, Gemini, OpenAI, Caching, Retry)
├── task5_classifier.py              # Intent classifier (Trivial Baseline, Keyword Baseline, Live LLM)
├── task6_retrieval.py               # SentenceTransformer vector retrieval system
├── task7_reply_generator.py         # Grounded reply generator with retrieved context
├── task8_escalation.py              # Pure LLM escalation decision engine and reason generator
├── task9_eval_harness.py            # Automated metrics, LLM-as-Judge, and Cohen's Kappa agreement
├── run_pipeline.py                  # One-command end-to-end execution pipeline
├── REPORT.md                        # Technical report (Framing, Baselines, Failures, Misleading Numbers)
├── DECISION_LOG.md                  # 14 non-obvious engineering decisions & trade-offs
├── failure_analysis.md              # Detailed analysis of top 5 failure modes with real examples
└── README.md                        # Setup and reproduction guide
```

---

## Quickstart & Reproduction Guide (<15 Minutes)

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn sentence-transformers groq google-genai openai python-dotenv
```

### 3. Configure API Key
Create a `.env` file in the root directory (or export environment variables):
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
# OR
GEMINI_API_KEY=your_gemini_api_key_here
# OR
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Reproduce Headline Results (Single Command)
Run the master pipeline script:
```bash
python run_pipeline.py
```
*Expected Execution Time*: < 20 seconds (with cached responses) / ~1 minute live.

---

## Headline Evaluation Results Summary

| Classifier Model | Accuracy | Macro F1-Score | Status |
| :--- | :---: | :---: | :--- |
| **Trivial Baseline (Majority Class)** | 38.50% | 0.0695 | Baseline 1 |
| **Keyword Baseline (Heuristic Rules)** | 43.00% | 0.3685 | Baseline 2 |
| **LLM Classifier (`llm_client.py`)** | **71.50%** | **0.6599** | Proposed System |

### Reply Quality & Human Agreement

- **LLM-as-a-Judge Overall Quality**: **4.25 / 5.0** (Relevance: 3.86, Grounding: 4.53, Tone: 4.05, Safety: 4.56)
- **Average Reply Cosine Semantic Similarity**: **0.7904**
- **Average Reply ROUGE-1 Recall**: **0.7439**
- **Human-LLM Intent Agreement**: **71.50%** (Cohen's Kappa $\kappa = 0.6305$ — *Substantial Agreement*)
- **Human-LLM Escalation Agreement**: **86.50%** (Cohen's Kappa $\kappa = 0.6782$ — *Substantial Agreement*)

---

## Documentation Links

- [Technical Report (REPORT.md)](file:///d:/AmazonHelp/REPORT.md)
- [Decision Log (DECISION_LOG.md)](file:///d:/AmazonHelp/DECISION_LOG.md)
- [Failure Analysis (failure_analysis.md)](file:///d:/AmazonHelp/failure_analysis.md)
- [Golden Set Notes (data/golden_eval_notes.md)](file:///d:/AmazonHelp/data/golden_eval_notes.md)
