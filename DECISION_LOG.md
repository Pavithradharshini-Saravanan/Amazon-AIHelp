# Decision Log: Engineering Decisions & Trade-Offs

This decision log documents 14 non-obvious design, architectural, and operational decisions made while building the AmazonHelp AI Support Agent system.

---

1. **Brand Selection (`AmazonHelp`)**:
   - *Decision*: Selected `AmazonHelp` from the 3M Twitter Customer Support dataset over other brands.
   - *Rationale*: Amazon handles high-volume, multi-intent transactional issues (delivery delays, returns, card holds, account locks) rather than purely marketing chatter, providing a rigorous, realistic testbed for customer support AI.

2. **Dataset Sub-Sampling (8,000 Conversation Pairs)**:
   - *Decision*: Sub-sampled 8,000 clean `customer_message -> brand_reply` pairs from the full dataset.
   - *Rationale*: Balances vector retrieval search speed (sub-10ms numpy cosine similarity matrix) without requiring complex vector database setup while maintaining representative coverage of historical resolution styles.

3. **8-Intent Taxonomy Definition**:
   - *Decision*: Defined exactly 8 mutually exclusive intents (`delivery_issue`, `damaged_or_wrong_item`, `refund_return`, `account_access`, `billing_payment`, `general_complaint`, `product_inquiry`, `other`).
   - *Rationale*: Granular enough to capture operational routing while avoiding fine-grained label overlaps that degrade classifier accuracy and human annotator consistency.

4. **Hand-Labeled Golden Evaluation Set (200 Samples)**:
   - *Decision*: Hand-labeled a dedicated 200-sample evaluation set with intents, escalation flags, reasons, and ideal reply guidelines.
   - *Rationale*: Synthetic or weak silver labels introduce noise. Hand-labeling guarantees 100% human ground truth quality for evaluating classifiers and escalation logic.

5. **Embedding Model Selection (`all-MiniLM-L6-v2`)**:
   - *Decision*: Used `all-MiniLM-L6-v2` for dense vector representations.
   - *Rationale*: Extremely lightweight (384 dimensions, ~80MB footprint), runs blazingly fast on CPU, and excels at short sentence semantic similarity.

6. **Top-3 Retrieval Context Window (RAG)**:
   - *Decision*: Set historical context retrieval top-$k = 3$.
   - *Rationale*: Provides strong stylistic and resolution pattern grounding without overwhelming LLM prompt windows or introducing context noise.

7. **Pure LLM Escalation Engine**:
   - *Decision*: Evaluated auto-handling vs. human escalation 100% via live LLM reasoning without heuristic rule shortcuts.
   - *Rationale*: Ensures escalation reasons and decisions are generated dynamically based on deep contextual reasoning of customer sentiment, monetary severity, and support safety.

8. **Persistent LLM Disk Caching (`data/.llm_cache.json`)**:
   - *Decision*: Built prompt-hashed local disk caching across all LLM inference scripts.
   - *Rationale*: Eliminates duplicate API costs, prevents loss of progress during network failures, and enables instant test re-runs.

9. **Multi-Provider LLM Architecture (`llm_client.py`) with Zero Fallbacks**:
   - *Decision*: Abstracted LLM execution into a unified multi-provider client supporting Groq (`qwen/qwen3.8-27b`), Google Gemini, and OpenAI with 8-attempt exponential backoff retry scaling.
   - *Rationale*: Ensures zero hardcoded fallback code exists in the repository. If an API key is missing or calls fail, it raises a real `RuntimeError` rather than returning synthetic fallback text.

10. **Public Twitter Privacy Enforcement in Prompts**:
    - *Decision*: Prompted reply generator to strictly instruct customers to move sensitive order/account details to private Direct Messages (DMs).
    - *Rationale*: Prevents exposure of Personally Identifiable Information (PII) on public Twitter threads while matching Amazon's real-world social media compliance policies.

11. **Evaluation via Cohen's Kappa ($\kappa$)**:
    - *Decision*: Included Cohen's Kappa alongside percentage agreement for human-LLM agreement checks.
    - *Rationale*: Percentage agreement can be inflated by chance agreement on dominant classes; Cohen's Kappa provides statistically rigorous proof of true annotator agreement.

12. **Dual Reply Evaluation Metrics (ROUGE-1 + Cosine Similarity)**:
    - *Decision*: Evaluated generated replies using both lexical n-gram overlap (ROUGE-1) and semantic dense vector similarity.
    - *Rationale*: Lexical overlap measures exact brand terminology adherence, while cosine similarity measures underlying intent alignment regardless of exact word phrasing.

13. **Strict Zero-Hallucination Prompting Constraints**:
    - *Decision*: Instructed reply generator never to invent order numbers, tracking dates, or monetary refund amounts.
    - *Rationale*: In customer support, hallucinating non-existent refund amounts or delivery promises causes severe customer dissatisfaction and operational liability.

14. **One-Command Reproducible Execution (`run_pipeline.py`)**:
    - *Decision*: Encapsulated the entire pipeline (classification, retrieval, generation, escalation, evaluation) into a single executable script.
    - *Rationale*: Fulfills assignment requirements for under 15-minute reproduction of headline results without manual setup steps.
