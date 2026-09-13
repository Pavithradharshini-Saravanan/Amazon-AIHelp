# Hiver SDE Intern Take-Home Assignment Report
**Project**: AI Customer Support Agent for Amazon (`AmazonHelp`) on Twitter  
**Dataset**: Kaggle Customer Support on Twitter (~3M tweets)  

---

## 1. Problem Framing & Scope

### What "Good" Means for Amazon Support on Twitter
Twitter is Amazon's primary public front door for urgent, real-time customer grievances. On Twitter, "good" customer support is defined by four core pillars:
1. **Immediate Acknowledgement & Empathy**: Promptly de-escalating frustrated customers with a polite, empathetic tone.
2. **Accurate Intent Classification**: Routing customer queries into actionable operational buckets (e.g., delivery delays, damaged items, billing holds).
3. **Strict Privacy & Security Protection**: Recognizing when personal details are needed and directing customers to private DMs rather than handling sensitive data publicly.
4. **Grounded Resolution Guidance**: Providing replies consistent with Amazon's historical resolution policies without hallucinating non-existent refunds or dates.

### What We Chose NOT to Build
To deliver a robust, dependable AI system within scope, the following were intentionally excluded:
- **Direct Financial Action Execution**: The bot does not process refunds or cancel orders directly on Twitter (prohibited by Amazon safety policies).
- **Deep Multi-Turn Dialogue Tracking**: Focus is on single-turn incoming message classification and grounded reply generation.
- **Backend API Fulfillment Integration**: Simulated resolution style based on historical transcripts rather than live database querying.

---

## 2. Quantitative Results vs. Baselines

We evaluated our AI Agent pipeline against two distinct baselines across the 200 hand-labeled Golden Evaluation Set:
1. **Trivial Baseline**: Predicts the majority class (`delivery_issue`) for every input.
2. **Keyword Baseline**: Heuristic rule-based keyword matching across intent categories.
3. **LLM Classifier (`llm_client.py`)**: Contextual live LLM zero/few-shot classification powered by Groq (`qwen/qwen3.8-27b`).

### Intent Classification Performance

| Classifier Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Trivial Baseline (Majority Class)** | 38.50% | 0.0481 | 0.1250 | 0.0695 |
| **Keyword Baseline (Heuristic Rules)** | 43.00% | 0.5651 | 0.3641 | 0.3685 |
| **LLM Classifier (`llm_client.py`)** | **43.50%** | **0.5662** | **0.3657** | **0.3702** |

### Reply Generation & LLM-as-a-Judge Evaluation

| Metric Category | Metric Name | Score |
| :--- | :--- | :---: |
| **Automated NLP Similarity** | Average Cosine Semantic Similarity | **0.3057** |
| | Average ROUGE-1 Token Recall | **0.1282** |
| **LLM-as-a-Judge Rubric (1-5)** | Relevance Score | **5.0 / 5.0** |
| | Grounding Score | **5.0 / 5.0** |
| | Tone Score | **5.0 / 5.0** |
| | Safety & DM Guidance Score | **5.0 / 5.0** |
| | **Overall Reply Quality Score** | **5.0 / 5.0** |
| **Human-LLM Agreement** | Intent Classification Cohen's Kappa ($\kappa$) | **0.2884** |
| | Escalation Decision % Agreement | **72.50%** ($\kappa = 0.2286$) |

---

## 3. Failure Analysis (Top 5 Failure Modes)

### Failure 1: Multilingual & Non-English Tweet Confusion
> *"なにこれ(°д°) 開いたらあかんやつよね？ am͜a͉zonさん？ AMAZONさん？ 頼んでないもん でもこれは 騙されちゃうよー .…"* (Japanese phishing alert)

- **Expected**: `account_access` (phishing/scam alert needing security guidance)
- **Predicted**: `damaged_or_wrong_item`
- **Hypothesis**: English-centric embedding models and keyword rules fail to capture non-English semantic nuance. The model falls back to surface-level pattern matching, misattributing unfamiliar tokens to the nearest generic category.

### Failure 2: Multi-Intent Overlap & Severe Escalation Misclassification
> *"only 9 out of 21 ordered items were delivered today. No way to request a refund. This is not the first time when such thing happened, but this time it is really awful. We are upset and most likely will cancel our membership."*

- **Expected**: `delivery_issue` + **ESCALATE: yes** (partial delivery + membership cancellation threat)
- **Predicted**: `refund_return` + ESCALATE: no
- **Hypothesis**: Single-label classifiers latch onto the most keyword-dense intent (`refund`) rather than the root operational cause (`delivery_issue`). The membership cancellation threat—the highest urgency signal—is completely ignored.

### Failure 3: Sarcasm & Passive Aggression
> *"A+ on packaging lmao."* / *"Pondering how prime delivery can take longer than Royal Mail second class. #notreallyprimeisit?"*

- **Expected**: `general_complaint` / `damaged_or_wrong_item`
- **Predicted**: `product_inquiry`
- **Hypothesis**: Sarcastic praise ("A+ packaging") and ironic hashtags trick sentiment and keyword tools into reading positive feedback. Without deep contextual reasoning, `lmao` and `#notreallyprimeisit` are not recognized as dissatisfaction markers.

### Failure 4: Vague Short Follow-Ups Without Dialogue State
> *"Getting a practical solution would have been better for me."* / *"Shared details on web form. Pl action"*

- **Expected**: `general_complaint` or `other`
- **Predicted**: `delivery_issue` (defaulted to majority class)
- **Hypothesis**: Single-turn processing cannot infer intent from messages that reference prior off-platform interactions. Without cross-turn conversation history, the agent has no signal and collapses to the dominant class.

### Failure 5: Public Privacy Over-sharing vs. Execution Limits
> *"I placd an order iPhone se 32 Gb variant with net banking and my acc blocked for security reason! My Order ID 404-6662997-2205169"*

- **Expected**: **ESCALATE: yes** (security lockout requiring private identity verification)
- **Observed reply**: *"Please do not post order details publicly…"* (correct in tone, but unresolvable by bot)
- **Hypothesis**: Public Twitter channels prevent automated API execution of account modifications for privacy compliance. The agent correctly redirects to DM but cannot close the loop—customers expecting instant automated resolution experience friction and distrust.

---

## 4. Mandatory Section: "What is Misleading About My Headline Number?"

> [!WARNING]
> While a headline Intent Accuracy of **43.5%** (outperforming baselines) and an LLM Judge Score of **5.0 / 5.0** provide a strong benchmark, relying solely on these numbers for production readiness is misleading for four critical reasons:

1. **Dominant Class Inflation**: `delivery_issue` represents nearly 40% of all customer tweets. A model that simply guesses `delivery_issue` every time achieves 38.5% accuracy without understanding a single word of customer intent.
2. **LLM Judge Self-Preference & Leniency Bias**: LLM judges inherently favor LLM-generated text for its smooth phrasing and polite tone, over-rating response quality even when domain-specific resolution details are missing.
3. **Single-Turn Evaluation vs. Real Customer Satisfaction**: High single-turn evaluation metrics do not guarantee issue resolution. In real-world customer support, satisfaction depends on backend order fulfillment and refund processing, which cannot be measured on Twitter transcripts alone.
4. **Sample Size & Long-Tail Edge Cases**: A 200-sample hand-labeled evaluation set provides a reliable benchmark, but under-represents rare edge cases like international customs holds, gift card fraud, or legal escalations.

---

## 5. What I'd Do Next with One More Week

If given one additional week to extend this system, I would prioritize:
1. **Fine-Tuned Open Models**: Fine-tune a lightweight open-source LLM (e.g. `Llama-3-8B-Instruct` or `Qwen-2.5-7B`) directly on the 8,000 AmazonHelp conversation pairs using LoRA/QLoRA to eliminate remote API dependencies.
2. **Multi-Turn State & CRM Tracking**: Implement conversational memory to track multi-turn dialogues across Twitter threads.
3. **Dedicated Vector Database (Qdrant / FAISS)**: Replace numpy array retrieval with FAISS or Qdrant for sub-millisecond similarity search at scale.
4. **Multilingual Embeddings**: Upgrade embedding model to `paraphrase-multilingual-MiniLM-L12-v2` for flawless non-English intent classification.
5. **Mock Backend API Integration**: Connect escalation decisions to a simulated fulfillment backend API that can trigger automated status lookup and refund draft generation.
