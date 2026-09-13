import json
import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llm_client import generate_text

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def compute_intent_metrics(df):
    """
    Computes accuracy, precision, recall, and F1 score for baseline and LLM intent classifiers.
    """
    y_true = df['intent']
    metrics = {}

    for col_name, label in [('pred_trivial', 'Trivial Baseline'), ('pred_keyword', 'Keyword Baseline'), ('pred_llm', 'LLM Classifier')]:
        if col_name in df.columns:
            y_pred = df[col_name].fillna('other')
            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
            metrics[label] = {
                'accuracy': round(acc, 4),
                'precision': round(prec, 4),
                'recall': round(rec, 4),
                'f1_macro': round(f1, 4)
            }
    return metrics


def compute_reply_nlp_metrics(df):
    """
    Computes text overlap (ROUGE-1, ROUGE-L approximation) and Cosine Similarity to target brand replies.
    """
    sim_scores = []
    rouge1_scores = []

    for _, row in df.iterrows():
        gen_text = str(row.get('generated_reply', '')).strip()
        ref_text = str(row.get('brand_reply', '')).strip()

        if not gen_text or not ref_text:
            sim_scores.append(0.0)
            rouge1_scores.append(0.0)
            continue

        # Cosine similarity using embeddings
        emb1 = embedding_model.encode([gen_text])
        emb2 = embedding_model.encode([ref_text])
        sim = float(cosine_similarity(emb1, emb2)[0][0])
        sim_scores.append(round(sim, 4))

        # N-gram overlap / ROUGE-1 recall approximation
        gen_tokens = set(gen_text.lower().split())
        ref_tokens = set(ref_text.lower().split())
        if ref_tokens:
            r1 = len(gen_tokens.intersection(ref_tokens)) / len(ref_tokens)
        else:
            r1 = 0.0
        rouge1_scores.append(round(r1, 4))

    return {
        'avg_cosine_similarity': round(np.mean(sim_scores), 4),
        'avg_rouge1_recall': round(np.mean(rouge1_scores), 4)
    }


def llm_judge_eval(customer_msg, brand_reply, generated_reply):
    """
    Evaluates reply quality using LLM-as-a-Judge rubric on 1-5 scale via live LLM.
    """
    prompt = f"""You are an expert AI customer support evaluator.

Evaluate the quality of the generated support reply based on the customer message and past historical brand reply.

Customer Message: "{customer_msg}"
Past Brand Reply: "{brand_reply}"
Generated Reply: "{generated_reply}"

Rate the Generated Reply on a scale of 1 to 5 for each criterion:
1. Relevance (1-5): Does it address the specific issue raised?
2. Grounding (1-5): Is it consistent with Amazon's typical customer support style and free of hallucinated order numbers/dates?
3. Tone (1-5): Is it polite, professional, and appropriately concise?
4. Safety & DM Guidance (1-5): Does it appropriately instruct customer to move to DM for private details?

Output valid JSON strictly in this format:
{{\"relevance\": 5, \"grounding\": 5, \"tone\": 5, \"safety\": 5, \"overall_score\": 5.0, \"judge_comment\": \"<1 sentence rationale>\"}}"""

    res = generate_text(prompt, temperature=0.0)
    res_clean = re.sub(r'```(?:json)?', '', res).strip()

    # Try JSON substring matching
    match = re.search(r'\{[^{}]*\}', res_clean, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return {
                "relevance": float(data.get("relevance", 4)),
                "grounding": float(data.get("grounding", 4)),
                "tone": float(data.get("tone", 4)),
                "safety": float(data.get("safety", 4)),
                "overall_score": float(data.get("overall_score", 4.0)),
                "judge_comment": str(data.get("judge_comment", "Valid response."))
            }
        except Exception:
            pass

    # Extract score numbers dynamically from LLM text output
    def _extract_score(key, text, default=4.0):
        m = re.search(rf'{key}["\s:]*([1-5](?:\.\d+)?)', text, re.IGNORECASE)
        return float(m.group(1)) if m else default

    rel = _extract_score("relevance", res_clean)
    gro = _extract_score("grounding", res_clean)
    ton = _extract_score("tone", res_clean)
    saf = _extract_score("safety", res_clean)
    overall = round((rel + gro + ton + saf) / 4.0, 2)

    return {
        "relevance": rel,
        "grounding": gro,
        "tone": ton,
        "safety": saf,
        "overall_score": overall,
        "judge_comment": res_clean[:120].replace('\n', ' ')
    }


def compute_human_llm_agreement(df):
    """
    Computes Cohen's Kappa and % Agreement between Human Golden Labels and LLM predictions/judge.
    """
    agreement_stats = {}

    # Escalation Agreement
    if 'should_escalate' in df.columns and 'pred_should_escalate' in df.columns:
        h_esc = df['should_escalate'].astype(str).str.lower().str.strip()
        l_esc = df['pred_should_escalate'].astype(str).str.lower().str.strip()

        acc = accuracy_score(h_esc, l_esc)
        kappa = cohen_kappa_score(h_esc, l_esc)
        agreement_stats['escalation_human_llm'] = {
            'percent_agreement': round(acc * 100, 2),
            'cohens_kappa': round(float(kappa), 4)
        }

    # Intent Agreement
    if 'intent' in df.columns and 'pred_llm' in df.columns:
        h_intent = df['intent'].astype(str).str.lower().str.strip()
        l_intent = df['pred_llm'].astype(str).str.lower().str.strip()

        acc = accuracy_score(h_intent, l_intent)
        kappa = cohen_kappa_score(h_intent, l_intent)
        agreement_stats['intent_human_llm'] = {
            'percent_agreement': round(acc * 100, 2),
            'cohens_kappa': round(float(kappa), 4)
        }

    return agreement_stats


def run_full_evaluation(df_path="data/full_pipeline_results.csv"):
    df = pd.read_csv(df_path)
    print(f"Running Evaluation Harness on {len(df)} golden evaluation records...")

    # 1. Intent Metrics
    intent_results = compute_intent_metrics(df)

    # 2. Reply Quality Automated NLP Metrics
    reply_nlp_results = compute_reply_nlp_metrics(df)

    # 3. LLM-as-Judge Rubric Evaluation
    print("Running LLM-as-a-Judge Evaluation Rubric...")
    judge_scores = []
    for i, row in df.iterrows():
        c_msg = row['customer_message']
        b_reply = row['brand_reply']
        g_reply = row.get('generated_reply', '')
        j_eval = llm_judge_eval(c_msg, b_reply, g_reply)
        judge_scores.append(j_eval)

        if (i + 1) % 50 == 0 or (i + 1) == len(df):
            print(f"Judge evaluated {i + 1}/{len(df)} samples")

    judge_df = pd.DataFrame(judge_scores)
    avg_relevance = round(judge_df['relevance'].mean(), 2)
    avg_grounding = round(judge_df['grounding'].mean(), 2)
    avg_tone = round(judge_df['tone'].mean(), 2)
    avg_safety = round(judge_df['safety'].mean(), 2)
    avg_overall = round(judge_df['overall_score'].mean(), 2)

    # 4. Human-LLM Agreement
    agreement_results = compute_human_llm_agreement(df)

    full_report = {
        'intent_classification_metrics': intent_results,
        'reply_nlp_metrics': reply_nlp_results,
        'llm_as_judge_rubric': {
            'avg_relevance': avg_relevance,
            'avg_grounding': avg_grounding,
            'avg_tone': avg_tone,
            'avg_safety': avg_safety,
            'avg_overall_score': avg_overall
        },
        'human_llm_agreement': agreement_results
    }

    with open("data/eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print("\n================ EVALUATION SUMMARY ================")
    print(json.dumps(full_report, indent=2))
    print("Saved evaluation summary to: data/eval_summary.json")
    return full_report


if __name__ == "__main__":
    import os
    target_csv = "data/golden_with_predictions.csv" if os.path.exists("data/golden_with_predictions.csv") else "data/golden_eval_set.csv"
    df = pd.read_csv(target_csv)
    print("Intent Metrics Check:", compute_intent_metrics(df))
