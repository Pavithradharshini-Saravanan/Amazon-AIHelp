import time
import pandas as pd
from task5_classifier import trivial_baseline, keyword_baseline, llm_classify
from task7_reply_generator import generate_reply
from task8_escalation import llm_escalate
from task9_eval_harness import run_full_evaluation


def run_end_to_end_pipeline(input_csv="data/golden_eval_set.csv", output_csv="data/full_pipeline_results.csv"):
    print("=" * 60)
    print("      STARTING FULL AI SUPPORT AGENT PIPELINE EXECUTION")
    print("=" * 60)
    start_time = time.time()

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} golden evaluation samples from: {input_csv}\n")

    # Step 1: Classify Intents (Baselines + LLM)
    print("[1/4] Running Intent Classification (Trivial, Keyword, LLM)...")
    most_common_intent = df['intent'].mode()[0] if 'intent' in df.columns else 'delivery_issue'
    df['pred_trivial'] = df['customer_message'].apply(lambda msg: most_common_intent)
    df['pred_keyword'] = df['customer_message'].apply(keyword_baseline)

    llm_intents = []
    for i, msg in enumerate(df['customer_message']):
        llm_intents.append(llm_classify(msg))
        if (i + 1) % 50 == 0 or (i + 1) == len(df):
            print(f"   Intent classification: {i + 1}/{len(df)} complete")
    df['pred_llm'] = llm_intents

    # Step 2 & 3: Retrieval & Grounded Reply Generation
    print("\n[2/4] Running RAG Retrieval & Grounded Reply Generation...")
    replies = []
    for i, row in df.iterrows():
        msg = row['customer_message']
        intent = row['pred_llm']
        r = generate_reply(msg, intent)
        replies.append(r)
        if (i + 1) % 50 == 0 or (i + 1) == len(df):
            print(f"   Reply generation: {i + 1}/{len(df)} complete")
    df['generated_reply'] = replies

    # Step 4: Escalation Decisions & Reasons
    print("\n[3/4] Running Escalation Decision Logic & Reason Engine...")
    escalations = []
    reasons = []
    for i, row in df.iterrows():
        msg = row['customer_message']
        intent = row['pred_llm']
        esc, reason = llm_escalate(msg, intent)
        escalations.append(esc)
        reasons.append(reason)
        if (i + 1) % 50 == 0 or (i + 1) == len(df):
            print(f"   Escalation logic: {i + 1}/{len(df)} complete")
    df['pred_should_escalate'] = escalations
    df['pred_escalation_reason'] = reasons

    # Save full pipeline predictions
    df.to_csv(output_csv, index=False)
    print(f"\n[4/4] Pipeline execution complete! Results saved to: {output_csv}")

    # Run Eval Harness
    print("\nRunning Evaluation Harness on results...")
    summary = run_full_evaluation(output_csv)

    elapsed = round(time.time() - start_time, 2)
    print(f"\nPipeline finished in {elapsed} seconds.")
    return df, summary


if __name__ == "__main__":
    run_end_to_end_pipeline()
