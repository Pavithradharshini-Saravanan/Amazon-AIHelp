import json
import re
import pandas as pd
from llm_client import generate_text


def llm_escalate(customer_message, intent):
    """
    Pure LLM escalation decision engine.
    """
    prompt = f"""You are an escalation decision engine for Amazon Twitter support.

Analyze this incoming customer message and decide if it can be AUTO-HANDLED by an AI bot or must be ESCALATED to a human support agent.

Criteria for ESCALATED ("yes"):
- Customer is extremely angry or threatening to leave/sue.
- Financial dispute (high refund amount, unauthorized card charges, lost $100+ merchandise).
- Repeated failure by previous support agents or missed multiple delivery windows.
- Complex account issue (locked account, security breach, phishing alert).

Criteria for AUTO-HANDLED ("no"):
- Standard tracking status inquiry.
- General product or policy question.
- Praise, compliment, or routine feedback.

Customer Message: "{customer_message}"
Intent: {intent}

Output valid JSON strictly in this format:
{{\"should_escalate\": \"yes\" or \"no\", \"reason\": \"<1 concise sentence explanation>\"}}"""

    res = generate_text(prompt, temperature=0.0)

    # Extract JSON substring
    match = re.search(r'\{.*\}', res, re.DOTALL)
    if match:
        data = json.loads(match.group(0))
        dec = str(data.get("should_escalate", "no")).lower().strip()
        dec = "yes" if dec in ["yes", "true", "y"] else "no"
        reason = str(data.get("reason", "Standard customer query."))
        return dec, reason

    return "no", res.strip()


def predict_escalation_dataset(csv_path="data/golden_eval_set.csv"):
    df = pd.read_csv(csv_path)
    print(f"Running LLM escalation decision engine on {len(df)} samples...")

    pred_esc = []
    reasons = []

    for i, row in df.iterrows():
        intent = row.get('intent', 'other')
        msg = row['customer_message']

        dec, reason = llm_escalate(msg, intent)
        pred_esc.append(dec)
        reasons.append(reason)

        if (i + 1) % 50 == 0 or (i + 1) == len(df):
            print(f"Processed escalation for {i + 1}/{len(df)} samples")

    df['pred_should_escalate'] = pred_esc
    df['pred_escalation_reason'] = reasons

    df.to_csv("data/golden_with_escalation.csv", index=False)
    print("Saved escalation predictions to: data/golden_with_escalation.csv")
    return df


if __name__ == "__main__":
    test_msg = "800 $ lost in merchandise and you just can not be held responsible"
    dec, reason = llm_escalate(test_msg, "delivery_issue")
    print(f"Test Msg Escalation: {dec} | Reason: {reason}")

    predict_escalation_dataset()
