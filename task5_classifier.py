import pandas as pd
from llm_client import generate_text

# Load the golden evaluation set
golden = pd.read_csv("data/golden_eval_set.csv")

# Find the most common intent
most_common_intent = golden['intent'].mode()[0]
print("Most common intent:", most_common_intent)

# Trivial baseline
def trivial_baseline(message):
    return most_common_intent

# Simple keyword baseline
def keyword_baseline(message):
    text = str(message).lower()

    if any(w in text for w in ['refund', 'return', 'money back']):
        return 'refund_return'
    elif any(w in text for w in ['damaged', 'broken', 'wrong item', 'defective', 'smashed']):
        return 'damaged_or_wrong_item'
    elif any(w in text for w in ['late', 'delayed', 'not arrived', 'tracking', 'deliver', 'ship', 'parcel', 'courier']):
        return 'delivery_issue'
    elif any(w in text for w in ['password', 'login', 'account', 'sign in', 'blocked', 'phishing']):
        return 'account_access'
    elif any(w in text for w in ['charged', 'billing', 'payment', 'card', 'deduct', 'price', 'fee']):
        return 'billing_payment'
    elif any(w in text for w in ['?', 'how do i', 'does this', 'where can i', 'can i']):
        return 'product_inquiry'
    else:
        return 'general_complaint'

INTENTS = [
    'delivery_issue',
    'damaged_or_wrong_item',
    'refund_return',
    'account_access',
    'billing_payment',
    'general_complaint',
    'product_inquiry',
    'other'
]

def llm_classify(message):
    prompt = f"""Classify this customer message into exactly one intent from this list:
{INTENTS}

Customer Message: "{message}"

Reply with ONLY the exact intent string from the list above, nothing else."""

    res = generate_text(prompt, temperature=0.0)
    cleaned = res.strip().replace("'", "").replace('"', "").lower()
    
    for intent in INTENTS:
        if intent in cleaned:
            return intent
            
    return 'other'

print("Running Intent Classifiers on Golden Set...")

golden['pred_trivial'] = golden['customer_message'].apply(trivial_baseline)
golden['pred_keyword'] = golden['customer_message'].apply(keyword_baseline)

llm_preds = []
for i, msg in enumerate(golden['customer_message']):
    label = llm_classify(msg)
    llm_preds.append(label)
    if (i + 1) % 50 == 0 or (i + 1) == len(golden):
        print(f"Processed {i + 1}/{len(golden)} classification samples")

golden['pred_llm'] = llm_preds

golden.to_csv("data/golden_with_predictions.csv", index=False)
print("Saved all intent predictions to: data/golden_with_predictions.csv")