import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llm_client import generate_text

# =========================
# Load historical data
# =========================
pairs = pd.read_csv("data/history_pairs.csv")
history_embeddings = np.load("data/history_embeddings.npy")
model = SentenceTransformer('all-MiniLM-L6-v2')

# =========================
# Retrieval function
# =========================
def retrieve_similar(new_message, k=3):
    new_embedding = model.encode([new_message])
    sims = cosine_similarity(
        new_embedding,
        history_embeddings
    )[0]

    top_k_idx = sims.argsort()[-k:][::-1]
    results = []

    for idx in top_k_idx:
        results.append({
            'similar_customer_message': pairs.iloc[idx]['customer_message'],
            'brand_reply': pairs.iloc[idx]['brand_reply'],
            'similarity_score': sims[idx]
        })

    return results

# =========================
# Build reply prompt
# =========================
def build_reply_prompt(customer_message, intent, similar_examples):
    examples_text = ""
    for i, ex in enumerate(similar_examples, 1):
        examples_text += f"""Example {i}:
Past customer message: "{ex['similar_customer_message']}"
Past brand reply: "{ex['brand_reply']}"
"""

    prompt = f"""You are a customer support agent for Amazon, replying on Twitter.

New customer message: "{customer_message}"
Detected intent: {intent}

Here is how similar past issues were resolved by the brand:
{examples_text}

Using the same tone, style, and approach shown above, write a reply to the new customer message.
Keep it short (Twitter-length), polite, and specific to their issue.
If the past replies ask the customer to DM for account details, do the same when appropriate.
Do not invent order numbers, refund amounts, or facts not in the message.

Reply:"""

    return prompt

# =========================
# Generate reply
# =========================
def generate_reply(customer_message, intent):
    similar_examples = retrieve_similar(customer_message, k=3)
    prompt = build_reply_prompt(
        customer_message,
        intent,
        similar_examples
    )
    return generate_text(prompt, temperature=0.0)

if __name__ == "__main__":
    # Test one example
    test_msg = "my package hasn't arrived and it's been 5 days"
    test_intent = "delivery_issue"

    reply = generate_reply(test_msg, test_intent)
    print("\nTest generated reply:")
    print(reply)

    # Run on golden set
    golden = pd.read_csv("data/golden_eval_set.csv")
    generated_replies = []

    print("\nGenerating grounded replies for Golden Set...")
    for i, row in golden.iterrows():
        r = generate_reply(
            row['customer_message'],
            row['intent']
        )
        generated_replies.append(r)

        if (i + 1) % 50 == 0 or (i + 1) == len(golden):
            print(f"Done {i + 1}/{len(golden)}")

    golden['generated_reply'] = generated_replies
    golden.to_csv("data/golden_with_replies.csv", index=False)

    print("\nReply generation completed successfully.")
    print("Saved to: data/golden_with_replies.csv")