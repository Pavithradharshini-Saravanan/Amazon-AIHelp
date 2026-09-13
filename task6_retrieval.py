import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Step 2: Load cleaned pairs
pairs = pd.read_csv("data/cleaned_pairs.csv")

print("Cleaned pairs:", pairs.shape)

# Step 3: Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 4 and 5:
# Load the embeddings and historical pairs that were already saved
history_embeddings = np.load("data/history_embeddings.npy")
pairs = pd.read_csv("data/history_pairs.csv")

print("History pairs:", pairs.shape)
print("Embeddings shape:", history_embeddings.shape)


# Step 6: Retrieval function
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


# Step 7: Quick test
test_msg = "my package hasn't arrived and it's been 5 days"

results = retrieve_similar(test_msg, k=3)

for r in results:
    print("\nSimilarity:", r['similarity_score'])
    print("Customer:", r['similar_customer_message'])
    print("AmazonHelp reply:", r['brand_reply'])


print("\nRetrieval test completed successfully.")