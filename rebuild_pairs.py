import pandas as pd
import re

# Step 1: Load the full dataset
df = pd.read_csv("data/twcs/twcs.csv")

# Step 2: Get AmazonHelp replies and customer messages
amazon_replies = df[df['author_id'] == 'AmazonHelp']
customer_msgs = df[df['author_id'] != 'AmazonHelp']

# Match each AmazonHelp reply with the customer message
pairs = amazon_replies.merge(
    customer_msgs,
    left_on='in_response_to_tweet_id',
    right_on='tweet_id',
    suffixes=('_reply', '_customer')
)

print("Matched pairs:", pairs.shape)

# Step 3: Subsample 8,000 matched pairs
SUBSAMPLE_SIZE = 8000

pairs = pairs.sample(n=SUBSAMPLE_SIZE, random_state=42) if len(pairs) > SUBSAMPLE_SIZE else pairs

print("After sampling:", pairs.shape)

# Step 4: Keep useful columns
pairs = pairs[['tweet_id_customer', 'text_customer', 'text_reply', 'created_at_customer']]

pairs.columns = ['tweet_id', 'customer_message', 'brand_reply', 'timestamp']

# Clean text
def clean_text(text):
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

pairs['customer_message'] = pairs['customer_message'].apply(clean_text)
pairs['brand_reply'] = pairs['brand_reply'].apply(clean_text)

# Remove junk
pairs = pairs[pairs['customer_message'].str.len() > 10]
pairs = pairs[pairs['brand_reply'].str.len() > 10]
pairs = pairs.drop_duplicates(subset='customer_message')
pairs = pairs.dropna()

# Reset index
pairs = pairs.reset_index(drop=True)

# Step 5: Save
pairs.to_csv("data/cleaned_pairs.csv", index=False)

print("Final cleaned pairs:", pairs.shape)
print(pairs.head())