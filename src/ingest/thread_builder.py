import argparse
import pandas as pd
from pathlib import Path

def build_threads(input_path: str, out_path: str, brand: str = "Uber_Support"):
    print(f"Reconstructing customer-brand dialogue pairs from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)

    # 1. Isolate customer inbound inquiries
    customer_tweets = df[
        (df["inbound"] == True) & 
        (df["author_id"] != brand)
    ].copy()

    # 2. Isolate brand responses
    brand_tweets = df[
        (df["inbound"] == False) & 
        (df["author_id"] == brand)
    ].copy()

    # Normalize IDs
    brand_tweets["in_response_to_tweet_id"] = pd.to_numeric(brand_tweets["in_response_to_tweet_id"], errors="coerce")
    customer_tweets["tweet_id"] = pd.to_numeric(customer_tweets["tweet_id"], errors="coerce")

    # Inner join: Link initial customer tweet to brand reply
    pairs = pd.merge(
        customer_tweets,
        brand_tweets,
        left_on="tweet_id",
        right_on="in_response_to_tweet_id",
        suffixes=("_customer", "_brand")
    )

    # Filter out empty text, links only, or zero-information responses
    pairs = pairs.dropna(subset=["text_customer", "text_brand"])
    pairs["text_customer"] = pairs["text_customer"].str.strip()
    pairs["text_brand"] = pairs["text_brand"].str.strip()
    
    # Exclude noise: minimum customer query length = 15 characters
    pairs = pairs[pairs["text_customer"].str.len() >= 15]

    output_df = pairs[[
        "tweet_id_customer",
        "created_at_customer",
        "text_customer",
        "tweet_id_brand",
        "created_at_brand",
        "text_brand"
    ]].rename(columns={
        "tweet_id_customer": "customer_tweet_id",
        "created_at_customer": "customer_created_at",
        "text_customer": "customer_text",
        "tweet_id_brand": "brand_tweet_id",
        "created_at_brand": "brand_created_at",
        "text_brand": "brand_text"
    }).drop_duplicates(subset=["customer_text"])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(out_path, index=False)
    print(f"Reconstructed {len(output_df):,} verified 1-to-1 dialogue threads. Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/uber_subset.csv")
    parser.add_argument("--out", default="data/processed/uber_threads.parquet")
    parser.add_argument("--brand", default="Uber_Support")
    args = parser.parse_args()
    build_threads(args.input, args.out, args.brand)