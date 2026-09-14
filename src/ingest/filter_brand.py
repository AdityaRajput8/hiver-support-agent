import argparse
import pandas as pd
from pathlib import Path

def filter_brand(input_path: str, brand: str, out_path: str):
    print(f"Reading {input_path} in chunks to filter for {brand}...")
    chunks = []
    chunk_size = 200_000
    
    # Read relevant columns only
    usecols = ["tweet_id", "author_id", "inbound", "created_at", 
               "text", "response_tweet_id", "in_response_to_tweet_id"]
    
    for chunk in pd.read_csv(input_path, usecols=usecols, chunksize=chunk_size, low_memory=False):
        # Match tweets written by brand or written directly to brand
        match = (chunk["author_id"] == brand) | (chunk["text"].str.contains(f"@{brand}", case=False, na=False))
        matched_chunk = chunk[match]
        if not matched_chunk.empty:
            chunks.append(matched_chunk)
            
    filtered = pd.concat(chunks, ignore_index=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(out_path, index=False)
    print(f"Saved {len(filtered):,} rows to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to twcs.csv")
    parser.add_argument("--brand", default="Uber_Support")
    parser.add_argument("--out", default="data/processed/uber_subset.csv")
    args = parser.parse_args()
    filter_brand(args.input, args.brand, args.out)