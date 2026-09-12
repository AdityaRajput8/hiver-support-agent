import argparse
import pandas as pd
from pathlib import Path

def create_sample_for_labeling(threads_path: str, out_path: str, n: int = 200):
    """
    Samples diverse customer inquiries for manual labeling.
    Stratified across short (<50 chars), medium (50-120 chars), and long (>120 chars) tweets.
    """
    df = pd.read_parquet(threads_path)
    df["char_len"] = df["customer_text"].str.len()
    
    bins = [0, 50, 120, 1000]
    labels = ["short", "medium", "long"]
    df["len_strata"] = pd.cut(df["char_len"], bins=bins, labels=labels)

    sample = df.groupby("len_strata", observed=False).apply(
        lambda x: x.sample(n=n // len(labels), random_state=42)
    ).reset_index(drop=True)

    # Columns required for golden evaluation
    sample["true_intent"] = ""
    sample["true_escalation"] = ""  # AUTO_HANDLE or ESCALATE
    sample["notes"] = ""

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sample[["customer_tweet_id", "customer_text", "brand_text", "true_intent", "true_escalation", "notes"]].to_csv(out_path, index=False)
    print(f"Generated raw golden sampling set with {len(sample)} rows at {out_path}.")
    print("Action Required: Complete manual annotation in 'true_intent' and 'true_escalation'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", default="data/processed/uber_threads.parquet")
    parser.add_argument("--out", default="data/golden/golden_set.csv")
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()
    create_sample_for_labeling(args.threads, args.out, args.n)