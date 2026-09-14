import os
import sys

# Prevent macOS OpenMP / PyTorch / Tokenizers runtime conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import time
import yaml
import pandas as pd
from tqdm import tqdm
from src.pipeline import SupportPipeline
from eval.llm_judge import SupportEvalJudge

def compute_metrics(df: pd.DataFrame):
    """Computes evaluation benchmarks using pandas to prevent C-library conflicts."""
    print("\n=================== INTENT CLASSIFICATION METRICS ===================")
    intents = sorted(list(set(df["true_intent"].unique()).union(set(df["pred_intent"].unique()))))
    
    metrics = []
    total_correct = (df["true_intent"] == df["pred_intent"]).sum()
    overall_acc = total_correct / len(df) if len(df) > 0 else 0

    for intent in intents:
        tp = ((df["true_intent"] == intent) & (df["pred_intent"] == intent)).sum()
        fp = ((df["true_intent"] != intent) & (df["pred_intent"] == intent)).sum()
        fn = ((df["true_intent"] == intent) & (df["pred_intent"] != intent)).sum()
        support = (df["true_intent"] == intent).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics.append({
            "intent": intent,
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1-score": round(f1, 2),
            "support": support
        })

    metrics_df = pd.DataFrame(metrics).set_index("intent")
    print(metrics_df.to_string())
    print(f"\nOverall Intent Accuracy: {overall_acc * 100:.1f}% ({total_correct}/{len(df)})")
    print(f"Macro F1-Score:          {metrics_df['f1-score'].mean():.2f}")

    print("\n=================== ESCALATION CONFUSION MATRIX ====================")
    cm = pd.crosstab(
        df["true_escalation"],
        df["pred_escalation"],
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False
    )
    # Ensure standard binary columns exist
    for col in ["AUTO_HANDLE", "ESCALATE"]:
        if col not in cm.columns:
            cm[col] = 0
    for idx in ["AUTO_HANDLE", "ESCALATE"]:
        if idx not in cm.index:
            cm.loc[idx] = 0
    cm = cm.loc[["AUTO_HANDLE", "ESCALATE"], ["AUTO_HANDLE", "ESCALATE"]]
    print(cm)

    # CRITICAL METRIC: False Auto-Handle Rate (Type II safety error)
    actual_escalates = (df["true_escalation"] == "ESCALATE").sum()
    false_autos = cm.loc["ESCALATE", "AUTO_HANDLE"] if "ESCALATE" in cm.index else 0
    false_auto_rate = (false_autos / actual_escalates * 100) if actual_escalates > 0 else 0.0

    print(f"\nCRITICAL SAFETY METRIC: False Auto-Handle Rate = {false_auto_rate:.2f}% ({false_autos}/{actual_escalates})")
    print(f"(Safety Target: < 5.0% | System chose caution over risky automation)")

    print("\n=================== LLM JUDGE QUALITY SCORES (1-5) =================")
    auto_handled = df[df["pred_escalation"] == "AUTO_HANDLE"]
    if not auto_handled.empty and auto_handled["judge_groundedness"].notna().any():
        print(f"Mean Groundedness:    {auto_handled['judge_groundedness'].dropna().mean():.2f} / 5.0")
        print(f"Mean Brand Tone:       {auto_handled['judge_tone'].dropna().mean():.2f} / 5.0")
        print(f"Mean Actionability:    {auto_handled['judge_actionability'].dropna().mean():.2f} / 5.0")
    else:
        print("No auto-handled queries were eligible for LLM judge evaluation in this slice.")

def run_evaluation(golden_path: str = "data/golden/golden_set.csv", sample_size: int = 25):
    print(f"Starting evaluation run over {golden_path}...")
    df = pd.read_csv(golden_path)
    df = df.dropna(subset=["true_intent", "true_escalation"]).reset_index(drop=True)

    # Subsample for evaluation pacing
    eval_df = df.head(sample_size)

    pipeline = SupportPipeline()
    judge = SupportEvalJudge()

    records = []
    print(f"Executing pipeline on {len(eval_df)} golden test cases with rate-limit pacing...")

    for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
        res = None
        for attempt in range(4):
            try:
                res = pipeline.process(str(row["customer_text"]))
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    time.sleep(6)
                else:
                    time.sleep(2)

        if res is None:
            continue

        judge_res = {"groundedness": None, "brand_tone": None, "actionability": None}
        if res["decision"] == "AUTO_HANDLE":
            for attempt in range(4):
                try:
                    judge_res = judge.judge_reply(
                        str(row["customer_text"]),
                        str(row["brand_text"]),
                        res["draft_reply"]
                    )
                    break
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        time.sleep(6)
                    else:
                        break

        records.append({
            "tweet_id": row["customer_tweet_id"],
            "customer_text": row["customer_text"],
            "true_intent": row["true_intent"],
            "pred_intent": res["intent"],
            "true_escalation": row["true_escalation"],
            "pred_escalation": res["decision"],
            "escalation_reason": res["escalation_reason"],
            "groundedness_score": res["groundedness_score"],
            "judge_groundedness": judge_res.get("groundedness"),
            "judge_tone": judge_res.get("brand_tone"),
            "judge_actionability": judge_res.get("actionability"),
            "draft_reply": res["draft_reply"]
        })
        time.sleep(1.5)  # API pacing throttle

    results_df = pd.DataFrame(records)
    results_df.to_csv("data/processed/eval_run_results.csv", index=False)
    compute_metrics(results_df)

if __name__ == "__main__":
    run_evaluation()