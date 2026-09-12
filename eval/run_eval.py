import yaml
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from src.pipeline import SupportPipeline
from eval.llm_judge import SupportEvalJudge

def run_evaluation(golden_path: str = "data/golden/golden_set.csv"):
    print(f"Starting evaluation run over {golden_path}...")
    df = pd.read_csv(golden_path)
    df = df.dropna(subset=["true_intent", "true_escalation"])

    pipeline = SupportPipeline()
    judge = SupportEvalJudge()

    records = []
    print(f"Executing pipeline on {len(df)} golden test cases...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        res = pipeline.process(row["customer_text"])
        
        # If auto-handled, run judge score
        judge_res = {"groundedness": None, "brand_tone": None, "actionability": None}
        if res["decision"] == "AUTO_HANDLE":
            judge_res = judge.judge_reply(
                row["customer_text"],
                row["brand_text"],
                res["draft_reply"]
            )

        records.append({
            "tweet_id": row["customer_tweet_id"],
            "customer_text": row["customer_text"],
            "true_intent": row["true_intent"],
            "pred_intent": res["intent"],
            "true_escalation": row["true_escalation"],
            "pred_escalation": res["decision"],
            "escalation_reason": res["escalation_reason"],
            "groundedness_score": res["groundedness_score"],
            "judge_groundedness": judge_res["groundedness"],
            "judge_tone": judge_res["brand_tone"],
            "judge_actionability": judge_res["actionability"],
            "draft_reply": res["draft_reply"]
        })

    results_df = pd.DataFrame(records)
    results_df.to_csv("data/processed/eval_run_results.csv", index=False)

    print("\n=================== INTENT CLASSIFICATION METRICS ===================")
    print(classification_report(results_df["true_intent"], results_df["pred_intent"], zero_division=0))

    print("\n=================== ESCALATION DECISION CONFUSION MATRIX ============")
    cm = confusion_matrix(results_df["true_escalation"], results_df["pred_escalation"], labels=["AUTO_HANDLE", "ESCALATE"])
    cm_df = pd.DataFrame(cm, index=["Actual AUTO", "Actual ESCALATE"], columns=["Pred AUTO", "Pred ESCALATE"])
    print(cm_df)

    # CRITICAL METRIC: False Auto-Handle Rate
    # (System auto-handled something that required human intervention - safety risk)
    false_autos = cm_df.loc["Actual ESCALATE", "Pred AUTO"]
    total_escalations = results_df["true_escalation"].value_counts().get("ESCALATE", 1)
    safety_failure_rate = (false_autos / total_escalations) * 100
    print(f"\nCRITICAL SAFETY METRIC: False Auto-Handle Rate = {safety_failure_rate:.2f}% ({false_autos}/{total_escalations})")

    print("\n=================== LLM JUDGE QUALITY SCORES (1-5) =================")
    auto_handled = results_df[results_df["pred_escalation"] == "AUTO_HANDLE"]
    print(f"Mean Groundedness:    {auto_handled['judge_groundedness'].mean():.2f} / 5.0")
    print(f"Mean Brand Tone:       {auto_handled['judge_tone'].mean():.2f} / 5.0")
    print(f"Mean Actionability:    {auto_handled['judge_actionability'].mean():.2f} / 5.0")

if __name__ == "__main__":
    run_evaluation()