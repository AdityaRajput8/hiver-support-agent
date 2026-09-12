import pandas as pd
from sklearn.metrics import cohen_kappa_score

def compute_judge_human_alignment(scored_csv_path: str):
    """
    Computes quadratic-weighted Cohen's Kappa between Human Judge and LLM Judge
    across a 30-sample validation subset to validate judge trustworthiness.
    """
    df = pd.read_csv(scored_csv_path)
    
    # Required columns: human_score, llm_score
    human = df["human_score"].dropna()
    llm = df["llm_score"].dropna()

    kappa = cohen_kappa_score(human, llm, weights="quadratic")
    percentage_agreement = (human == llm).mean() * 100

    print("--- LLM Judge vs. Human Alignment Benchmark ---")
    print(f"Sample Size Evaluated: {len(human)}")
    print(f"Exact Agreement: {percentage_agreement:.1f}%")
    print(f"Quadratic Weighted Cohen's Kappa: {kappa:.3f}")
    if kappa > 0.6:
        print("Verdict: Substantial agreement. Judge rubric is calibrated for automated benchmarking.")
    else:
        print("Verdict: Moderate to low agreement. Rubric requires recalibration.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        compute_judge_human_alignment(sys.argv[1])