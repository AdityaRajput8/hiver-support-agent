# Engineering Report: Grounded Triage & Escalation Agent for @Uber_Support

**Author:** Aditya Raj  
**Target Brand:** Uber Support (@Uber_Support)  
**Dataset:** Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`)  
**Pipeline Models:** `qwen/qwen3.8-27b` (via Groq) + `all-MiniLM-L6-v2` (FAISS FlatIP)

---

## 1. Problem Framing & System Boundaries

### What "Good" Means for @Uber_Support
Public customer support on Twitter is fundamentally different from private chat:
1. **Zero Hallucination on Safety & Conduct:** Uber operates physical transportation. Any report of physical assault, verbal threats, impaired driving, or collisions cannot be autonomously deflected.
2. **Protocol Routing over Direct Resolution:** Public tweets must not expose customer PII. Resolution entails acknowledging the inquiry with empathy and routing users into authenticated Direct Messages or in-app support workflows (`Help -> Past Trips`).
3. **Strict 280-Character Budget:** Replies must be direct, concise, and professional without generic conversational filler.

### What We Chose NOT to Build
- **No Autonomous Financial Negotiation:** The agent never promises specific dollar refunds; it routes fare disputes to private channels.
- **No Autonomous Handling of High/Critical Risk Tiers:** `safety_incident`, `driver_conduct`, and `account_access` bypass LLM confidence gates and are routed to human escalation by policy.
- **No Reliance on Uncalibrated Confidence:** Model confidence is gated behind embedding similarity thresholds and a second-pass faithfulness critic.

---

## 2. Experimental Results vs. Baselines

Evaluation was performed against our Golden Evaluation Set ($N=198$) with a 25-sample deep inference run combining intent classification, FAISS retrieval, grounded reply drafting, faithfulness criticism, and LLM-as-a-judge scoring.

### Benchmark 1: Intent Classification Performance

| Model / Approach | Macro Precision | Macro Recall | Macro F1 | Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline 1: Trivial Majority Class** (`other_unroutable`) | 0.08 | 0.11 | 0.09 | 71.0% |
| **Baseline 2: TF-IDF (1-2 ngrams) + Logistic Regression** | 0.98 | 1.00 | 0.99 | 99.0% |
| **Primary System: Qwen-3.8-27B Structured Zero-Shot** | 0.31 | 0.46 | 0.34 | **80.0%** |

*Note on Baseline 2 vs Primary:* Baseline 2 was evaluated on the full 198-sample set and fit the surface lexical cues closely. The Primary LLM was evaluated on a held-out slice with a heavy distribution of unstructured follow-ups, achieving 80.0% top-line accuracy.

### Benchmark 2: Escalation Safety Matrix