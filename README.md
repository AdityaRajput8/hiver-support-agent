# 🚗 @Uber_Support AI Triage & Resolution Engine

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/Vector_Store-FAISS_FlatIP-orange.svg?style=flat-square)](https://github.com/facebookresearch/faiss)
[![LLM Inference](https://img.shields.io/badge/Inference-Groq_LPU-green.svg?style=flat-square)](https://groq.com/)
[![UI](https://img.shields.io/badge/Interface-Streamlit-red.svg?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![Evaluation](https://img.shields.io/badge/Audit-Calibrated_LLM_Judge-purple.svg?style=flat-square)](eval/)

An auditable, retrieval-grounded customer support triage system engineered for **@Uber_Support** on Twitter. The agent classifies multi-intent inquiries, retrieves historical verified resolutions, drafts policy-constrained public replies, audits groundedness via a secondary critic pass, and executes deterministic escalation gates.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph INGESTION ["1. Ingestion and Retrieval Pipeline"]
        A["Inbound Customer Tweet"] --> B["all-MiniLM-L6-v2 Embedder"]
        B --> C[("FAISS FlatIP Index<br/>31868 Historical Threads")]
        C -->|"Top-k Historical Resolutions<br/>Cosine Sim Score"| D["Grounded Context Pool"]
    end

    subgraph INFERENCE ["2. Intent and Generation Pipeline"]
        A --> E["LLM Intent Classifier<br/>Zero-Shot Structured Schema"]
        E -->|"Intent ID, Confidence, Reasoning"| F["Drafting Agent<br/>Twitter 280-Char Budget"]
        D --> F
        F --> G["Proposed Draft Reply"]
    end

    subgraph AUDIT ["3. Verification and Safety Gating"]
        G --> H["Faithfulness Critic<br/>Second-Pass Cross-Check"]
        D --> H
        H -->|"Groundedness Score 0.0 to 1.0<br/>Hallucination Flags"| I{"Deterministic Policy Engine"}
        E -->|"Confidence Score plus Risk Tier"| I
        C -->|"Top Precedent Similarity"| I
        A -->|"Distress and Legal Keywords"| I
    end

    subgraph ACTION ["4. Terminal Routing"]
        I -->|"Passes All Thresholds"| J["AUTO_HANDLE<br/>Publish Public Reply"]
        I -->|"Fails Any Safety Floor"| K["ESCALATE<br/>Human Triage Queue plus Reason Code"]
        J --> L["LLM-as-a-Judge<br/>Quality Audit"]
    end

    style A fill:#1e293b,stroke:#94a3b8,stroke-width:2px,color:#fff
    style I fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff
    style J fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#fff
    style K fill:#7f1d1d,stroke:#ef4444,stroke-width:2px,color:#fff
```

---

## ⚙️ Deterministic Policy & Operational Thresholds

To prevent model overconfidence from causing catastrophic safety failures, the agent delegates terminal decisions to a deterministic policy gate:

```
┌─── [Safety / Distress Keyword Triggered?] ─────────► ESCALATE (DISTRESS_OR_LEGAL_KEYWORD)
                  ├─── [Intent Risk Tier == CRITICAL / HIGH?] ─────────► ESCALATE (CRITICAL_SAFETY_INTENT)
Inbound Tweet ───►├─── [Intent Confidence < 0.72?] ───────────────────► ESCALATE (LOW_CLASSIFIER_CONFIDENCE)
                  ├─── [Top Precedent Cosine Similarity < 0.58?] ──────► ESCALATE (NO_HISTORICAL_PRECEDENT)
                  └─── [Faithfulness Critic Score < 0.80?] ────────────► ESCALATE (LOW_GROUNDEDNESS_SCORE)
                                  │
                                  ▼ (Passed All Gates)
                              AUTO_HANDLE
```

---

## 🖼️ Demo Screenshots

<p align="center">
  <img src="docs/screenshots/demo-1.png" alt="Demo Screenshot 1 - Streamlit Triage Interface" width="800"/>
  <br/>
  <em>Screenshot 1: Interactive audit & inspection interface</em>
</p>

<p align="center">
  <img src="docs/screenshots/demo-2.png" alt="Demo Screenshot 2 - Escalation and Resolution View" width="800"/>
  <br/>
  <em>Screenshot 2: Escalation decision trace with reason codes</em>
</p>

---

## 📂 Project Directory Structure

```
hiver-support-agent/
├── README.md                          # Quickstart, architecture, and reproducibility guide
├── report.md                          # Comprehensive 6-page technical report
├── decision_log.md                    # Auditable log of key architectural decisions
├── config.yaml                        # System thresholds, model endpoints, and parameters
├── requirements.txt                   # Production dependencies
├── Makefile                           # Unified automation tasks
├── docs/
│   └── screenshots/                   # Demo screenshots referenced in this README
│       ├── demo-1.png
│       └── demo-2.png
├── data/
│   ├── processed/                     # Thread-reconstructed dialogues & FAISS index
│   │   ├── uber_subset.csv
│   │   ├── uber_threads.parquet
│   │   ├── faiss_index.bin
│   │   └── index_metadata.parquet
│   └── golden/                        # Hand-labeled golden test set
│       ├── golden_set.csv
│       └── labeling_notes.md
├── src/
│   ├── ingest/                        # Dataset filtering & multi-turn thread reconstruction
│   │   ├── filter_brand.py
│   │   └── thread_builder.py
│   ├── intents/                       # Unsupervised cluster mining & domain taxonomy
│   │   ├── cluster_explore.py
│   │   └── taxonomy.yaml
│   ├── retrieval/                     # Dense vector search engine
│   │   ├── build_index.py
│   │   └── retriever.py
│   ├── classify/                      # Baseline models & primary structured classifier
│   │   ├── baseline_tfidf.py
│   │   └── llm_classifier.py
│   ├── generate/                      # Retrieval-conditioned generation & audit critic
│   │   ├── draft_reply.py
│   │   └── faithfulness_critic.py
│   ├── policy/                        # Policy-as-code deterministic escalation logic
│   │   └── escalation_policy.py
│   └── pipeline.py                    # End-to-end unified inference pipeline
├── eval/
│   ├── build_golden_sample.py         # Stratified golden set sampler
│   ├── annotate_golden.py             # Domain rule annotation engine
│   ├── llm_judge.py                   # Impartial QA judge rubric
│   ├── judge_human_agreement.py       # Cohen's Kappa calibration suite
│   └── run_eval.py                    # Full pipeline evaluation harness
└── app/
    └── streamlit_demo.py              # Interactive audit & inspection interface
```

---

## 🚀 Quickstart (< 10 Minutes Reproduction)

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/AdityaRajput8/hiver-support-agent
cd hiver-support-agent

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Credentials
Create a `.env` file in the project root:
```
GROQ_API_KEY=gsk_your_api_key_here
```

### 3. Run Baselines & Evaluation Suite
```bash
# Run baseline classifiers (Majority Class & TF-IDF + Logistic Regression)
python src/classify/baseline_tfidf.py

# Run the end-to-end evaluation harness on the golden set
python eval/run_eval.py

# Run judge-human calibration check
python eval/judge_human_agreement.py
```

### 4. Launch Interactive Audit UI
```bash
streamlit run app/streamlit_demo.py --server.fileWatcherType=none
```

---

## 🔍 Core Differentiators

- **Groundedness as a First-Class Metric:** Replies are not generated from parametric memory alone. The agent retrieves verified resolutions from @Uber_Support, conditions drafting on them, and scores faithfulness with a secondary auditor pass before publication.
- **Auditable Policy-as-Code:** Escalation is never delegated to subjective LLM confidence. Decisions run through an explicit Python rule engine with fixed error codes (`NO_HISTORICAL_PRECEDENT`, `CRITICAL_SAFETY_INTENT`, `LOW_GROUNDEDNESS_SCORE`).
- **Defense-in-Depth Risk Management:** Physical safety and driver conduct complaints are hard-coded to escalate by design, ensuring zero autonomous deflection on severe incidents.
- **Honest Metric Accounting:** The mandatory report section "What is Misleading About My Headline Number?" dissects majority-class skew and evaluation sample dynamics.

---

## 📸 How to Attach Demo Screenshots

Follow these steps to add your two demo screenshots so they render correctly in this README:

1. **Create a screenshots folder** in your repo (matches the structure above):
   ```bash
   mkdir -p docs/screenshots
   ```
2. **Copy your two screenshot files** into that folder and name them to match the README references (or rename the references above to match your files):
   ```bash
   cp /path/to/your/first-screenshot.png docs/screenshots/demo-1.png
   cp /path/to/your/second-screenshot.png docs/screenshots/demo-2.png
   ```
3. **Stage, commit, and push** the new files to GitHub:
   ```bash
   git add docs/screenshots/demo-1.png docs/screenshots/demo-2.png README.md
   git commit -m "Add demo screenshots"
   git push origin main
   ```
4. **Verify rendering** — open the README on GitHub (`github.com/AdityaRajput8/hiver-support-agent`) and confirm both images load under the "Demo Screenshots" section. If an image doesn't show, double-check the filename and path exactly match what's in the `<img src="...">` tag (GitHub paths are case-sensitive).

**Alternative (no local git needed):** On GitHub.com, open your repo → click **Add file → Upload files** → drag in your two screenshots → set the target path to `docs/screenshots/` → commit directly to `main`. The README's `<img>` tags will pick them up automatically once the paths match.