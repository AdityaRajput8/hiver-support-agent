cat << 'EOF' > README.md
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
    subgraph INGESTION ["1. Ingestion & Retrieval Pipeline"]
        A[Inbound Customer Tweet] --> B[all-MiniLM-L6-v2 Embedder]
        B --> C[(FAISS FlatIP Index\n31,868 Historical Threads)]
        C -->|Top-k Historical Resolutions\nCosine Sim Score| D[Grounded Context Pool]
    end

    subgraph INFERENCE ["2. Intent & Generation Pipeline"]
        A --> E[LLM Intent Classifier\nZero-Shot Structured Schema]
        E -->|Intent ID, Confidence, Reasoning| F[Drafting Agent\nTwitter 280-Char Budget]
        D --> F
        F --> G[Proposed Draft Reply]
    end

    subgraph AUDIT ["3. Verification & Safety Gating"]
        G --> H[Faithfulness Critic\nSecond-Pass Cross-Check]
        D --> H
        H -->|Groundedness Score 0.0 - 1.0\nHallucination Flags| I{Deterministic Policy Engine}
        E -->|Confidence Score + Risk Tier| I
        C -->|Top Precedent Similarity| I
        A -->|Distress & Legal Keywords| I
    end

    subgraph ACTION ["4. Terminal Routing"]
        I -->|Passes All Thresholds| J[AUTO_HANDLE\nPublish Public Reply]
        I -->|Fails Any Safety Floor| K[ESCALATE\nHuman Triage Queue + Reason Code]
        J --> L[LLM-as-a-Judge\nQuality Audit]
    end

    style A fill:#1e293b,stroke:#94a3b8,stroke-width:2px,color:#fff
    style I fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff
    style J fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#fff
    style K fill:#7f1d1d,stroke:#ef4444,stroke-width:2px,color:#fff