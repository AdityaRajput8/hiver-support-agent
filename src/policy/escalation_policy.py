import yaml
from enum import Enum
from typing import Dict, Any, Tuple

class EscalationReason(Enum):
    CRITICAL_SAFETY_INTENT = "CRITICAL_SAFETY_INTENT"
    LOW_CLASSIFIER_CONFIDENCE = "LOW_CLASSIFIER_CONFIDENCE"
    NO_HISTORICAL_PRECEDENT = "NO_HISTORICAL_PRECEDENT"
    LOW_GROUNDEDNESS_SCORE = "LOW_GROUNDEDNESS_SCORE"
    DISTRESS_OR_LEGAL_KEYWORD = "DISTRESS_OR_LEGAL_KEYWORD"
    INTENT_REQUIRES_HUMAN = "INTENT_REQUIRES_HUMAN"
    RESOLVABLE_AUTO = "RESOLVABLE_AUTO"

class EscalationEngine:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        with open(self.cfg["paths"]["taxonomy"], "r") as f:
            tax = yaml.safe_load(f)["intents"]
            self.taxonomy_lookup = {item["id"]: item for item in tax}

        self.conf_thresh = self.cfg["policy"]["confidence_threshold"]
        self.ground_thresh = self.cfg["policy"]["groundedness_threshold"]
        self.sim_thresh = self.cfg["retrieval"]["min_similarity_threshold"]
        self.keywords = self.cfg["policy"]["sentiment_escalation_keywords"]

    def evaluate(
        self,
        query: str,
        classification: Dict[str, Any],
        retrieved_items: list,
        groundedness: Dict[str, Any]
    ) -> Tuple[str, EscalationReason, str]:
        """
        Deterministic, auditable policy gate.
        Returns: (decision: 'AUTO_HANDLE' | 'ESCALATE', reason_enum, description)
        """
        intent_id = classification.get("intent", "other_unroutable")
        confidence = classification.get("confidence", 0.0)
        ground_score = groundedness.get("groundedness_score", 0.0)
        top_sim = retrieved_items[0]["similarity_score"] if retrieved_items else 0.0

        # Rule 1: Distress/Legal/Emergency Keywords
        query_lower = query.lower()
        for kw in self.keywords:
            if kw in query_lower:
                return (
                    "ESCALATE",
                    EscalationReason.DISTRESS_OR_LEGAL_KEYWORD,
                    f"Triggered by safety/escalation keyword match: '{kw}'"
                )

        # Rule 2: Non-negotiable safety/critical risk intent
        intent_meta = self.taxonomy_lookup.get(intent_id, {})
        if intent_meta.get("risk_level") == "CRITICAL" or not intent_meta.get("auto_handle_eligible", True):
            return (
                "ESCALATE",
                EscalationReason.CRITICAL_SAFETY_INTENT,
                f"Intent '{intent_id}' is marked as non-auto-handleable by brand risk policy."
            )

        # Rule 3: Classification Confidence Floor
        if confidence < self.conf_thresh:
            return (
                "ESCALATE",
                EscalationReason.LOW_CLASSIFIER_CONFIDENCE,
                f"Intent confidence ({confidence:.2f}) is below operational floor ({self.conf_thresh:.2f})."
            )

        # Rule 4: Out-of-Distribution Precedent Check
        if top_sim < self.sim_thresh:
            return (
                "ESCALATE",
                EscalationReason.NO_HISTORICAL_PRECEDENT,
                f"Top retrieved precedent similarity ({top_sim:.2f}) is below threshold ({self.sim_thresh:.2f})."
            )

        # Rule 5: Hallucination/Low Groundedness Check
        if ground_score < self.ground_thresh:
            return (
                "ESCALATE",
                EscalationReason.LOW_GROUNDEDNESS_SCORE,
                f"Faithfulness score ({ground_score:.2f}) indicates potential hallucination or unsupported claim."
            )

        # Passes all gates
        return (
            "AUTO_HANDLE",
            EscalationReason.RESOLVABLE_AUTO,
            "Passed all confidence, safety, precedent, and groundedness thresholds."
        )