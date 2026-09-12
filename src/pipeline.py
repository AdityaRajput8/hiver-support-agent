from typing import Dict, Any
from src.retrieval.retriever import SupportRetriever
from src.classify.llm_classifier import LLMClassifier
from src.generate.draft_reply import ReplyDrafter
from src.generate.faithfulness_critic import FaithfulnessCritic
from src.policy.escalation_policy import EscalationEngine

class SupportPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        self.retriever = SupportRetriever(config_path)
        self.classifier = LLMClassifier(config_path)
        self.drafter = ReplyDrafter(config_path)
        self.critic = FaithfulnessCritic(config_path)
        self.policy = EscalationEngine(config_path)

    def process(self, customer_query: str) -> Dict[str, Any]:
        # Step 1: Intent Classification
        classification = self.classifier.classify(customer_query)
        intent = classification["intent"]

        # Step 2: Historical Precedent Retrieval
        retrieved = self.retriever.retrieve(customer_query)

        # Step 3: Draft Grounded Reply
        draft = self.drafter.draft(customer_query, intent, retrieved)

        # Step 4: Critique Groundedness & Hallucination Risk
        critique = self.critic.critique(customer_query, draft, retrieved)

        # Step 5: Deterministic Escalation Policy Check
        decision, reason, detail = self.policy.evaluate(
            query=customer_query,
            classification=classification,
            retrieved_items=retrieved,
            groundedness=critique
        )

        return {
            "query": customer_query,
            "intent": intent,
            "intent_confidence": classification.get("confidence", 0.0),
            "intent_reasoning": classification.get("reasoning", ""),
            "decision": decision,
            "escalation_reason": reason.value,
            "escalation_detail": detail,
            "draft_reply": draft,
            "groundedness_score": critique.get("groundedness_score", 0.0),
            "critic_reason": critique.get("critique_reason", ""),
            "retrieved_precedents": [
                {"score": round(r["similarity_score"], 3), "reply": r["historical_reply"]}
                for r in retrieved[:2]
            ]
        }