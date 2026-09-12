import json
import yaml
from typing import List, Dict
from anthropic import Anthropic

class FaithfulnessCritic:
    """
    Second-pass verification. Validates whether the draft introduces hallucinated
    actions, non-existent contact methods, or unsupported promises.
    """
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.client = Anthropic()
        self.model = self.cfg["models"]["critic"]

    def critique(self, query: str, draft: str, precedents: List[Dict]) -> dict:
        evidence_text = "\n".join([f"- {p['historical_reply']}" for p in precedents])
        
        system_prompt = (
            "You are an AI audit critic reviewing drafted customer support replies for brand safety.\n"
            "Evaluate if the drafted reply is strictly grounded in historical resolutions or general verified brand protocols.\n"
            "Flag any hallucinated phone numbers, promised refunds not in evidence, or unverified claims.\n"
            "Return JSON matching:\n"
            "{\n"
            '  "groundedness_score": <float 0.0 to 1.0>,\n'
            '  "is_hallucinated": <true/false>,\n'
            '  "critique_reason": "<brief justification>"\n'
            "}"
        )

        user_message = (
            f"Customer Tweet: {query}\n"
            f"Retrieved Precedents:\n{evidence_text}\n\n"
            f"Proposed Draft Reply: {draft}\n"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())