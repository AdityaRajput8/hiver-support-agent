import json
import yaml
import os
from dotenv import load_dotenv
from typing import List, Dict
from groq import Groq

load_dotenv()

class FaithfulnessCritic:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
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

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=120,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        return json.loads(response.choices[0].message.content.strip())