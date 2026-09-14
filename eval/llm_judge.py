import json
import yaml
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class SupportEvalJudge:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = self.cfg["models"]["judge"]

    def judge_reply(self, customer_query: str, ground_truth_reply: str, model_draft: str) -> dict:
        prompt = (
            "You are an impartial Quality Assurance auditor for Uber Customer Support.\n"
            "Score the AI Agent's draft reply against the customer inquiry and the historical brand reply.\n\n"
            f"Customer Tweet: \"{customer_query}\"\n"
            f"Historical Support Reply: \"{ground_truth_reply}\"\n"
            f"Agent Proposed Reply: \"{model_draft}\"\n\n"
            "Score on a 1 to 5 scale (integer only):\n"
            "1. groundedness: Does it avoid hallucinating policies/phone numbers?\n"
            "2. brand_tone: Is it professional, empathetic, and concise (<280 chars)?\n"
            "3. actionability: Does it direct the user to the proper protocol (DM, Help section)?\n\n"
            "Output valid JSON only matching:\n"
            "{\n"
            '  "groundedness": <1-5>,\n'
            '  "brand_tone": <1-5>,\n'
            '  "actionability": <1-5>,\n'
            '  "verdict_reason": "<1-2 sentence explanation>"\n'
            "}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=150,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response.choices[0].message.content.strip())