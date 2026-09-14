import json
import yaml
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class LLMClassifier:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        with open(self.cfg["paths"]["taxonomy"], "r") as f:
            self.taxonomy = yaml.safe_load(f)["intents"]

        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = self.cfg["models"]["classifier"]

    def _build_system_prompt(self) -> str:
        intents_desc = "\n".join([f"- `{item['id']}`: {item['description']}" for item in self.taxonomy])
        return (
            "You are a specialized customer intent classifier for Uber Twitter support.\n"
            "Analyze the customer tweet and assign the most appropriate intent from this list:\n"
            f"{intents_desc}\n\n"
            "Return JSON matching this exact schema:\n"
            "{\n"
            '  "intent": "<intent_id>",\n'
            '  "confidence": <float between 0.0 and 1.0>,\n'
            '  "reasoning": "<concise rationale>"\n'
            "}"
        )

    def classify(self, text: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                max_tokens=120,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": f"Customer Tweet: \"{text}\""}
                ]
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            return {"intent": "other_unroutable", "confidence": 0.0, "reasoning": f"Parse failure: {str(e)}"}