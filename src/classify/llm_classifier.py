import json
import yaml
from anthropic import Anthropic

class LLMClassifier:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        with open(self.cfg["paths"]["taxonomy"], "r") as f:
            self.taxonomy = yaml.safe_load(f)["intents"]

        self.client = Anthropic()
        self.model = self.cfg["models"]["classifier"]

    def _build_system_prompt(self) -> str:
        intents_desc = "\n".join([
            f"- `{item['id']}`: {item['description']}" 
            for item in self.taxonomy
        ])
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
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=0.0,
                system=self._build_system_prompt(),
                messages=[{"role": "user", "content": f"Customer Tweet: \"{text}\""}]
            )
            raw_content = response.content[0].text.strip()
            # Handle possible markdown fencing
            if raw_content.startswith("```"):
                raw_content = raw_content.split("```")[1]
                if raw_content.startswith("json"):
                    raw_content = raw_content[4:]
            return json.loads(raw_content.strip())
        except Exception as e:
            return {
                "intent": "other_unroutable",
                "confidence": 0.0,
                "reasoning": f"Classification parse failure: {str(e)}"
            }