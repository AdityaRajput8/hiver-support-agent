import yaml
from typing import List, Dict
from anthropic import Anthropic

class ReplyDrafter:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.client = Anthropic()
        self.model = self.cfg["models"]["drafter"]

    def draft(self, customer_query: str, intent: str, retrieved_resolutions: List[Dict]) -> str:
        resolutions_str = "\n".join([
            f"[Historical Precedent {i+1}]: {item['historical_reply']}"
            for i, item in enumerate(retrieved_resolutions)
        ])

        system_prompt = (
            "You are an AI support agent drafting public Twitter replies for @Uber_Support.\n"
            "Rules:\n"
            "1. Adhere strictly to Uber's tone: empathetic, professional, concise, direct.\n"
            "2. Under NO circumstances fabricate URLs, phone numbers, or policy terms.\n"
            "3. Reference historical precedent for how Uber directs users (e.g. asking for DM with account email, directing to in-app help/trip history).\n"
            "4. Keep response under 280 characters to fit a single tweet."
        )

        user_content = (
            f"Customer Inquiry: \"{customer_query}\"\n"
            f"Identified Intent: {intent}\n\n"
            f"Precedent Resolutions by @Uber_Support for similar issues:\n"
            f"{resolutions_str}\n\n"
            "Draft the reply:"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=150,
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}]
        )
        return response.content[0].text.strip()