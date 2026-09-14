import os
import yaml
from dotenv import load_dotenv
from groq import Groq

def fix_config():
    load_dotenv()
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    # Fetch active models allowed by your specific API key
    available_models = [m.id for m in client.models.list().data if "whisper" not in m.id]
    
    if not available_models:
        print("Error: No text models found for this API key.")
        return

    # Auto-select the most capable available model
    chosen_model = available_models[0]
    for m in available_models:
        if "llama" in m or "gemma" in m:
            chosen_model = m
            break

    print(f"✅ Auto-selected active model: {chosen_model}")

    # Read and update config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    for role in ["classifier", "drafter", "critic", "judge"]:
        config["models"][role] = chosen_model

    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print("✅ config.yaml updated! You are ready to run Streamlit.")

if __name__ == "__main__":
    fix_config()