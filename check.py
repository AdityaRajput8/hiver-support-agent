import os
from dotenv import load_dotenv
from groq import Groq

# Load the working key directly from your .env file
load_dotenv()

try:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    models = client.models.list()
    print("\n✅ YOUR API KEY HAS ACCESS TO THESE MODELS:")
    for m in models.data:
        if "whisper" not in m.id: 
            print(f'  - "{m.id}"')
except Exception as e:
    print(f"Error: {e}")