import os
from dotenv import load_dotenv

load_dotenv(override=True)

key = os.getenv("OPENAI_API_KEY", "")
pine = os.getenv("PINECONE_API_KEY", "")

print("=== OpenAI Key ===")
print(f"  Length : {len(key)} chars")
print(f"  Valid  : {key.startswith('sk-')}")
print(f"  Preview: {key[:12]}...{key[-6:]}")

print("\n=== Pinecone Key ===")
print(f"  Length : {len(pine)} chars")
print(f"  Preview: {pine[:8]}...{pine[-6:]}")
