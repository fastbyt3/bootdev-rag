import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")
print(f"Using key {api_key[:6]}...")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
)

# print(f"Response = {response.text}")
print(f"Prompt Tokens: {response.usage_metadata.prompt_token_count}")
print(f"Response Tokens: {response.usage_metadata.candidates_token_count}")
