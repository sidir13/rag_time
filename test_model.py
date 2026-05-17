import os
import httpx
from openai import OpenAI

key = os.environ.get("OPENROUTER_API_KEY", "")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=key,
    http_client=httpx.Client(timeout=15.0)
)

models = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
]

# Test with a longer prompt to detect rate-limit issues under load
LONG_PROMPT = "You are a support agent. A customer says: 'I cannot login to my account on the mobile app. I tried resetting the password but the email never arrives. It worked yesterday.' Summarize the issue in 2 sentences."

for model in models:
    try:
        r = client.chat.completions.create(
            model=model,
            max_tokens=60,
            messages=[{"role": "user", "content": LONG_PROMPT}]
        )
        content = r.choices[0].message.content or ""
        print(f"OK: {model} -> {content[:80]!r}")
    except Exception as e:
        print(f"FAIL {model}: {str(e)[:100]}")
