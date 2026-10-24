"""Talk to an LLM through the guard using the official OpenAI SDK.

The only change from normal OpenAI code is `base_url`.

    python examples/chat.py
"""

import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("LEAKGUARD_URL", "http://localhost:8080/v1"),  # <- the one line
    api_key=os.environ.get("OPENAI_API_KEY", "not-needed-for-mock"),
)
model = os.environ.get("LEAKGUARD_MODEL", "mock")
messages = [{"role": "user", "content": "Hi, my phone number is 9876543210."}]

reply = client.chat.completions.create(model=model, messages=messages)
print("Full reply:   ", reply.choices[0].message.content)

print("Streamed reply:", end=" ", flush=True)
for chunk in client.chat.completions.create(model=model, messages=messages, stream=True):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="|", flush=True)  # '|' shows chunk borders
print()
