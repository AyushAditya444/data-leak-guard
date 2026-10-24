"""Proves the README's 'change one line' claim with the official OpenAI SDK."""

from openai import OpenAI


def _client(guard):
    # TestClient is an httpx.Client subclass, so the SDK accepts it as its HTTP client.
    return OpenAI(base_url="http://testserver/v1", api_key="sk-test", http_client=guard)


def test_openai_sdk_non_stream(guard):
    out = _client(guard).chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}])
    assert out.choices[0].message.content == "You said: hi"


def test_openai_sdk_stream(guard):
    stream = _client(guard).chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}], stream=True)
    assert "".join(c.choices[0].delta.content or "" for c in stream if c.choices) == "You said: hi"
