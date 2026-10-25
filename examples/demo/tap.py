"""A demo-only wire tap: sits between the guard and the LLM provider and records what the
guard actually sent and what the provider sent back. Not part of the guard.

It is an httpx transport wrapped around the guard's real one, so it sees exactly the bytes
on the wire. It reads only the request's JSON body: never headers, so never the API key.
"""

import json
import time

import httpx

from leakguard.sse import SSEParser


def _delta_text(data: str | None) -> str:
    """The reply text in one streamed event ("" for role, finish and [DONE] events)."""
    try:
        choice = json.loads(data or "")["choices"][0]
        return choice["delta"].get("content") or ""
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        return ""


def _reply_text(raw: bytes) -> str:
    """A non-streamed reply's text; the raw body if it is not a chat completion (errors)."""
    try:
        return json.loads(raw)["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError, TypeError):
        return raw.decode("utf-8", errors="replace")


class WireTap(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport):
        self._inner = inner
        # One record per upstream request: {"messages", "status", "chunks": [[t, text]]},
        # t = seconds since the request was sent.
        self.records: list[dict] = []

    def clear(self) -> None:
        self.records = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        request.headers["accept-encoding"] = "identity"  # plain bytes, not gzip
        body = json.loads(request.content or b"{}")
        record = {"messages": body.get("messages", []), "status": 0, "chunks": []}
        self.records.append(record)
        started = time.perf_counter()
        response = await self._inner.handle_async_request(request)
        record["status"] = response.status_code
        streamed = (response.is_success
                    and "text/event-stream" in response.headers.get("content-type", ""))
        upstream = response.stream

        class Recorded(httpx.AsyncByteStream):
            async def __aiter__(self):
                parser, raw = SSEParser(), bytearray()
                async for chunk in upstream:
                    now = round(time.perf_counter() - started, 3)
                    if streamed:
                        record["chunks"] += [[now, text] for event in parser.feed(chunk)
                                             if (text := _delta_text(event.data))]
                    else:
                        raw.extend(chunk)
                    yield chunk
                if not streamed:
                    now = round(time.perf_counter() - started, 3)
                    record["chunks"] = [[now, _reply_text(bytes(raw))]]

            async def aclose(self):
                await upstream.aclose()

        return httpx.Response(response.status_code, headers=response.headers,
                              stream=Recorded(), extensions=response.extensions)

    async def aclose(self) -> None:
        await self._inner.aclose()
