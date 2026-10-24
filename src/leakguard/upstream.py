"""The HTTP client that talks to the real (or mock) LLM provider."""

from collections.abc import Mapping

import httpx

from leakguard.config import Settings


def upstream_url(base_url: str, path: str) -> str:
    """Join like the OpenAI SDK does: base URLs may or may not end with '/'."""
    return base_url.rstrip("/") + path


class UpstreamClient:
    """One shared AsyncClient for the app's lifetime, so connections are pooled
    and reused instead of opening a new TCP/TLS connection per request."""

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self._base_url = settings.upstream_base_url
        self._client = httpx.AsyncClient(timeout=settings.upstream_timeout_s, transport=transport)

    async def send(
        self, path: str, body: dict, headers: Mapping[str, str], stream: bool
    ) -> httpx.Response:
        """POST `body` as JSON. With stream=True the body is not read yet:
        the caller must iterate it and then `aclose()` the response."""
        request = self._client.build_request(
            "POST", upstream_url(self._base_url, path), json=body, headers=dict(headers)
        )
        return await self._client.send(request, stream=stream)

    async def aclose(self) -> None:
        await self._client.aclose()
