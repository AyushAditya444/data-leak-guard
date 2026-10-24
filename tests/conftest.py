from contextlib import ExitStack

import httpx
import pytest
from fastapi.testclient import TestClient

from leakguard.app import create_app
from leakguard.config import Settings
from leakguard.mock_llm import create_mock_app
from leakguard.policy import Policy


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def mock_app():
    return create_mock_app(chunk_size=3)


@pytest.fixture
def make_guard(mock_app):
    """Build a guard test client. By default its upstream is the in-process mock LLM;
    pass `transport=httpx.MockTransport(...)` to fake specific upstream behaviour."""
    with ExitStack() as stack:
        def make(transport: httpx.AsyncBaseTransport | None = None, policy: Policy | None = None,
                 **overrides) -> TestClient:
            settings = Settings(**{"upstream_base_url": "http://mock/v1", **overrides})
            app = create_app(settings, transport or httpx.ASGITransport(app=mock_app), policy)
            return stack.enter_context(TestClient(app))  # entering runs the app's lifespan

        yield make


@pytest.fixture
def guard(make_guard):
    return make_guard()
