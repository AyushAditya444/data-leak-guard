import pytest

from leakguard.mock_llm import create_mock_app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def mock_app():
    return create_mock_app(chunk_size=3)
