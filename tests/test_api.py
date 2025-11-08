from typing import Optional

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app, get_pipeline, get_settings
from app.models import QueryResponse, SearchDocument


@pytest.fixture(autouse=True)
def cleanup_overrides():
    app.dependency_overrides = {}
    yield
    app.dependency_overrides = {}


@pytest.fixture()
def test_client() -> TestClient:
    return TestClient(app)


class SuccessfulPipeline:
    def __init__(self):
        self.calls: list[tuple[str, int, Optional[str]]] = []

    async def run(self, query: str, top_k: int, image_base64: Optional[str] = None) -> QueryResponse:
        self.calls.append((query, top_k, image_base64))
        return QueryResponse(
            answer="Grounded response",
            sources=[
                SearchDocument(
                    url="https://example.com/article",
                    title="Example Article",
                    snippet="Example snippet",
                    summary="Clean summary",
                )
            ],
        )


class FailingPipeline:
    def __init__(self, exc: Exception):
        self.exc = exc

    async def run(self, query: str, top_k: int, image_base64: Optional[str] = None) -> QueryResponse:
        raise self.exc


def test_query_endpoint_success(test_client: TestClient) -> None:
    pipeline = SuccessfulPipeline()
    settings = Settings(brave_api_key="testing-key")
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_settings] = lambda: settings

    response = test_client.post(
        "/query",
        json={"query": "Latest AI breakthroughs", "top_k": 10},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Grounded response"
    assert payload["sources"][0]["url"] == "https://example.com/article"
    assert pipeline.calls == [("Latest AI breakthroughs", settings.brave_result_count, None)]


def test_query_endpoint_invalid_base64_returns_400(test_client: TestClient) -> None:
    pipeline = FailingPipeline(ValueError("Invalid base64 image payload"))
    settings = Settings(brave_api_key="testing-key")
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_settings] = lambda: settings

    response = test_client.post(
        "/query",
        json={"query": "Explain this image", "top_k": 3, "image_base64": "@@not-base64@@"},
    )

    assert response.status_code == 400
    assert "Invalid base64" in response.json()["detail"]
