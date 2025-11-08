from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException

from app.config import Settings, get_settings
from app.models import HealthResponse, QueryRequest, QueryResponse
from app.pipelines.retrieval_augmented import RetrievalAugmentedPipeline
from app.services.brave_search import BraveSearchError, BraveSearchService
from app.services.llm_runtime import LocalLLM
from app.services.scraper import WebScraperService
from app.services.summarizer import SummarizerService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Scarper Research Service",
    version="0.1.0",
    description=(
        "Bridges a local multimodal LLaVA instance with real-time web search, scraping, and summarisation."
    ),
)


def get_llm() -> LocalLLM:
    try:
        return _get_llm_instance()
    except FileNotFoundError as exc:  # pragma: no cover - configuration error
        logger.error("Local model missing: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@lru_cache
def _get_llm_instance() -> LocalLLM:
    settings = get_settings()
    return LocalLLM(settings)


@lru_cache
def get_brave_service() -> BraveSearchService:
    return BraveSearchService(get_settings())


@lru_cache
def get_scraper_service() -> WebScraperService:
    return WebScraperService(get_settings())


@lru_cache
def get_summarizer_service() -> SummarizerService:
    return SummarizerService(get_llm(), get_settings())


@lru_cache
def get_pipeline() -> RetrievalAugmentedPipeline:
    settings = get_settings()
    return RetrievalAugmentedPipeline(
        settings=settings,
        search=get_brave_service(),
        scraper=get_scraper_service(),
        summarizer=get_summarizer_service(),
        llm=get_llm(),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    pipeline: RetrievalAugmentedPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    top_k = min(request.top_k, settings.brave_result_count)

    try:
        return await pipeline.run(
            query=request.query,
            top_k=top_k,
            image_base64=request.image_base64,
        )
    except BraveSearchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        message = str(exc)
        if "Invalid base64" in message:
            raise HTTPException(status_code=400, detail=message) from exc
        raise HTTPException(status_code=404, detail=message)
