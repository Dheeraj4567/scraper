from __future__ import annotations

import logging
from typing import List

import httpx
from tenacity import AsyncRetrying, RetryError, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import Settings
from app.models import SearchDocument

logger = logging.getLogger(__name__)


class BraveSearchError(RuntimeError):
    pass


class BraveSearchService:
    def __init__(self, settings: Settings):
        self._settings = settings

    async def search(self, query: str, count: int) -> List[SearchDocument]:
        params = {
            "q": query,
            "count": min(count, 20),
            "search_lang": "en",
            "safesearch": self._settings.brave_safe_search,
        }
        headers = {
            "User-Agent": self._settings.user_agent,
            "Accept": "application/json",
            "X-Subscription-Token": self._settings.brave_api_key,
        }

        async def _do_request() -> List[SearchDocument]:
            async with httpx.AsyncClient(timeout=self._settings.fetch_timeout_seconds) as client:
                response = await client.get(
                    self._settings.brave_search_endpoint,
                    params=params,
                    headers=headers,
                    follow_redirects=True,
                )
                response.raise_for_status()
                payload = response.json()

            web = payload.get("web", {})
            results = web.get("results", [])

            documents: List[SearchDocument] = []
            for item in results:
                url = item.get("url")
                title = item.get("title", "")
                snippet = item.get("description")
                if not url or not title:
                    continue
                documents.append(
                    SearchDocument(
                        url=url,
                        title=title,
                        snippet=snippet,
                    )
                )
            return documents

        retrying = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            retry=retry_if_exception_type(httpx.HTTPError),
            reraise=True,
        )

        try:
            async for attempt in retrying:
                with attempt:
                    return await _do_request()
        except RetryError as exc:  # pragma: no cover - defensive logging
            logger.error("Brave search failed after retries", exc_info=exc)
            raise BraveSearchError("Unable to retrieve search results from Brave") from exc
