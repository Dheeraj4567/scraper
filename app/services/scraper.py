from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import httpx
import trafilatura
from bs4 import BeautifulSoup
from readability import Document

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ScrapedDocument:
    url: str
    title: Optional[str]
    text: Optional[str]


class WebScraperService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_fetches)

    async def fetch_documents(self, documents: Iterable[tuple[str, str]]) -> List[ScrapedDocument]:
        """Fetch multiple documents concurrently.

        Args:
            documents: Iterable of (url, fallback_title)
        """
        tasks = [self._bounded_fetch(url, title) for url, title in documents]
        return [doc for doc in await asyncio.gather(*tasks) if doc is not None]

    async def _bounded_fetch(self, url: str, fallback_title: str) -> Optional[ScrapedDocument]:
        async with self._semaphore:
            return await self._fetch(url, fallback_title)

    async def _fetch(self, url: str, fallback_title: str) -> Optional[ScrapedDocument]:
        headers = {
            "User-Agent": self._settings.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        try:
            async with httpx.AsyncClient(timeout=self._settings.fetch_timeout_seconds, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                html = response.text
        except Exception as exc:  # pragma: no cover - network failure is external
            logger.warning("Unable to fetch %s: %s", url, exc)
            return None

        try:
            downloaded = trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=False,
                no_fallback=True,
            )
        except Exception as exc:  # pragma: no cover - library failure
            logger.warning("Trafilatura extraction failed for %s: %s", url, exc)
            downloaded = None

        text_content = downloaded.strip() if downloaded else None
        fallback_html_title: Optional[str] = None

        if not text_content:
            text_content, fallback_html_title = self._fallback_extract(html)
            if not text_content:
                logger.debug("Unable to extract meaningful text for %s", url)
                return None

        metadata = trafilatura.extract_metadata(html, url=url) if downloaded else None
        meta_title = metadata.title if metadata else None
        title = meta_title or fallback_html_title or fallback_title

        return ScrapedDocument(url=url, title=title, text=text_content)

    def _fallback_extract(self, html: str) -> Tuple[Optional[str], Optional[str]]:
        readability_text, readability_title = self._readability_extract(html)
        if readability_text:
            return readability_text, readability_title
        return self._beautifulsoup_extract(html)

    def _readability_extract(self, html: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            document = Document(html)
            summary_html = document.summary()
            summary_text = BeautifulSoup(summary_html, "html.parser").get_text(" ", strip=True)
            title = document.short_title() or document.title()
            clean_text = summary_text.strip() if summary_text else None
            clean_title = title.strip() if title else None
            return clean_text if clean_text else None, clean_title
        except Exception as exc:  # pragma: no cover - library failure
            logger.debug("Readability extraction failed: %s", exc)
            return None, None

    def _beautifulsoup_extract(self, html: str) -> Tuple[Optional[str], Optional[str]]:
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript", "template"]):
            tag.decompose()

        title = None
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        text = " ".join(s.strip() for s in soup.stripped_strings)
        cleaned = text.strip()
        return (cleaned if cleaned else None, title)
