from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from app.config import Settings
from app.models import QueryResponse, SearchDocument
from app.services.brave_search import BraveSearchService
from app.services.llm_runtime import LocalLLM
from app.services.scraper import ScrapedDocument, WebScraperService
from app.services.summarizer import SummarizerService

logger = logging.getLogger(__name__)


class RetrievalAugmentedPipeline:
    def __init__(
        self,
        settings: Settings,
        search: BraveSearchService,
        scraper: WebScraperService,
        summarizer: SummarizerService,
        llm: LocalLLM,
    ) -> None:
        self._settings = settings
        self._search = search
        self._scraper = scraper
        self._summarizer = summarizer
        self._llm = llm

    async def run(
        self,
        query: str,
        top_k: int,
        image_base64: Optional[str] = None,
    ) -> QueryResponse:
        search_results = await self._search.search(query, count=top_k)
        if not search_results:
            raise ValueError("No search results returned for the supplied query")

        scraped_docs = await self._scraper.fetch_documents(
            (doc.url, doc.title) for doc in search_results
        )
        summary_map = await self._summaries_by_url(scraped_docs)

        contexts = []
        filtered_sources: List[SearchDocument] = []
        for idx, doc in enumerate(search_results, start=1):
            summary = summary_map.get(doc.url)
            if summary:
                doc.summary = summary
                contexts.append(
                    f"[{idx}] {doc.title}\nURL: {doc.url}\nSummary:\n{summary}"
                )
                filtered_sources.append(doc)

        context_section = "\n\n".join(contexts)
        system_prompt = (
            "You are Scarper, a focused research assistant connected to a private local LLM. "
            "You must ground every answer in the supplied web context. "
            "Cite sources using bracketed indices like [1]. If the context is insufficient, say you don't know."
        )

        user_prompt_parts = [
            f"User question: {query}",
        ]
        if context_section:
            user_prompt_parts.append("Relevant context:")
            user_prompt_parts.append(context_section)
        else:
            user_prompt_parts.append(
                "No external context was retrieved. Answer conservatively and note the limitation."
            )

        user_prompt_parts.append(
            "Craft a precise, factual response under 250 words. Place source citations inline, e.g., [2]."
        )
        user_prompt = "\n\n".join(user_prompt_parts)

        user_content: List[dict] = [LocalLLM.format_text_content(user_prompt)]
        if image_base64:
            cleaned = _strip_data_url_prefix(image_base64)
            try:
                user_content.append(LocalLLM.format_image_content(cleaned))
            except ValueError as exc:
                raise ValueError("Invalid base64 image payload") from exc

        messages = [
            {"role": "system", "content": [LocalLLM.format_text_content(system_prompt)]},
            {"role": "user", "content": user_content},
        ]

        answer = await self._llm.chat(
            messages,
            max_tokens=self._settings.final_answer_max_tokens,
            temperature=self._settings.llm_temperature,
            top_p=self._settings.llm_top_p,
        )

        return QueryResponse(answer=answer, sources=filtered_sources or search_results)

    async def _summaries_by_url(
        self, scraped_docs: Iterable[ScrapedDocument]
    ) -> dict[str, str]:
        summary_map: dict[str, str] = {}
        for doc in scraped_docs:
            summary = await self._summarizer.summarize(doc)
            if summary:
                summary_map[doc.url] = summary
        return summary_map


def _strip_data_url_prefix(image_base64: str) -> str:
    if "," in image_base64 and image_base64.startswith("data:"):
        return image_base64.split(",", 1)[1]
    return image_base64
