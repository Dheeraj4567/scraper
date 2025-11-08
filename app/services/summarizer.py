from __future__ import annotations

import logging
from typing import Optional

from app.config import Settings
from app.services.llm_runtime import LocalLLM
from app.services.scraper import ScrapedDocument

logger = logging.getLogger(__name__)


class SummarizerService:
    def __init__(self, llm: LocalLLM, settings: Settings):
        self._llm = llm
        self._settings = settings

    async def summarize(self, document: ScrapedDocument) -> Optional[str]:
        if not document.text:
            return None

        prepared = self._prepare_text(document.text)
        if not prepared:
            return None

        system_prompt = (
            "You are a precise research assistant. Produce concise but information-dense summaries "
            "highlighting key facts, statistics, quotes, and caveats. Use bullet points when appropriate."
        )

        user_prompt = (
            "Summarize the following webpage content so it can be used as grounding context for another "
            "larger prompt. Keep it under 200 words and avoid redundancy.\n\n"
            f"Content from {document.title or document.url}:\n\n{prepared}"
        )

        messages = [
            {"role": "system", "content": [LocalLLM.format_text_content(system_prompt)]},
            {"role": "user", "content": [LocalLLM.format_text_content(user_prompt)]},
        ]

        try:
            summary = await self._llm.chat(
                messages,
                max_tokens=self._settings.summary_max_tokens,
                temperature=0.1,
                top_p=0.9,
            )
        except Exception as exc:  # pragma: no cover - local runtime failure
            logger.error("Failed to summarize document %s: %s", document.url, exc)
            return None

        return summary.strip()

    def _prepare_text(self, text: str) -> str:
        words = text.split()
        limit = self._settings.summary_chunk_size
        if len(words) <= limit:
            return text
        return " ".join(words[:limit])
