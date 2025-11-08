from __future__ import annotations

import asyncio
import base64
import binascii
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_cpp import Llama

from app.config import Settings

logger = logging.getLogger(__name__)


class LocalLLM:
    """Wrapper around llama.cpp for multimodal chat completions."""

    def __init__(self, settings: Settings):
        self._settings = settings
        model_path = settings.llm_model_path.expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(
                f"Local model not found at {model_path}. Please run scripts/download_model.py"
            )

        mmproj_path: Optional[Path] = None
        if settings.llm_projector_path:
            resolved = settings.llm_projector_path.expanduser().resolve()
            if resolved.exists():
                mmproj_path = resolved
            else:
                logger.warning("Configured projector path %s does not exist", resolved)

        kwargs: Dict[str, Any] = {
            "model_path": str(model_path),
            "n_ctx": settings.llm_context_window,
            "n_threads": settings.llm_threads,
            "n_batch": settings.llm_batch_size,
            "seed": 0,
            "verbose": False,
        }

        if settings.enable_metal_acceleration:
            kwargs["n_gpu_layers"] = settings.llm_gpu_layers
        if mmproj_path:
            kwargs["mmproj_path"] = str(mmproj_path)

        self._llama = Llama(**kwargs)

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        completion_kwargs = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature if temperature is not None else self._settings.llm_temperature,
            "top_p": top_p if top_p is not None else self._settings.llm_top_p,
        }

        logger.debug("Dispatching chat completion with %d messages", len(messages))
        response = await asyncio.to_thread(self._llama.create_chat_completion, **completion_kwargs)
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError("No choices returned by local LLM")
        return choices[0]["message"]["content"].strip()

    @staticmethod
    def format_text_content(text: str) -> Dict[str, str]:
        return {"type": "text", "text": text}

    @staticmethod
    def format_image_content(image_base64: str) -> Dict[str, Any]:
        try:
            image_bytes = base64.b64decode(image_base64, validate=True)
        except binascii.Error as exc:
            raise ValueError("Invalid base64 supplied for image content") from exc
        return {
            "type": "image",
            "image": {
                "id": 0,
                "data": image_bytes,
            },
        }
