from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SearchDocument(BaseModel):
    url: str
    title: str
    snippet: Optional[str] = None
    summary: Optional[str] = None


class QueryRequest(BaseModel):
    query: str = Field(..., description="User text query")
    top_k: int = Field(5, ge=1, le=12, description="Number of web results to ground the answer with")
    image_base64: Optional[str] = Field(
        None,
        description=(
            "Optional base64 encoded image to include as part of the prompt for multimodal reasoning. "
            "Should be encoded as PNG or JPEG."
        ),
    )


class QueryResponse(BaseModel):
    answer: str
    sources: List[SearchDocument]


class HealthResponse(BaseModel):
    status: str
