# Scarper – Local Multimodal Research Backend

Scarper pairs a performant local multimodal LLM with real-time web search, high
quality scraping, and targeted summarisation. The service is designed as a
standalone backend that can be connected to [Pluely](https://pluely.ai/) or any
other orchestration layer capable of issuing HTTP requests. It focuses on
keeping Apple Silicon (M1 Pro) hardware responsive while still relying on an LLM
large enough to deliver dependable answers.

## Highlights

- **LLaVA v1.6 Mistral 7B (Q4_K_M)** – a well-reviewed multimodal model that
  balances quality, hallucination resistance, and performance on M1 Pro chips.
- **Metal acceleration ready** – defaults to leveraging GPU layers with
  `llama-cpp-python` for best throughput on Apple Silicon.
- **Multimodal input** – the `/query` endpoint accepts both text and optional
  base64-encoded images.
- **Real-time Brave search** – uses Brave's Search API for live, privacy
  friendly results.
- **Best-in-class scraping** – integrates the open-source
  [Trafilatura](https://github.com/adbar/trafilatura) extractor to retrieve clean
  article text, paired with resilient HTTP fetching.
- **Automatic summarisation** – each fetched document is summarised before being
  injected into the LLM prompt, preventing context overflow and keeping the
  model focused.
- **Source-aware answers** – final responses include inline citations that map to
  the supporting documents.

## Project Structure

```
app/
  config.py                # Pydantic settings (env-driven)
  main.py                  # FastAPI application
  models.py                # Request/response schemas
  pipelines/
    retrieval_augmented.py # Retrieval + summarisation + generation pipeline
  services/
    brave_search.py        # Brave Search API client
    llm_runtime.py         # llama.cpp runtime wrapper
    scraper.py             # Web scraping via Trafilatura
    summarizer.py          # LLM-powered summarisation helper
scripts/
  download_model.py        # Utility to fetch the recommended LLaVA weights
```

## Getting Started

### 1. Install dependencies (Apple Silicon / M1 Pro)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Install llama-cpp-python with Metal acceleration
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
pip install -e .
```

The explicit installation of `llama-cpp-python` with `LLAMA_METAL=on` ensures
Metal layers are available, preventing CPU bottlenecks on M1 Pro machines.

### 2. Download the multimodal model

```bash
python scripts/download_model.py --target ./models
```

By default the service looks for:

- `./models/llava-v1.6-mistral-7b.Q4_K_M.gguf`
- `./models/llava-v1.6-mistral-7b-mmproj.gguf`

You can override these via environment variables if you prefer a different
location or quantisation.

### 3. Configure environment variables

Copy the example configuration and fill in your Brave API key:

```bash
cp .env.example .env
```

Edit `.env` and set at least:

```
SCARPER_BRAVE_API_KEY=your_brave_key_here
```

Optional overrides:

- `SCARPER_LLM_MODEL_PATH`
- `SCARPER_LLM_PROJECTOR_PATH`
- `SCARPER_ENABLE_METAL_ACCELERATION` (set to `false` to force CPU-only)
- `SCARPER_BRAVE_RESULT_COUNT`

### 4. Run the service

```bash
uvicorn app.main:app --reload --port 8080
```

The API documentation is automatically exposed at `http://localhost:8080/docs`.

## API Overview

### `GET /health`

Simple readiness probe returning `{"status": "ok"}`.

### `POST /query`

Executes the full pipeline: Brave search → scraping → summarisation → local LLM
response.

**Request body**

```json
{
  "query": "What is the latest research on quantum dot solar cells?",
  "top_k": 5,
  "image_base64": null
}
```

- `query` (string, required) – natural language question.
- `top_k` (integer, optional) – number of search results to ingest (default 5,
  capped by `SCARPER_BRAVE_RESULT_COUNT`).
- `image_base64` (string, optional) – base64 image (with or without a Data URL
  prefix) for multimodal grounding.

**Response body**

```json
{
  "answer": "...",
  "sources": [
    {
      "url": "https://example.com/...",
      "title": "Example title",
      "snippet": "Search snippet",
      "summary": "Condensed context used by the LLM"
    }
  ]
}
```

### Integrating with Pluely

Pluely (or any local orchestration layer) can treat Scarper as a research tool:

1. Capture the user message and optional image inside Pluely.
2. Forward the payload to `POST http://localhost:8080/query`.
3. Insert the returned `answer` and `sources` back into the Pluely conversation
   flow, or use the summaries as grounding data for additional prompts.

The endpoint is synchronous and designed for local-first operation; you can wrap
it in WebSockets or task queues if tighter coupling is required.

## Design Notes

- **Model choice**: LLaVA v1.6 Mistral 7B (Q4_K_M) is widely benchmarked, offers
  strong multimodal reasoning, and remains performant on an M1 Pro when Metal
  acceleration is enabled. The quantisation keeps VRAM usage manageable without
  heavily impacting answer quality.
- **Scraping quality**: Trafilatura is battle-tested for structured text
  extraction, minimising boilerplate noise before the summarisation stage.
- **Summarisation guardrail**: Summaries cap the amount of text the model needs
  to ingest, which keeps prompts well within the 8K context window of the chosen
  GGUF build and reduces hallucination risk.
- **Citations**: The final prompt template instructs the LLM to reference sources
  explicitly (e.g., `[1]`) so downstream consumers can map statements back to
  the original URLs.

## Environment Variables

All configuration is driven by environment variables with the `SCARPER_` prefix.
Relevant options are documented in `app/config.py` and surfaced through
`.env.example`.

## Testing & Further Work

- Unit tests can be added under `tests/` (framework hooks already in
  `pyproject.toml`).
- For heavy workloads, consider running Scarper behind a queue and pooling model
  workers to serialise access to the LLaVA weights.
- Additional scrapers (e.g. Playwright-based solutions) can be layered on for
  highly dynamic sites.

## License

MIT – see `pyproject.toml` for details.
