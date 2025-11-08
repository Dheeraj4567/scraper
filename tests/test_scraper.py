import pytest

from app.config import Settings
from app.services.scraper import WebScraperService


@pytest.fixture()
def scraper_service() -> WebScraperService:
    settings = Settings(brave_api_key="test-key")
    return WebScraperService(settings)


def test_readability_fallback_extracts_article(scraper_service: WebScraperService) -> None:
    html = """
    <html>
      <head><title>Example Site</title></head>
      <body>
        <header><h1>Landing Page</h1></header>
        <article>
          <h1>Breakthrough Discovery</h1>
          <p>Scientists announced a major breakthrough in clean energy.</p>
          <p>The findings were published in a peer-reviewed journal.</p>
        </article>
      </body>
    </html>
    """

    text, title = scraper_service._fallback_extract(html)  # pylint: disable=protected-access

    assert text is not None
    assert "clean energy" in text
    assert title in {"Breakthrough Discovery", "Example Site"}


def test_beautifulsoup_fallback_used_when_readability_empty(
    scraper_service: WebScraperService, monkeypatch: pytest.MonkeyPatch
) -> None:
    html = """
    <html>
      <head><title>Fallback Title</title></head>
      <body>
        <div>Some <span>basic</span> text.</div>
      </body>
    </html>
    """

    async def fake_readability(self, content: str):  # pylint: disable=unused-argument
        return None, None

    monkeypatch.setattr(
        WebScraperService,
        "_readability_extract",
        lambda self, _: (None, None),
    )

    text, title = scraper_service._fallback_extract(html)  # pylint: disable=protected-access

    assert text == "Some basic text."
    assert title == "Fallback Title"
