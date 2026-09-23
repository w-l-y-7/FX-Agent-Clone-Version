import os
from typing import Any, Dict, List

import requests

from ..core.abstractions.base_tool import BaseTool

_SEARCH_URL = "https://api.firecrawl.dev/v1/search"

# 整篇网页正文可能上万字，全塞进状态会撑爆下游 LLM 的上下文，这里只留开头
_MAX_CONTENT_CHARS = 2000


class UnstructuredDataScraper(BaseTool):
    """Searches the web for financial news and returns their text content."""

    name = "unstructured_data_scraper"
    description = (
        "Searches financial news about a query and returns the article text, "
        "useful for sentiment analysis."
    )

    def __init__(self, timeout: int = 120):
        self._timeout = timeout

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        query = kwargs.get("query", "EUR/USD exchange rate news")
        limit = int(kwargs.get("limit", 5))

        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise RuntimeError(
                "FIRECRAWL_API_KEY is not set. Add it to the .env file."
            )

        response = requests.post(
            _SEARCH_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "query": query,
                "limit": limit,
                "scrapeOptions": {
                    "formats": ["markdown"],
                    "onlyMainContent": True,
                },
            },
            timeout=self._timeout,
        )
        response.raise_for_status()

        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(f"Firecrawl search failed: {payload.get('error')}")

        return {
            "source": "firecrawl:search",
            "query": query,
            "articles": self._to_articles(payload.get("data") or []),
        }

    @staticmethod
    def _to_articles(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "title": item.get("title") or "",
                "url": item.get("url") or "",
                "description": item.get("description") or "",
                "content": (item.get("markdown") or "")[:_MAX_CONTENT_CHARS],
            }
            for item in items
        ]
