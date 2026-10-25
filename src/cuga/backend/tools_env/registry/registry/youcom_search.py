"""You.com web search, used when Tavily isn't configured.

No key: GET https://api.you.com/v1/agents/search (free tier, 100 searches/day per IP).
With ``YDC_API_KEY``: POST https://ydc-index.io/v1/search with the ``X-API-Key`` header.

Results are mapped to the Tavily response shape so ``search_web`` keeps one schema.
"""

import os
import time
from typing import Any, Dict, Optional

import httpx

FREE_URL = "https://api.you.com/v1/agents/search"
KEYED_URL = "https://ydc-index.io/v1/search"
USER_AGENT = "youdotcom-integration/cuga-project-cuga-agent"
DEFAULT_COUNT = 5


class YouComSearchError(Exception):
    """Non-2xx answer from You.com. ``retry_after`` is set (seconds) when the server sends it."""

    def __init__(self, message: str, status_code: int, retry_after: Optional[float] = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


def _retry_after(response: httpx.Response) -> Optional[float]:
    try:
        value = float(response.headers.get("retry-after", ""))
    except ValueError:
        return None
    return value if value > 0 else None


def _raise_for_status(response: httpx.Response, keyed: bool) -> None:
    code = response.status_code
    if code < 400:
        return
    if code in (401, 403):
        reason = "invalid or unauthorized YDC_API_KEY" if keyed else "access denied by the keyless tier"
    elif code == 402:
        reason = "out of credits" if keyed else "free daily quota used up; set YDC_API_KEY for more"
    elif code == 429:
        reason = "rate limited"
    else:
        reason = f"HTTP {code}"
    retry_after = _retry_after(response)
    suffix = f", retry after {retry_after:g}s" if retry_after else ""
    raise YouComSearchError(f"You.com search failed: {reason}{suffix}", code, retry_after)


def _to_tavily_shape(query: str, payload: Dict[str, Any], elapsed: float) -> Dict[str, Any]:
    web = (payload.get("results") or {}).get("web") or []
    results = []
    for item in web:
        url = item.get("url")
        if not url:
            continue
        content = item.get("description") or " ".join(item.get("snippets") or [])
        results.append(
            {
                "url": url,
                "title": item.get("title") or "",
                "content": content,
                "score": None,
                "raw_content": None,
            }
        )
    return {
        "query": query,
        "follow_up_questions": None,
        "answer": None,
        "images": [],
        "results": results,
        "response_time": round(elapsed, 3),
    }


class YouComSearchClient:
    """Small async client. Keyless unless ``YDC_API_KEY`` is set."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        count: int = DEFAULT_COUNT,
        timeout: float = 20.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.api_key = api_key if api_key is not None else os.getenv("YDC_API_KEY") or None
        self.count = count
        self.timeout = timeout
        self._transport = transport

    @property
    def keyed(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str) -> Dict[str, Any]:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        started = time.monotonic()
        async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
            if self.keyed:
                headers["X-API-Key"] = self.api_key
                response = await client.post(
                    KEYED_URL, json={"query": query, "count": self.count}, headers=headers
                )
            else:
                response = await client.get(
                    FREE_URL, params={"query": query, "count": self.count}, headers=headers
                )
        _raise_for_status(response, self.keyed)
        return _to_tavily_shape(query, response.json(), time.monotonic() - started)
