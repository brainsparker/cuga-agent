import json

import httpx
import pytest

from cuga.backend.tools_env.registry.registry import api_registry as api_registry_module
from cuga.backend.tools_env.registry.registry.youcom_search import (
    FREE_URL,
    KEYED_URL,
    USER_AGENT,
    YouComSearchClient,
    YouComSearchError,
)

PAYLOAD = {
    "results": {
        "web": [
            {"url": "https://a.example", "title": "A", "description": "desc a", "snippets": ["s"]},
            {"url": "https://b.example", "title": "B", "description": "", "snippets": ["one", "two"]},
            {"url": "", "title": "no url"},
        ]
    }
}


def _transport(seen, status=200, body=None, headers=None):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=PAYLOAD if body is None else body, headers=headers or {})

    return httpx.MockTransport(handler)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_keyless_get_maps_to_tavily_shape():
    seen = []
    client = YouComSearchClient(api_key="", transport=_transport(seen))

    result = await client.search("cuga agent")

    request = seen[0]
    assert request.method == "GET"
    assert str(request.url).startswith(FREE_URL)
    assert request.url.params["query"] == "cuga agent"
    assert request.url.params["count"] == "5"
    assert request.headers["user-agent"] == USER_AGENT
    assert "x-api-key" not in request.headers
    assert result["query"] == "cuga agent"
    assert [r["url"] for r in result["results"]] == ["https://a.example", "https://b.example"]
    assert result["results"][0]["content"] == "desc a"
    assert result["results"][1]["content"] == "one two"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_keyed_post_uses_api_key_header(monkeypatch):
    monkeypatch.setenv("YDC_API_KEY", "k-123")
    seen = []
    client = YouComSearchClient(transport=_transport(seen))

    await client.search("q")

    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == KEYED_URL
    assert request.headers["x-api-key"] == "k-123"
    assert json.loads(request.content) == {"query": "q", "count": 5}


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,needle",
    [(401, "unauthorized"), (403, "unauthorized"), (402, "out of credits"), (429, "rate limited")],
)
async def test_keyed_errors_are_explained(status, needle):
    client = YouComSearchClient(api_key="k", transport=_transport([], status=status, body={}))

    with pytest.raises(YouComSearchError, match=needle) as exc:
        await client.search("q")
    assert exc.value.status_code == status


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rate_limit_carries_retry_after():
    client = YouComSearchClient(
        api_key="", transport=_transport([], status=429, body={}, headers={"Retry-After": "30"})
    )

    with pytest.raises(YouComSearchError, match="retry after 30s") as exc:
        await client.search("q")
    assert exc.value.retry_after == 30


@pytest.mark.unit
@pytest.mark.asyncio
async def test_keyless_quota_message_points_to_key():
    client = YouComSearchClient(api_key="", transport=_transport([], status=402, body={}))

    with pytest.raises(YouComSearchError, match="YDC_API_KEY"):
        await client.search("q")


class _FakeManager:
    def get_apps(self):
        return []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_registry_falls_back_to_youcom_without_tavily_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("YDC_API_KEY", raising=False)
    monkeypatch.setattr(api_registry_module.ApiRegistry, "_is_web_search_enabled", lambda self: True)

    registry = api_registry_module.ApiRegistry(client=_FakeManager())

    assert registry.tavily_client is None
    assert isinstance(registry.youcom_client, YouComSearchClient)
    apps = await registry.show_applications()
    assert apps[-1].description == "Web search tool powered by You.com"

    async def fake_search(query):
        return {"query": query, "results": []}

    monkeypatch.setattr(registry.youcom_client, "search", fake_search)
    result = await registry.call_function("web", "search_web", {"query": "hi"})
    assert json.loads(result[0].text) == {"query": "hi", "results": []}


@pytest.mark.unit
def test_registry_keeps_tavily_when_key_set(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-x")
    monkeypatch.setattr(api_registry_module.ApiRegistry, "_is_web_search_enabled", lambda self: True)
    monkeypatch.setattr(api_registry_module, "TAVILY_AVAILABLE", True)
    monkeypatch.setattr(api_registry_module, "TavilyClient", lambda key: object())

    registry = api_registry_module.ApiRegistry(client=_FakeManager())

    assert registry.tavily_client is not None
    assert registry.youcom_client is None
