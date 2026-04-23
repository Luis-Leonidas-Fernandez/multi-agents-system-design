"""Runtime liviano para búsqueda y fetch web inspirado en OpenClaw."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import HumanMessage

from application.services.web_fetch_registry import (
    get_web_fetch_provider_spec,
    resolve_web_fetch_provider_name,
)
from application.services.web_search_registry import (
    get_web_search_provider_spec,
    resolve_web_search_provider_candidates,
)
from application.services.web_runtime_helpers import _build_web_search_payload, _classify_web_fetch_status
from application.policies.security_flow import input_guard


@dataclass(frozen=True)
class WebSearchRequest:
    query: str
    allowed_domains: Optional[list[str]] = None
    blocked_domains: Optional[list[str]] = None
    max_age_days: Optional[int] = None
    topic: Optional[str] = None
    time_range: Optional[str] = None
    count: int = 8
    provider: Optional[str] = None
    runtime_selected_provider: Optional[str] = None
    runtime_provider_configured: Optional[str] = None
    use_cache: bool = False


@dataclass(frozen=True)
class WebSearchHit:
    title: str
    url: str
    snippet: str
    hit_type: str


@dataclass(frozen=True)
class WebSearchResponse:
    provider_name: str
    raw_text: str
    hits: tuple[WebSearchHit, ...]


@dataclass(frozen=True)
class WebFetchRequest:
    url: str
    prompt: str
    mode: str = "dynamic"
    use_cache: bool = False


@dataclass(frozen=True)
class WebFetchResponse:
    provider_name: str
    url: str
    content: str
    fetch_kind: str
    status: str


def _extract_hits_from_search_text(raw_text: str) -> tuple[WebSearchHit, ...]:
    from application.use_cases.web_scraping_flow import _extract_generic_search_candidates

    hits = []
    for row in _extract_generic_search_candidates(raw_text):
        hits.append(
            WebSearchHit(
                title=str(row.get("title") or row.get("url") or "result"),
                url=str(row.get("url") or ""),
                snippet=str(row.get("snippet") or ""),
                hit_type=str(row.get("hit_type") or ""),
            )
        )
    return tuple(hits)


class WebSearchRuntime:
    """Resuelve provider y ejecuta búsqueda con un contrato estable."""

    async def search(self, request: WebSearchRequest) -> WebSearchResponse:
        from tools.search_tools import search_web

        provider_candidates = resolve_web_search_provider_candidates(
            explicit_provider=request.provider,
            runtime_selected_provider=request.runtime_selected_provider,
            runtime_provider_configured=request.runtime_provider_configured,
        )
        selected_provider = provider_candidates[0]
        payload = _build_web_search_payload(request, selected_provider.name)
        search_fn = getattr(search_web, "func", None)
        if callable(search_fn):
            raw_text = str(search_fn(**payload))
        else:
            raw_text = str(search_web.invoke(payload))
        return WebSearchResponse(
            provider_name=get_web_search_provider_spec(selected_provider.name).name,
            raw_text=raw_text,
            hits=_extract_hits_from_search_text(raw_text),
        )


class WebFetchRuntime:
    """Resuelve provider y ejecuta fetch con un contrato estable."""

    async def fetch(self, request: WebFetchRequest) -> WebFetchResponse:
        from tools.scraping_tools import fetch_web_page

        provider_name = resolve_web_fetch_provider_name()
        guard_result = input_guard({"messages": [HumanMessage(content=f"URL: {request.url}\n\nPrompt: {request.prompt}")]})
        if isinstance(guard_result, dict) and guard_result.get("blocked"):
            blocked_content = getattr(guard_result["messages"][0], "content", str(guard_result["messages"][0])) if guard_result.get("messages") else "Solicitud bloqueada por política de seguridad."
            return WebFetchResponse(
                provider_name=get_web_fetch_provider_spec(provider_name).name,
                url=request.url,
                content=blocked_content,
                fetch_kind="dynamic" if request.mode != "static" else "static",
                status="rejected",
            )

        use_dynamic = request.mode != "static"
        content = await fetch_web_page(
            url=request.url,
            prompt=request.prompt,
            use_dynamic=use_dynamic,
            use_cache=request.use_cache,
        )
        return WebFetchResponse(
            provider_name=get_web_fetch_provider_spec(provider_name).name,
            url=request.url,
            content=content,
            fetch_kind="dynamic" if use_dynamic else "static",
            status=_classify_web_fetch_status(content),
        )


__all__ = [
    "WebFetchRequest",
    "WebFetchResponse",
    "WebFetchRuntime",
    "WebSearchHit",
    "WebSearchRequest",
    "WebSearchResponse",
    "WebSearchRuntime",
]
