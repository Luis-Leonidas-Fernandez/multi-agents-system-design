"""Context quarantine y retry del web scraping."""
from __future__ import annotations

from typing import Any, Callable, Optional

from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig


async def _summarize_if_long(
    text: str,
    rid: str,
    get_llm_fn: Callable,
    *,
    is_retry: bool = False,
) -> str:
    from features.web_scraping.application import flow as _flow

    if len(text.split()) <= 200:
        return text

    sources_block = ""
    body_text = text
    if "Sources:" in text:
        body_text, sources_block = text.split("Sources:", 1)
        sources_block = "Sources:" + sources_block

    tags = ["web_scraping", "context_quarantine", "summary"]
    if is_retry:
        tags.append("retry")
    try:
        llm = get_llm_fn()
        summary_response = await llm.ainvoke(
            [HumanMessage(content=(
                "Resume el siguiente texto en máximo 200 palabras, "
                f"conservando los datos más importantes:\n\n{body_text[:4000]}"
            ))],
            config=RunnableConfig(
                tags=tags,
                metadata={
                    "node":              "web_scraping_node",
                    "request_id":        rid,
                    "raw_words":         len(body_text.split()),
                    "summary_triggered": True,
                },
            ),
        )
        summary = str(summary_response.content)
        if sources_block:
            summary = f"{summary.strip()}\n\n{sources_block.strip()}"
        return summary
    except Exception:
        truncated = " ".join(body_text.split()[:200])
        if sources_block:
            return f"{truncated}\n\n{sources_block.strip()}"
        return truncated


async def _run_retry_agent(
    agent,
    last_message: str,
    rid: str,
    get_llm_fn: Callable,
) -> tuple[Optional[str], list[str], dict[str, Any], dict[str, Any]]:
    from features.web_scraping.application import flow as _flow

    retry_hint = (
        f"[Sistema | auto-retry por bajo rendimiento | estrategia=force_search]\n"
        + "Usa search_web directamente — no intentes scraping de páginas.\n\n"
    )
    retry_result = await agent.ainvoke(
        {"messages": [HumanMessage(content=retry_hint + last_message)]},
        config=RunnableConfig(
            tags=["web_scraping", "agent", "high_risk", "context_quarantine", "retry"],
            metadata={
                "node":       "web_scraping_node",
                "agent":      "web_scraping_agent",
                "request_id": rid,
                "retry":      True,
            },
        ),
    )

    retry_text = _flow.extract_final_ai_text(retry_result.get("messages", []))
    if not retry_text:
        return None, [], {}, {}

    summary = await _summarize_if_long(retry_text, rid, get_llm_fn, is_retry=True)
    return (
        summary,
        retry_text.split(),
        _flow._extract_tokens(retry_result),
        _flow._extract_quality(retry_result),
    )


__all__ = ["_summarize_if_long", "_run_retry_agent"]
