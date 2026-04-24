from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_web_scraping_ownership_docs_exist():
    from pathlib import Path

    use_cases_doc = Path("application/use_cases/WEB_SCRAPING_OWNERSHIP.md")
    runtime_doc = Path("application/services/WEB_RUNTIME_OWNERSHIP.md")

    assert use_cases_doc.exists()
    assert runtime_doc.exists()
    assert "web_scraping_flow.py" in use_cases_doc.read_text(encoding="utf-8")
    assert "web_scraping_api.py" in use_cases_doc.read_text(encoding="utf-8")
    assert "web_runtime.py" in runtime_doc.read_text(encoding="utf-8")


def test_web_scraping_api_barrel_exports_stable_symbols():
    from features.web_scraping import api as web_scraping_api

    assert callable(web_scraping_api.run_web_scraping_flow)
    assert callable(web_scraping_api._run_generic_web_search_fetch)
    assert hasattr(web_scraping_api, "GenericWebSearchStrategy")
    assert hasattr(web_scraping_api, "CountryRecentNewsStrategy")


@pytest.mark.asyncio
async def test_web_scraping_agent_strategy_loads_prompt_from_file():
    from features.web_scraping.application.agent_strategy import _run_web_scraping_agent_strategy
    from langchain_core.messages import AIMessage

    agent = SimpleNamespace(ainvoke=AsyncMock(return_value={"messages": [AIMessage(content="respuesta " * 30)]}))
    state = {"messages": [], "next_agent": ""}

    with (
        patch("features.web_scraping.application.agent_strategy.load_agent_prompt", return_value="BASE PROMPT") as prompt_loader,
        patch("features.web_scraping.application.flow._summarize_if_long", AsyncMock(return_value="summary")),
        patch("features.web_scraping.application.flow._finalize_web_user_summary", return_value=("summary", [], ["uno", "dos"])),
        patch("features.web_scraping.application.flow._scrape_reliability", return_value="ok_strong"),
        patch("features.web_scraping.application.flow._update_scrape_tracker", return_value=({}, {})),
        patch("features.web_scraping.application.flow._get_category_score", return_value=0.5),
        patch("features.web_scraping.application.flow._emit_node_outcome"),
        patch("features.web_scraping.application.flow._extract_tokens", return_value={}),
        patch("features.web_scraping.application.flow._extract_quality", return_value={}),
        patch("features.web_scraping.application.flow._extract_followup", return_value={}),
        patch("features.web_scraping.application.flow._node_meta", return_value={}),
    ):
        result = await _run_web_scraping_agent_strategy(
            state=state,
            agent=agent,
            get_llm_fn=MagicMock(),
            last_message="Buscá noticias",
            category="general",
            tracker={},
            turn_count=1,
            prior_score=0.0,
            prior_reliability="ok",
            ml_recommended=None,
            prediction_match=None,
            rid="rid-1",
            t0=0.0,
            web_search_runtime_args={},
            should_evaluate_guard_fn=lambda *_: False,
            evaluate_trajectory_safe_fn=AsyncMock(return_value=(True, {})),
        )

    prompt_loader.assert_called_once_with("web_scraping_agent")
    assert agent.ainvoke.await_count == 1
    sent_message = agent.ainvoke.await_args.args[0]["messages"][0].content
    assert "BASE PROMPT" in sent_message
    assert result["messages"][0].content


@pytest.mark.asyncio
async def test_generic_week_strategy_keeps_partial_results_when_one_entry_fails():
    from features.web_scraping.application.generic_strategy import _run_generic_web_search_strategy_impl

    diverse_candidates = [
        {"title": "Entry one", "url": "https://example.com/one", "snippet": "one snippet has enough words for the contract"},
        {"title": "Entry two", "url": "https://example.com/two", "snippet": "two snippet has enough words for the contract"},
    ]

    async def fake_week_pipeline(*args, **kwargs):
        return diverse_candidates, "search text", None

    async def fake_gather(*args, **kwargs):
        assert kwargs.get("return_exceptions") is True
        return [RuntimeError("boom"), ("Entry two", "two content has enough words for the contract", False)]

    with (
        patch("features.web_scraping.application.generic_strategy.asyncio.gather", new=fake_gather),
        patch("features.web_scraping.application.flow._run_week_search_pipeline", new=fake_week_pipeline),
        patch("features.web_scraping.application.flow._is_article_url", return_value=True),
        patch("features.web_scraping.application.flow._extract_generic_content_lines", return_value=["line one", "line two", "line three"]),
        patch("features.web_scraping.application.flow._build_source_backed_response", side_effect=lambda lines, sources: "\n".join(lines)),
        patch("features.web_scraping.application.flow._extract_sources_from_text", return_value=[]),
        patch("features.web_scraping.application.flow._is_no_info_response", return_value=False),
        patch("features.web_scraping.application.flow._web_debug"),
        patch("features.web_scraping.application.flow._clean_source_url", side_effect=lambda url: url),
    ):
        result = await _run_generic_web_search_strategy_impl("dame noticias de esta semana")

    assert result is not None
    assert "Entry one" in result["summary"]
    assert "Entry two" in result["summary"]


def test_generic_strategy_exposes_compatibility_class():
    from features.web_scraping.application.generic_strategy import GenericWebSearchStrategy

    strategy = GenericWebSearchStrategy(search_runtime=object(), fetch_runtime=object())
    assert hasattr(strategy, "execute")
