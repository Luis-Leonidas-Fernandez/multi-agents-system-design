from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_web_scraping_ownership_docs_exist():
    from pathlib import Path

    assert Path("application/use_cases/WEB_SCRAPING_OWNERSHIP.md").exists()
    assert Path("application/services/WEB_RUNTIME_OWNERSHIP.md").exists()


@pytest.mark.asyncio
async def test_web_scraping_agent_strategy_loads_prompt_from_file():
    from application.use_cases.web_scraping_agent_strategy import _run_web_scraping_agent_strategy
    from langchain_core.messages import AIMessage

    agent = SimpleNamespace(ainvoke=AsyncMock(return_value={"messages": [AIMessage(content="respuesta " * 30)]}))
    state = {"messages": [], "next_agent": ""}

    with (
        patch("application.use_cases.web_scraping_agent_strategy.load_agent_prompt", return_value="BASE PROMPT") as prompt_loader,
        patch("application.use_cases.web_scraping_flow._summarize_if_long", AsyncMock(return_value="summary")),
        patch("application.use_cases.web_scraping_flow._finalize_web_user_summary", return_value=("summary", [], ["uno", "dos"])),
        patch("application.use_cases.web_scraping_flow._scrape_reliability", return_value="ok_strong"),
        patch("application.use_cases.web_scraping_flow._update_scrape_tracker", return_value=({}, {})),
        patch("application.use_cases.web_scraping_flow._get_category_score", return_value=0.5),
        patch("application.use_cases.web_scraping_flow._emit_node_outcome"),
        patch("application.use_cases.web_scraping_flow._extract_tokens", return_value={}),
        patch("application.use_cases.web_scraping_flow._extract_quality", return_value={}),
        patch("application.use_cases.web_scraping_flow._extract_followup", return_value={}),
        patch("application.use_cases.web_scraping_flow._node_meta", return_value={}),
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
    from application.use_cases.web_scraping_generic_strategy import _run_generic_web_search_strategy_impl

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
        patch("application.use_cases.web_scraping_generic_strategy.asyncio.gather", new=fake_gather),
        patch("application.use_cases.web_scraping_flow._run_week_search_pipeline", new=fake_week_pipeline),
        patch("application.use_cases.web_scraping_flow._is_article_url", return_value=True),
        patch("application.use_cases.web_scraping_flow._extract_generic_content_lines", return_value=["line one", "line two", "line three"]),
        patch("application.use_cases.web_scraping_flow._build_source_backed_response", side_effect=lambda lines, sources: "\n".join(lines)),
        patch("application.use_cases.web_scraping_flow._extract_sources_from_text", return_value=[]),
        patch("application.use_cases.web_scraping_flow._is_no_info_response", return_value=False),
        patch("application.use_cases.web_scraping_flow._web_debug"),
        patch("application.use_cases.web_scraping_flow._clean_source_url", side_effect=lambda url: url),
    ):
        result = await _run_generic_web_search_strategy_impl("dame noticias de esta semana")

    assert result is not None
    assert "Entry one" in result["summary"]
    assert "Entry two" in result["summary"]


def test_generic_strategy_exposes_compatibility_class():
    from application.use_cases.web_scraping_generic_strategy import GenericWebSearchStrategy

    strategy = GenericWebSearchStrategy(search_runtime=object(), fetch_runtime=object())
    assert hasattr(strategy, "execute")
