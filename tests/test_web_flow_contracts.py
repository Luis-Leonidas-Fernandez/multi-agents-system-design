from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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
