"""Contexto de selección de estrategia para web scraping."""
from __future__ import annotations

from typing import Callable, Optional

from core.domain.models import AgentState


def _select_strategy_context(state: AgentState, last_message: str, get_runtime_policy: Callable[[], dict]) -> dict:
    from application.policies.agentdog import _is_allowed_public_price_request
    from application.policies.scrape_tracker import (
        _API_VALIDATION_EPSILON,
        _detect_query_category,
        _exploration_rate,
        _get_strategy,
        _score_to_reliability,
    )

    tracker = state.get("scrape_tracker") or {}
    turn_count = (tracker.get("_turn_count") or 0) + 1
    category = _detect_query_category(last_message)
    prior_score = _get_category_score(tracker, category, turn_count)
    prior_reliability = _score_to_reliability(prior_score)

    _rt = get_runtime_policy().get(category, {})
    _top_promoted = (_rt.get("promoted") or [None])[0]
    ml_recommended: Optional[str] = _top_promoted.get("strategy") if isinstance(_top_promoted, dict) else _top_promoted

    import random

    if _is_allowed_public_price_request(state.get("messages", []), "web_scraping_node"):
        strategy, exploring = "api_price", False
        exp_rate = 0.0
    elif category == "crypto_price":
        if random.random() < _API_VALIDATION_EPSILON:
            strategy, exploring = "force_search", True
        else:
            strategy, exploring = "api_price", False
        exp_rate = _API_VALIDATION_EPSILON
    else:
        exp_rate = _exploration_rate(prior_score)
        exploring = random.random() < exp_rate
        strategy = _get_strategy(tracker, category, prior_score, exploring=exploring)

    prediction_match: Optional[bool] = strategy == ml_recommended if ml_recommended is not None else None

    return {
        "tracker": tracker,
        "turn_count": turn_count,
        "category": category,
        "prior_score": prior_score,
        "prior_reliability": prior_reliability,
        "ml_recommended": ml_recommended,
        "strategy": strategy,
        "exploring": exploring,
        "exp_rate": exp_rate,
        "prediction_match": prediction_match,
    }


def _get_category_score(tracker: dict, category: str, turn_count: int):
    from application.policies.scrape_tracker import _get_category_score as _impl

    return _impl(tracker, category, turn_count)


__all__ = ["_select_strategy_context"]
