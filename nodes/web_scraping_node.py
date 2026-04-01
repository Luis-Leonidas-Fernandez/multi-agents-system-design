"""
Nodo del agente de web scraping con HITL + context quarantine.
Factory pattern: make_web_scraping_node(agent, get_llm_fn) → async web_scraping_node(state).

Patrón context quarantine: el sub-agente absorbe el HTML/texto crudo en su
propio contexto aislado. Solo devuelve al estado compartido un resumen de ≤200
palabras, evitando contaminar el historial del supervisor con miles de tokens.
"""
import random
import re
import time
import uuid
from typing import Callable, Awaitable, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig

from audit import (
    _emit_node_outcome,
    _extract_tokens,
    _extract_quality,
    _extract_followup,
    _node_meta,
    _get_model_name,
)
from agentdog import evaluate_trajectory_safe, _should_evaluate_guard
from security import _HITL_ENABLED, _ask_confirmation
from scrape_tracker import (
    _detect_query_category,
    _get_category_score,
    _score_to_reliability,
    _get_strategy,
    _STRATEGY_HINTS,
    _API_VALIDATION_EPSILON,
    _exploration_rate,
    _update_scrape_tracker,
    _STRUCTURED_SOURCE_STRATEGIES,
    _RETRY_ON_RELIABILITY,
    _scrape_reliability,
    get_runtime_policy,
)
from price_helpers import (
    _detect_coin_from_query,
    _format_price_response,
    _extract_price_from_messages,
    _extract_structured_price,
    _get_crypto_price_fn,
)
from state import AgentState


# ==================== HELPERS ====================

def _select_strategy_context(state: AgentState, last_message: str) -> dict:
    """Selecciona la estrategia de scraping y retorna el contexto completo."""
    tracker    = state.get("scrape_tracker") or {}
    turn_count = (tracker.get("_turn_count") or 0) + 1
    category   = _detect_query_category(last_message)
    prior_score       = _get_category_score(tracker, category, turn_count)
    prior_reliability = _score_to_reliability(prior_score)

    _rt           = get_runtime_policy().get(category, {})
    _top_promoted = (_rt.get("promoted") or [None])[0]
    ml_recommended: Optional[str] = (
        _top_promoted.get("strategy") if isinstance(_top_promoted, dict) else _top_promoted
    )

    if category == "crypto_price":
        if random.random() < _API_VALIDATION_EPSILON:
            strategy, exploring = "force_search", True
        else:
            strategy, exploring = "api_price", False
        exp_rate = _API_VALIDATION_EPSILON
    else:
        exp_rate  = _exploration_rate(prior_score)
        exploring = random.random() < exp_rate
        strategy  = _get_strategy(tracker, category, prior_score, exploring=exploring)

    prediction_match: Optional[bool] = (
        (strategy == ml_recommended) if ml_recommended is not None else None
    )

    return {
        "tracker": tracker, "turn_count": turn_count, "category": category,
        "prior_score": prior_score, "prior_reliability": prior_reliability,
        "ml_recommended": ml_recommended, "strategy": strategy,
        "exploring": exploring, "exp_rate": exp_rate, "prediction_match": prediction_match,
    }


async def _summarize_if_long(
    text: str, rid: str, get_llm_fn: Callable, *, is_retry: bool = False
) -> str:
    """Retorna un resumen de ≤200 palabras si el texto lo supera; si no, lo retorna sin cambios."""
    if len(text.split()) <= 200:
        return text
    tags = ["web_scraping", "context_quarantine", "summary"]
    if is_retry:
        tags.append("retry")
    llm = get_llm_fn()
    summary_response = await llm.ainvoke(
        [HumanMessage(content=(
            "Resume el siguiente texto en máximo 200 palabras, "
            f"conservando los datos más importantes:\n\n{text[:4000]}"
        ))],
        config=RunnableConfig(
            tags=tags,
            metadata={
                "node":              "web_scraping_node",
                "request_id":        rid,
                "raw_words":         len(text.split()),
                "summary_triggered": True,
            },
        ),
    )
    return summary_response.content


async def _run_retry_agent(
    agent,
    last_message: str,
    rid: str,
    get_llm_fn: Callable,
) -> tuple:
    """
    Reintenta la invocación del agente con estrategia force_search.
    Retorna (summary, words, tokens, quality) con los datos del retry,
    o (None, None, None, None) si el retry no produjo texto.
    """
    retry_hint = (
        f"[Sistema | auto-retry por bajo rendimiento | estrategia=force_search]\n"
        + _STRATEGY_HINTS["force_search"]
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

    retry_text = ""
    for msg in reversed(retry_result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            retry_text = msg.content
            break

    if not retry_text:
        return None, None, None, None

    summary = await _summarize_if_long(retry_text, rid, get_llm_fn, is_retry=True)
    return (
        summary,
        retry_text.split(),
        _extract_tokens(retry_result),
        _extract_quality(retry_result),
    )


# ==================== FACTORY ====================

def make_web_scraping_node(
    agent,
    get_llm_fn: Callable,
) -> Callable[[AgentState], Awaitable[AgentState]]:
    """
    Retorna web_scraping_node con agente y factory de LLM inyectados como closure.

    Args:
        agent: web_scraping_agent (create_react_agent instance)
        get_llm_fn: callable sin args que retorna el LLM configurado (get_llm())
    """

    async def web_scraping_node(state: AgentState) -> AgentState:
        messages     = state["messages"]
        last_message = messages[-1].content if messages else ""
        rid          = state.get("request_id", str(uuid.uuid4()))
        t0           = time.time()

        if _HITL_ENABLED:
            urls     = re.findall(r'https?://\S+', last_message)
            url_info = f" → URLs: {', '.join(urls)}" if urls else ""
            preview  = last_message[:120] + ("..." if len(last_message) > 120 else "")
            confirmed = await _ask_confirmation(
                f"\n[HITL] web_scraping_agent va a procesar: \"{preview}\"{url_info}\n¿Confirmar? [s/n]: "
            )
            if not confirmed:
                _emit_node_outcome(
                    rid, "web_scraping_node", "blocked", phase="pre_guard",
                    agent="web_scraping_agent",
                    duration_ms=int((time.time() - t0) * 1000),
                    reason="hitl_rejected",
                )
                return {"messages": [AIMessage(content="Operación cancelada por el usuario.")]}

        try:
            ctx = _select_strategy_context(state, last_message)
            tracker          = ctx["tracker"]
            turn_count       = ctx["turn_count"]
            category         = ctx["category"]
            prior_score      = ctx["prior_score"]
            prior_reliability = ctx["prior_reliability"]
            ml_recommended   = ctx["ml_recommended"]
            strategy         = ctx["strategy"]
            exploring        = ctx["exploring"]
            exp_rate         = ctx["exp_rate"]
            prediction_match = ctx["prediction_match"]

            # ================================================================
            # FAST PATH: api_price → llamada directa sin overhead de LLM.
            # ================================================================
            if strategy == "api_price":
                from models import PriceToolResponse
                coin     = _detect_coin_from_query(last_message)
                api_json: Optional[str] = None
                try:
                    import asyncio
                    loop     = asyncio.get_running_loop()
                    api_json = await loop.run_in_executor(
                        None, lambda: _get_crypto_price_fn(coin=coin, vs_currency="usd")
                    )
                except Exception:
                    pass  # fallback al agente normal

                if api_json:
                    try:
                        price_resp = PriceToolResponse.model_validate_json(api_json)
                    except Exception:
                        price_resp = None

                    if price_resp and price_resp.is_valid_price():
                        formatted   = _format_price_response(price_resp.model_dump())
                        duration_ms = int((time.time() - t0) * 1000)
                        tokens_fast = {
                            "model": _get_model_name(), "tokens_available": False,
                            "prompt_tokens": 0, "completion_tokens": 0,
                            "total_tokens": 0, "estimated_cost_usd": 0.0,
                        }
                        quality_fast  = {"output_length": len(formatted), "tool_calls_count": 1}
                        followup_fast = {"followup_likely": False}
                        meta          = _node_meta()

                        new_tracker, analytics = _update_scrape_tracker(
                            tracker, category, 200, turn_count,
                            duration_ms=duration_ms, cost_usd=0.0,
                            source_type="structured", reliability_override="ok_strong",
                        )
                        new_score          = _get_category_score(new_tracker, category, turn_count)
                        quality_target_val = analytics.get("quality_target", 0)
                        ml_would_succeed: Optional[bool] = (
                            bool(quality_target_val) if prediction_match is True else None
                        )
                        _emit_node_outcome(
                            rid, "web_scraping_node", "success", phase="agent",
                            agent="web_scraping_agent", duration_ms=duration_ms,
                            category=category, strategy=strategy, exploring=exploring,
                            exp_rate=exp_rate, source_type="structured",
                            price_extracted=price_resp.price, parse_success=True,
                            scrape_reliability="ok_strong",
                            prior_reliability=prior_reliability, prior_score=prior_score,
                            scrape_score=new_score, retry_done=False,
                            ml_recommended=ml_recommended, prediction_match=prediction_match,
                            ml_would_succeed=ml_would_succeed,
                            **tokens_fast, **quality_fast, **followup_fast, **analytics, **meta,
                        )
                        return {
                            "messages": [AIMessage(content=formatted)],
                            "scrape_tracker": new_tracker,
                        }
                    # price_resp inválido → continuar con el agente normal como fallback

            agent_hint = _STRATEGY_HINTS[strategy]
            if agent_hint:
                agent_hint = (
                    f"[Sistema | categoría={category} score={prior_score:+.2f} "
                    f"estrategia={strategy} exploring={exploring} exp_rate={exp_rate:.0%}]\n{agent_hint}"
                )
            agent_message = agent_hint + last_message

            # --- Fase 1: sub-agente extrae contenido crudo (contexto aislado) ---
            raw_result = await agent.ainvoke(
                {"messages": [HumanMessage(content=agent_message)]},
                config=RunnableConfig(
                    tags=["web_scraping", "agent", "high_risk", "context_quarantine"],
                    metadata={
                        "node":              "web_scraping_node",
                        "agent":             "web_scraping_agent",
                        "request_id":        rid,
                        "input_chars":       len(last_message),
                        "prior_reliability": prior_reliability,
                    },
                ),
            )

            tokens   = _extract_tokens(raw_result)
            quality  = _extract_quality(raw_result)
            followup = _extract_followup(raw_result, "success")
            meta     = _node_meta()

            # --- Guardrail AgentDoG: solo la trayectoria del sub-agente ---
            if _should_evaluate_guard("web_scraping_node"):
                is_safe, _ = await evaluate_trajectory_safe(
                    {
                        "messages":   raw_result.get("messages", []),
                        "next_agent": state.get("next_agent", ""),
                    },
                    "web_scraping_node",
                )
                if not is_safe:
                    _emit_node_outcome(
                        rid, "web_scraping_node", "blocked", phase="post_guard",
                        agent="web_scraping_agent",
                        duration_ms=int((time.time() - t0) * 1000),
                        reason="agentdog",
                        followup_likely=True,
                        **tokens, **quality, **meta,
                    )
                    return {"messages": [AIMessage(content="Respuesta retenida por política de seguridad.")]}

            # --- Fase 2: extraer solo la respuesta final del sub-agente ---
            raw_messages = raw_result.get("messages", [])
            raw_text = ""
            for msg in reversed(raw_messages):
                if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                    raw_text = msg.content
                    break

            if not raw_text:
                _emit_node_outcome(
                    rid, "web_scraping_node", "error", phase="agent",
                    agent="web_scraping_agent",
                    duration_ms=int((time.time() - t0) * 1000),
                    reason="empty_response",
                    followup_likely=True,
                    **meta,
                )
                return {"messages": [AIMessage(content="No se pudo extraer información de la página.")]}

            # --- Fase 3: resumir a ≤200 palabras ---
            summary           = await _summarize_if_long(raw_text, rid, get_llm_fn)
            words             = raw_text.split()
            summary_triggered = len(words) > 200

            duration_ms = int((time.time() - t0) * 1000)
            reliability = _scrape_reliability(len(words))
            retry_done  = False

            source_type   = "structured" if strategy in _STRUCTURED_SOURCE_STRATEGIES else "unstructured"
            parsed_price: Optional[float] = None
            parse_success: Optional[bool] = None
            price_data: Optional[dict]    = None

            if source_type == "structured":
                price_data = _extract_price_from_messages(raw_result)
                if price_data:
                    parsed_price  = price_data["price"]
                    parse_success = True
                    reliability   = "ok_strong"
                elif raw_text:
                    parsed_price  = _extract_structured_price(raw_text)
                    parse_success = parsed_price is not None
                    reliability   = "ok_strong" if parse_success else "unreliable"
                else:
                    parse_success = False
                    reliability   = "unreliable"

            # --- Auto-retry si el contenido es insuficiente ---
            if reliability in _RETRY_ON_RELIABILITY and strategy != "force_search":
                _emit_node_outcome(
                    rid, "web_scraping_node", "retry", phase="agent",
                    agent="web_scraping_agent", duration_ms=duration_ms,
                    reason=f"auto_retry:{reliability}",
                    scrape_reliability=reliability, strategy=strategy,
                    source_type=source_type, category=category, **tokens, **_node_meta(),
                )
                retry_summary, retry_words, retry_tokens, retry_quality = await _run_retry_agent(
                    agent, last_message, rid, get_llm_fn,
                )
                if retry_summary is not None:
                    summary           = retry_summary
                    words             = retry_words
                    summary_triggered = len(words) > 200
                    tokens            = retry_tokens
                    quality           = retry_quality

                strategy    = "force_search"
                reliability = _scrape_reliability(len(words))
                retry_done  = True
                duration_ms = int((time.time() - t0) * 1000)

            # --- Validación de output: si sigue sin contenido, fallar limpiamente ---
            if reliability == "unreliable":
                new_tracker, analytics = _update_scrape_tracker(
                    tracker, category, len(words), turn_count,
                    duration_ms=duration_ms, cost_usd=tokens.get("estimated_cost_usd"),
                    source_type=source_type, reliability_override=reliability,
                )
                _emit_node_outcome(
                    rid, "web_scraping_node", "low_confidence", phase="agent",
                    agent="web_scraping_agent", duration_ms=duration_ms,
                    scrape_reliability=reliability, strategy=strategy,
                    retry_done=retry_done, category=category,
                    source_type=source_type, price_extracted=parsed_price, parse_success=parse_success,
                    ml_recommended=ml_recommended, prediction_match=prediction_match,
                    ml_would_succeed=(False if prediction_match is True else None),
                    **tokens, **quality, **_node_meta(), **analytics,
                )
                return {
                    "messages": [AIMessage(content=(
                        "No pude obtener información confiable para esta consulta. "
                        "Intenta proporcionar una URL específica o reformular la pregunta."
                    ))],
                    "scrape_tracker": new_tracker,
                }

            cost_usd    = tokens.get("estimated_cost_usd")
            new_tracker, analytics = _update_scrape_tracker(
                tracker, category, len(words), turn_count,
                duration_ms=duration_ms, cost_usd=cost_usd,
                source_type=source_type, reliability_override=reliability,
            )
            new_score = _get_category_score(new_tracker, category, turn_count)

            if reliability not in ("ok_weak", "ok_strong"):
                followup = {"followup_likely": True}

            quality_target_val = analytics.get("quality_target", 0)
            ml_would_succeed = (
                bool(quality_target_val) if prediction_match is True else None
            )

            _emit_node_outcome(
                rid, "web_scraping_node", "success", phase="agent",
                agent="web_scraping_agent", duration_ms=duration_ms,
                summary_triggered=summary_triggered, raw_words=len(words),
                category=category, exploring=exploring, strategy=strategy, exp_rate=exp_rate,
                scrape_reliability=reliability, prior_reliability=prior_reliability,
                prior_score=prior_score, scrape_score=new_score,
                retry_done=retry_done,
                source_type=source_type, price_extracted=parsed_price, parse_success=parse_success,
                ml_recommended=ml_recommended, prediction_match=prediction_match,
                ml_would_succeed=ml_would_succeed,
                **tokens, **quality, **followup, **analytics, **meta,
            )
            return {
                "messages": [AIMessage(content=summary)],
                "scrape_tracker": new_tracker,
            }

        except Exception as e:
            _emit_node_outcome(
                rid, "web_scraping_node", "error", phase="agent",
                agent="web_scraping_agent",
                duration_ms=int((time.time() - t0) * 1000),
                reason=str(e),
                followup_likely=True,
                **_node_meta(),
            )
            raise

    return web_scraping_node


__all__ = ["make_web_scraping_node"]
