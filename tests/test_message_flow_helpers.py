"""Tests de helpers compartidos de mensajes."""
from langchain_core.messages import AIMessage, HumanMessage


def test_get_last_message_text_returns_last_text():
    from application.helpers.message_flow_helpers import get_last_message_text

    messages = [HumanMessage(content="hola"), AIMessage(content="ok")]
    assert get_last_message_text(messages) == "ok"


def test_is_btc_price_query_detects_query():
    from application.helpers.message_flow_helpers import is_btc_price_query

    assert is_btc_price_query("¿Cuál es el precio del bitcoin?") is True
    assert is_btc_price_query("Analiza este dataset") is False


def test_extract_final_ai_text_returns_last_ai_message():
    from application.helpers.message_flow_helpers import extract_final_ai_text

    messages = [
        HumanMessage(content="hola"),
        AIMessage(content="respuesta intermedia"),
        AIMessage(content="respuesta final"),
    ]
    assert extract_final_ai_text(messages) == "respuesta final"
