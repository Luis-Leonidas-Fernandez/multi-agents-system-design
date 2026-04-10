"""Helpers compartidos para casos de uso basados en mensajes."""

from langchain_core.messages import AIMessage


def get_last_message_text(messages) -> str:
    if not messages:
        return ""
    last = messages[-1]
    content = getattr(last, "content", "")
    return content if isinstance(content, str) else ""


def is_btc_price_query(text: str) -> bool:
    lm = text.lower()
    return ("bitcoin" in lm or "btc" in lm) and any(
        k in lm for k in ["precio", "price", "cotiza", "cotización", "cotizacion"]
    )


def is_web_information_query(text: str) -> bool:
    lm = text.lower()
    return any(
        k in lm
        for k in [
            "internet",
            "web",
            "online",
            "buscar",
            "busca",
            "buscá",
            "lookup",
            "search",
            "noticias",
            "noticia",
            "news",
        ]
    )


def extract_final_ai_text(messages) -> str:
    for msg in reversed(messages or []):
        content = getattr(msg, "content", None)
        if isinstance(msg, AIMessage) and isinstance(content, str) and content and not getattr(msg, "tool_calls", None):
            return content
    return ""


__all__ = ["get_last_message_text", "is_btc_price_query", "is_web_information_query", "extract_final_ai_text"]
