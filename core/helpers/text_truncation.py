"""Helpers compartidos de truncado de texto."""


def truncate_head_tail(text: str, *, max_chars: int, head_chars: int, tail_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[:head_chars]
    tail = text[-tail_chars:]
    return head + "\n...[truncated]...\n" + tail


def truncate_suffix(text: str, *, max_chars: int, suffix: str = "... [texto truncado]") -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + suffix


__all__ = ["truncate_head_tail", "truncate_suffix"]
