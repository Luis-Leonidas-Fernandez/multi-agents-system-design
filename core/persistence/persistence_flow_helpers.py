"""Helpers de serialización y JSONL para persistencia de sesiones."""
import json
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def _role_from_msg(msg: BaseMessage) -> str:
    return "human" if isinstance(msg, HumanMessage) else "ai"


def _row_to_msg(role: str, content: str) -> BaseMessage:
    return HumanMessage(content=content) if role == "human" else AIMessage(content=content)


def _msg_to_jsonl_dict(msg: BaseMessage) -> dict:
    return {"type": msg.__class__.__name__, "content": msg.content}


def _jsonl_dict_to_msg(d: dict) -> BaseMessage:
    role_or_type = d.get("type") or d.get("role", "")
    if role_or_type in ("HumanMessage", "human"):
        return HumanMessage(content=d["content"])
    return AIMessage(content=d["content"])


def _load_jsonl(session_id: str, sessions_dir: Path) -> list[BaseMessage]:
    path = sessions_dir / f"{session_id}.jsonl"
    if not path.exists():
        return []
    messages: list[BaseMessage] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            messages.append(_jsonl_dict_to_msg(json.loads(line)))
        except (json.JSONDecodeError, KeyError):
            pass
    return messages


def _save_jsonl(session_id: str, messages: list[BaseMessage], sessions_dir: Path) -> None:
    sessions_dir.mkdir(exist_ok=True)
    path = sessions_dir / f"{session_id}.jsonl"
    lines = [json.dumps(_msg_to_jsonl_dict(m), ensure_ascii=False) for m in messages]
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "_role_from_msg",
    "_row_to_msg",
    "_msg_to_jsonl_dict",
    "_jsonl_dict_to_msg",
    "_load_jsonl",
    "_save_jsonl",
]
