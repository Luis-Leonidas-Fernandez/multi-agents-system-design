"""Contracto JSON para la web frontend en tiempo real."""
from __future__ import annotations

import re
from urllib.parse import urlparse
from dataclasses import asdict, dataclass
from typing import Any, Literal

PROTOCOL_VERSION = "1.0"


@dataclass(frozen=True)
class DashboardAgent:
    id: str
    name: str
    status: Literal["idle", "running", "success", "error"]


@dataclass(frozen=True)
class DashboardEvent:
    id: str
    kind: Literal["success", "error", "info", "warning", "token", "trace", "action"]
    title: str
    detail: str
    at: str
    agentId: str


@dataclass(frozen=True)
class DashboardLog:
    id: str
    level: Literal["debug", "info", "warn", "error"]
    message: str
    at: str


@dataclass(frozen=True)
class DashboardTokens:
    prompt: int
    completion: int
    total: int


@dataclass(frozen=True)
class DashboardSnapshot:
    activeAgent: DashboardAgent
    reasoning: str
    conclusion: str
    finalResponse: str
    turnId: str
    turnLatencyMs: int
    messageCount: int
    lastUserMessage: str
    lastAssistantResponse: str
    events: list[DashboardEvent]
    logs: list[DashboardLog]
    tokens: DashboardTokens
    sessionId: str


@dataclass(frozen=True)
class DashboardStatus:
    connected: bool
    mode: Literal["mock", "websocket"]


@dataclass(frozen=True)
class DashboardAction:
    agentId: str
    message: str


@dataclass(frozen=True)
class DashboardActionMessage:
    type: Literal["action"] = "action"
    payload: DashboardAction | None = None


def to_jsonable(value: Any) -> Any:
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(val) for key, val in asdict(value).items()}
    return value


def parse_response_sections(text: str) -> tuple[str, str, str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return ("", "", "")
    lower = cleaned.lower()
    markers = ["reasoning", "conclusion", "final response"]
    if all(marker in lower for marker in markers):
        # naive, but enough for dashboards until the backend emits structured payloads
        parts = {"reasoning": "", "conclusion": "", "final response": ""}
        current = None
        for raw_line in cleaned.splitlines():
            line = raw_line.strip()
            key = line.lower().rstrip(":")
            if key in parts:
                current = key
                continue
            if current:
                parts[current] = (parts[current] + "\n" + raw_line).strip()
        return parts["reasoning"], parts["conclusion"], parts["final response"]

    source_titles: list[str] = []
    seen: set[str] = set()
    for title, url in re.findall(r"\[([^\]]+)\]\((https?://[^)]+)\)", cleaned):
        label = f"{title.strip()} ({urlparse(url).hostname or url})".strip()
        if label and label not in seen:
            seen.add(label)
            source_titles.append(label)

    body = re.split(r"\n\s*Sources\s*:\s*\n", cleaned, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    if source_titles:
        reasoning = "Leyendo estas fuentes y extrayendo lo relevante:\n" + "\n".join(f"- {title}" for title in source_titles[:8])
        return reasoning, "", body or cleaned

    return ("", "", cleaned)
