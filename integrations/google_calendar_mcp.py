"""
Google Calendar MCP server for Codex.

This server is intentionally dependency-light:
- no OpenAI SDK
- no FastMCP dependency
- only stdlib + requests

It speaks MCP over stdio using JSON-RPC messages, so Codex can discover and
call the tools from any project once the global Codex config points here.

Authentication model:
- OAuth 2.0 refresh token flow
- required env vars:
  - GOOGLE_OAUTH_CLIENT_ID
  - GOOGLE_OAUTH_CLIENT_SECRET
  - GOOGLE_OAUTH_REFRESH_TOKEN
- optional env vars:
  - GOOGLE_CALENDAR_ID (defaults to "primary")

You can store those in:
- ~/.codex/google-calendar.env for global use
- ./.env for repo-local testing
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests


GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3"
GOOGLE_CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar.events"
SUPPORTED_PROTOCOL_VERSIONS = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]


def _load_env_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file(Path.home() / ".codex" / "google-calendar.env")
_load_env_file(Path.cwd() / ".env")


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _calendar_id() -> str:
    return os.getenv("GOOGLE_CALENDAR_ID", "primary")


def _access_token() -> str:
    response = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "client_id": _required_env("GOOGLE_OAUTH_CLIENT_ID"),
            "client_secret": _required_env("GOOGLE_OAUTH_CLIENT_SECRET"),
            "refresh_token": _required_env("GOOGLE_OAUTH_REFRESH_TOKEN"),
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    response.raise_for_status()
    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("Google OAuth token refresh did not return an access_token.")
    return token


def _api_request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {_access_token()}",
        "Content-Type": "application/json",
    }
    response = requests.request(
        method=method,
        url=f"{GOOGLE_CALENDAR_API_BASE}{path}",
        headers=headers,
        params=params,
        json=json_body,
        timeout=30,
    )
    response.raise_for_status()
    if response.text.strip():
        return response.json()
    return {"ok": True}


def _default_timed_window() -> tuple[str, str]:
    start = datetime.now(timezone.utc).replace(microsecond=0)
    end = start + timedelta(hours=1)
    return start.isoformat(), end.isoformat()


def _next_day(date_value: str) -> str:
    parsed = datetime.fromisoformat(date_value)
    return (parsed + timedelta(days=1)).date().isoformat()


def _normalize_event_time(value: str) -> dict[str, Any]:
    if "T" not in value:
        return {"date": value}
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    return {"dateTime": normalized}


def _tool_text(payload: Any) -> dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            }
        ],
        "isError": False,
    }


def _tool_error(message: str) -> dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": message,
            }
        ],
        "isError": True,
    }


TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_events",
        "description": "List Google Calendar events within a time window.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "time_min": {"type": "string", "description": "RFC3339 time lower bound."},
                "time_max": {"type": "string", "description": "RFC3339 time upper bound."},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 250, "default": 10},
                "query": {"type": "string", "description": "Free text search query."},
                "calendar_id": {"type": "string", "description": "Calendar ID, defaults to primary."},
            },
        },
    },
    {
        "name": "create_event",
        "description": "Create a Google Calendar event.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "start": {"type": "string", "description": "RFC3339 or YYYY-MM-DD."},
                "end": {"type": "string", "description": "RFC3339 or YYYY-MM-DD."},
                "timezone_name": {"type": "string", "description": "IANA timezone, e.g. America/Argentina/Buenos_Aires."},
                "description": {"type": "string"},
                "location": {"type": "string"},
                "calendar_id": {"type": "string"},
            },
            "required": ["summary"],
        },
    },
    {
        "name": "update_event",
        "description": "Update an existing Google Calendar event.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "summary": {"type": "string"},
                "start": {"type": "string"},
                "end": {"type": "string"},
                "timezone_name": {"type": "string"},
                "description": {"type": "string"},
                "location": {"type": "string"},
                "calendar_id": {"type": "string"},
            },
            "required": ["event_id"],
        },
    },
    {
        "name": "delete_event",
        "description": "Delete a Google Calendar event.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "calendar_id": {"type": "string"},
            },
            "required": ["event_id"],
        },
    },
]


def list_events(args: dict[str, Any]) -> dict[str, Any]:
    time_min = args.get("time_min") or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    params: dict[str, Any] = {
        "timeMin": time_min,
        "maxResults": int(args.get("max_results", 10)),
        "singleEvents": "true",
        "orderBy": "startTime",
    }
    if args.get("time_max"):
        params["timeMax"] = args["time_max"]
    if args.get("query"):
        params["q"] = args["query"]
    return _api_request("GET", f"/calendars/{args.get('calendar_id') or _calendar_id()}/events", params=params)


def create_event(args: dict[str, Any]) -> dict[str, Any]:
    start = args.get("start")
    end = args.get("end")
    if start is None and end is None:
        start, end = _default_timed_window()
    elif start is not None and end is None:
        if "T" in start:
            parsed_start = start[:-1] + "+00:00" if start.endswith("Z") else start
            end = (datetime.fromisoformat(parsed_start) + timedelta(hours=1)).isoformat()
        else:
            end = _next_day(start)

    event: dict[str, Any] = {
        "summary": args["summary"],
        "start": _normalize_event_time(start),
        "end": _normalize_event_time(end),
    }
    if args.get("timezone_name"):
        event["start"]["timeZone"] = args["timezone_name"]
        event["end"]["timeZone"] = args["timezone_name"]
    if args.get("description"):
        event["description"] = args["description"]
    if args.get("location"):
        event["location"] = args["location"]
    return _api_request("POST", f"/calendars/{args.get('calendar_id') or _calendar_id()}/events", json_body=event)


def update_event(args: dict[str, Any]) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    if args.get("summary") is not None:
        patch["summary"] = args["summary"]
    if args.get("start") is not None:
        patch["start"] = _normalize_event_time(args["start"])
    if args.get("end") is not None:
        patch["end"] = _normalize_event_time(args["end"])
    if args.get("timezone_name") and "start" in patch:
        patch["start"]["timeZone"] = args["timezone_name"]
    if args.get("timezone_name") and "end" in patch:
        patch["end"]["timeZone"] = args["timezone_name"]
    if args.get("description") is not None:
        patch["description"] = args["description"]
    if args.get("location") is not None:
        patch["location"] = args["location"]
    return _api_request(
        "PATCH",
        f"/calendars/{args.get('calendar_id') or _calendar_id()}/events/{args['event_id']}",
        json_body=patch,
    )


def delete_event(args: dict[str, Any]) -> dict[str, Any]:
    return _api_request(
        "DELETE",
        f"/calendars/{args.get('calendar_id') or _calendar_id()}/events/{args['event_id']}",
    )


TOOL_HANDLERS = {
    "list_events": list_events,
    "create_event": create_event,
    "update_event": update_event,
    "delete_event": delete_event,
}


def _jsonrpc_error(req_id: Any, code: int, message: str, data: Any | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}
    if data is not None:
        payload["error"]["data"] = data
    return payload


def _initialize_result(requested_version: str) -> dict[str, Any]:
    protocol_version = requested_version if requested_version in SUPPORTED_PROTOCOL_VERSIONS else SUPPORTED_PROTOCOL_VERSIONS[0]
    return {
        "protocolVersion": protocol_version,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {
            "name": "google-calendar-mcp",
            "title": "Google Calendar MCP",
            "version": "1.0.0",
            "description": "Google Calendar integration for Codex via MCP.",
        },
    }


def _handle_request(message: dict[str, Any]) -> dict[str, Any] | None:
    if message.get("jsonrpc") != "2.0":
        return _jsonrpc_error(message.get("id"), -32600, "Invalid Request")

    method = message.get("method")
    req_id = message.get("id")
    params = message.get("params") or {}

    if method == "initialize":
        requested = params.get("protocolVersion", SUPPORTED_PROTOCOL_VERSIONS[0])
        return {"jsonrpc": "2.0", "id": req_id, "result": _initialize_result(requested)}

    if method == "notifications/initialized":
        return None

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            return _jsonrpc_error(req_id, -32602, f"Unknown tool: {name}")
        try:
            result = handler(arguments)
            return {"jsonrpc": "2.0", "id": req_id, "result": _tool_text(result)}
        except Exception as exc:  # noqa: BLE001 - tool errors must be returned, not raised
            return {"jsonrpc": "2.0", "id": req_id, "result": _tool_error(str(exc))}

    return _jsonrpc_error(req_id, -32601, f"Method not found: {method}")


def main() -> int:
    print("Google Calendar MCP server ready", file=sys.stderr)
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            response = _jsonrpc_error(None, -32700, "Parse error", {"details": str(exc)})
            print(json.dumps(response), flush=True)
            continue

        response = _handle_request(message)
        if response is not None and message.get("id") is not None:
            print(json.dumps(response), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
