"""
persistence.py — Capa de persistencia de sesiones.

Backend principal: SQLite (sessions/sessions.db).
Backend fallback:  JSONL (sessions/{session_id}.jsonl) si USE_SQLITE=false.

Tabla messages:
    id         INTEGER PRIMARY KEY AUTOINCREMENT
    session_id TEXT NOT NULL
    role       TEXT NOT NULL   -- "human" | "ai"
    content    TEXT NOT NULL
    ts         INTEGER NOT NULL  -- unix epoch ms
    request_id TEXT             -- para JOIN futuro con audit log (aún sin usar)

Diseño:
  - Conexión singleton: CLI es single-threaded, no necesita pool.
  - Commit en cada write: safe frente a crashes.
  - Migración one-shot desde JSONL en el primer load (no borra el JSONL original).
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

# ==================== CONFIG ====================

# USE_SQLITE=false → fallback a JSONL (útil para debug o si SQLite no está disponible)
_USE_SQLITE: bool = os.environ.get("USE_SQLITE", "true").lower() != "false"

_SESSIONS_DIR = Path(__file__).parent / "sessions"
_DB_PATH      = _SESSIONS_DIR / "sessions.db"

# ==================== SCHEMA ====================

_DDL_TABLE = """
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL,
    role       TEXT    NOT NULL,
    content    TEXT    NOT NULL,
    ts         INTEGER NOT NULL,
    request_id TEXT
);
"""

# Índice para load_messages y futuros JOINs con audit log por session_id + orden temporal
_DDL_IDX = """
CREATE INDEX IF NOT EXISTS idx_session_ts ON messages (session_id, ts, id);
"""

# ==================== CONNECTION ====================

_conn: Optional[sqlite3.Connection] = None


def init_db() -> sqlite3.Connection:
    """Crea la DB, tabla e índice si no existen. Retorna conexión lista."""
    _SESSIONS_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")   # mejor concurrencia futura
    conn.execute(_DDL_TABLE)
    conn.execute(_DDL_IDX)
    conn.commit()
    return conn


def _get_conn() -> sqlite3.Connection:
    """Singleton de conexión. Solo inicializa al primer uso."""
    global _conn
    if _conn is None:
        _conn = init_db()
    return _conn

# ==================== SERIALIZACIÓN ====================

def _role_from_msg(msg: BaseMessage) -> str:
    return "human" if isinstance(msg, HumanMessage) else "ai"


def _row_to_msg(role: str, content: str) -> BaseMessage:
    return HumanMessage(content=content) if role == "human" else AIMessage(content=content)


def _msg_to_jsonl_dict(msg: BaseMessage) -> dict:
    """Formato legacy JSONL: {"type": "HumanMessage", "content": "..."}"""
    return {"type": msg.__class__.__name__, "content": msg.content}


def _jsonl_dict_to_msg(d: dict) -> BaseMessage:
    """Convierte dict JSONL (type/role) a mensaje LangChain."""
    role_or_type = d.get("type") or d.get("role", "")
    if role_or_type in ("HumanMessage", "human"):
        return HumanMessage(content=d["content"])
    return AIMessage(content=d["content"])

# ==================== API PÚBLICA ====================

def save_message(
    session_id: str,
    role: str,                        # "human" | "ai"
    content: str,
    request_id: Optional[str] = None,
) -> None:
    """Persiste un mensaje de forma incremental. Commit inmediato (CLI-safe).

    En modo JSONL esta función es no-op: save_session() maneja la escritura al final.
    """
    if not _USE_SQLITE:
        return
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO messages (session_id, role, content, ts, request_id) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, int(time.time() * 1000), request_id),
        )
        conn.commit()
    except Exception as exc:
        # No romper el flujo si la DB falla
        print(f"  [persistence] warning: no se pudo guardar mensaje: {exc}")


def load_messages(session_id: str) -> list[BaseMessage]:
    """Carga todos los mensajes de la sesión en orden cronológico.

    Si USE_SQLITE=true:
      - Migra automáticamente desde JSONL si existe (one-shot, sin borrar el JSONL).
      - Lee desde SQLite.
    Si USE_SQLITE=false:
      - Lee directamente desde JSONL.
    """
    if not _USE_SQLITE:
        return _load_jsonl(session_id)

    _maybe_migrate_jsonl(session_id)

    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY ts, id",
            (session_id,),
        ).fetchall()
        return [_row_to_msg(role, content) for role, content in rows]
    except Exception as exc:
        print(f"  [persistence] warning: error leyendo SQLite, usando JSONL: {exc}")
        return _load_jsonl(session_id)


def save_session(session_id: str, messages: list[BaseMessage]) -> None:
    """Compatibilidad con la interfaz anterior de main.py.

    SQLite: no-op (los mensajes ya se guardaron turno a turno con save_message).
    JSONL:  escribe el historial completo (comportamiento original).
    """
    if not _USE_SQLITE:
        _save_jsonl(session_id, messages)


def list_sessions() -> list[str]:
    """Lista IDs de sesiones únicas (SQLite + JSONL deduplicado, orden alfabético)."""
    ids: set[str] = set()

    if _USE_SQLITE:
        try:
            rows = _get_conn().execute(
                "SELECT DISTINCT session_id FROM messages"
            ).fetchall()
            ids.update(r[0] for r in rows)
        except Exception:
            pass

    # Incluir JSONL existentes para no perder sesiones previas a la migración
    if _SESSIONS_DIR.exists():
        ids.update(p.stem for p in _SESSIONS_DIR.glob("*.jsonl"))

    return sorted(ids)

# ==================== BACKEND JSONL (FALLBACK) ====================

def _load_jsonl(session_id: str) -> list[BaseMessage]:
    path = _SESSIONS_DIR / f"{session_id}.jsonl"
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
            pass   # línea corrupta → ignorar sin romper el resto
    return messages


def _save_jsonl(session_id: str, messages: list[BaseMessage]) -> None:
    _SESSIONS_DIR.mkdir(exist_ok=True)
    path = _SESSIONS_DIR / f"{session_id}.jsonl"
    lines = [json.dumps(_msg_to_jsonl_dict(m), ensure_ascii=False) for m in messages]
    path.write_text("\n".join(lines), encoding="utf-8")

# ==================== MIGRACIÓN JSONL → SQLITE ====================

_migrated: set[str] = set()   # evita re-chequear en el mismo proceso


def _maybe_migrate_jsonl(session_id: str) -> None:
    """Migra los mensajes del JSONL a SQLite exactamente una vez por sesión y proceso.

    Reglas:
      - Solo migra si el JSONL existe.
      - Si ya hay filas en SQLite para esa sesión, asume que ya fue migrado y no duplica.
      - No borra el JSONL original (sirve como backup y debug).
    """
    if session_id in _migrated:
        return

    path = _SESSIONS_DIR / f"{session_id}.jsonl"
    if not path.exists():
        _migrated.add(session_id)
        return

    conn = _get_conn()
    count = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
    ).fetchone()[0]

    if count > 0:
        _migrated.add(session_id)
        return   # Ya tiene datos → no duplicar

    msgs = _load_jsonl(session_id)
    if not msgs:
        _migrated.add(session_id)
        return

    # JSONL no tiene timestamps reales → asignar ts sintéticos secuenciales
    # (1 segundo entre mensajes, terminando 1 segundo antes del "ahora")
    base_ts = int(time.time() * 1000) - len(msgs) * 1000
    for i, msg in enumerate(msgs):
        conn.execute(
            "INSERT INTO messages (session_id, role, content, ts, request_id) VALUES (?, ?, ?, ?, ?)",
            (session_id, _role_from_msg(msg), msg.content, base_ts + i * 1000, None),
        )
    conn.commit()
    _migrated.add(session_id)
    print(f"  [persistence] migrado {len(msgs)} mensajes JSONL → SQLite (sesión {session_id})")
