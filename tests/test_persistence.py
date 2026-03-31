"""
Tests unitarios para persistence.py.

Usa tmp_path (SQLite en archivo temporal) para aislar cada test.
No testea la migración JSONL — TODO: test de migración requiere setup de filesystem complejo.
"""
import os
import sqlite3
import pytest

from langchain_core.messages import HumanMessage, AIMessage


# ==================== FIXTURES ====================

@pytest.fixture(autouse=True)
def _isolate_persistence(tmp_path, monkeypatch):
    """
    Aisla el módulo persistence para cada test:
    - Apunta _SESSIONS_DIR y _DB_PATH al tmp_path único del test
    - Resetea el singleton _conn y el set _migrated
    - Fuerza USE_SQLITE=true
    """
    import persistence

    sessions_dir = tmp_path / "sessions"
    db_path = sessions_dir / "sessions.db"

    monkeypatch.setattr(persistence, "_SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(persistence, "_DB_PATH", db_path)
    monkeypatch.setattr(persistence, "_conn", None)
    monkeypatch.setattr(persistence, "_migrated", set())
    monkeypatch.setattr(persistence, "_USE_SQLITE", True)

    yield

    # Cerrar la conexión al salir para que tmp_path pueda limpiarse
    if persistence._conn is not None:
        try:
            persistence._conn.close()
        except Exception:
            pass
        persistence._conn = None


# ==================== ROUND-TRIP save/load ====================

def test_save_and_load_messages_round_trip_human():
    import persistence
    persistence.save_message("sess-1", "human", "Hola mundo")
    msgs = persistence.load_messages("sess-1")
    assert len(msgs) == 1
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "Hola mundo"


def test_save_and_load_messages_round_trip_ai():
    import persistence
    persistence.save_message("sess-1", "ai", "Respuesta del agente")
    msgs = persistence.load_messages("sess-1")
    assert len(msgs) == 1
    assert isinstance(msgs[0], AIMessage)
    assert msgs[0].content == "Respuesta del agente"


def test_save_and_load_multiple_roles():
    import persistence
    persistence.save_message("sess-2", "human", "¿Qué es el BTC?")
    persistence.save_message("sess-2", "ai",    "Bitcoin es...")
    persistence.save_message("sess-2", "human", "¿Y el ETH?")
    msgs = persistence.load_messages("sess-2")
    assert len(msgs) == 3
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage)
    assert isinstance(msgs[2], HumanMessage)
    assert msgs[2].content == "¿Y el ETH?"


# ==================== Orden cronológico ====================

def test_multiple_messages_cargados_en_orden():
    """Los mensajes deben cargarse en orden de inserción (ts, id ASC)."""
    import persistence
    contents = [f"mensaje {i}" for i in range(5)]
    for i, c in enumerate(contents):
        role = "human" if i % 2 == 0 else "ai"
        persistence.save_message("sess-orden", role, c)
    msgs = persistence.load_messages("sess-orden")
    assert [m.content for m in msgs] == contents


# ==================== Sesión inexistente ====================

def test_load_messages_sesion_inexistente_retorna_lista_vacia():
    import persistence
    msgs = persistence.load_messages("sesion-que-no-existe-abc123")
    assert msgs == []


# ==================== list_sessions ====================

def test_list_sessions_retorna_sesiones_guardadas():
    import persistence
    persistence.save_message("alfa", "human", "msg1")
    persistence.save_message("beta", "ai",    "msg2")
    sessions = persistence.list_sessions()
    assert "alfa" in sessions
    assert "beta" in sessions


def test_list_sessions_orden_alfabetico():
    import persistence
    persistence.save_message("zeta",   "human", "z")
    persistence.save_message("alfa",   "human", "a")
    persistence.save_message("medios", "human", "m")
    sessions = persistence.list_sessions()
    # Solo evaluar el subconjunto de las sesiones creadas en este test
    created = [s for s in sessions if s in {"zeta", "alfa", "medios"}]
    assert created == sorted(created)


def test_list_sessions_sin_duplicados():
    import persistence
    for _ in range(3):
        persistence.save_message("unica", "human", "msg")
    sessions = persistence.list_sessions()
    assert sessions.count("unica") == 1


def test_list_sessions_vacio_cuando_no_hay_sesiones():
    import persistence
    sessions = persistence.list_sessions()
    # Con el directorio vacío (tmp_path nuevo), no debe haber sesiones
    assert isinstance(sessions, list)
    assert len(sessions) == 0


# ==================== Aislamiento entre sesiones ====================

def test_sesiones_aisladas():
    """Mensajes de una sesión no aparecen en otra."""
    import persistence
    persistence.save_message("sess-A", "human", "solo en A")
    persistence.save_message("sess-B", "human", "solo en B")

    msgs_a = persistence.load_messages("sess-A")
    msgs_b = persistence.load_messages("sess-B")

    assert len(msgs_a) == 1
    assert msgs_a[0].content == "solo en A"
    assert len(msgs_b) == 1
    assert msgs_b[0].content == "solo en B"


# ==================== save_message con request_id ====================

def test_save_message_con_request_id_no_falla():
    import persistence
    persistence.save_message("sess-rid", "human", "msg", request_id="req-uuid-123")
    msgs = persistence.load_messages("sess-rid")
    assert len(msgs) == 1
    assert msgs[0].content == "msg"


# ==================== save_session (compatibilidad) ====================

def test_save_session_en_modo_sqlite_es_noop():
    """save_session en modo SQLite no debe escribir nada ni fallar."""
    import persistence
    msgs = [HumanMessage(content="test")]
    # No debe lanzar excepción
    persistence.save_session("sess-noop", msgs)
    # El único mensaje guardado vía save_message es el que guardamos explícitamente
    persistence.save_message("sess-noop", "human", "test")
    loaded = persistence.load_messages("sess-noop")
    assert len(loaded) == 1


# ==================== _row_to_msg ====================

def test_row_to_msg_human():
    from persistence import _row_to_msg
    msg = _row_to_msg("human", "contenido")
    assert isinstance(msg, HumanMessage)
    assert msg.content == "contenido"


def test_row_to_msg_ai():
    from persistence import _row_to_msg
    msg = _row_to_msg("ai", "contenido ai")
    assert isinstance(msg, AIMessage)
    assert msg.content == "contenido ai"


def test_row_to_msg_unknown_role_retorna_ai():
    """Roles desconocidos caen a AIMessage."""
    from persistence import _row_to_msg
    msg = _row_to_msg("tool", "msg herramienta")
    assert isinstance(msg, AIMessage)


# ==================== TODO ====================
# test_maybe_migrate_jsonl: requiere setup de filesystem JSONL complejo.
# Dejar para un fixture de integración separado.
