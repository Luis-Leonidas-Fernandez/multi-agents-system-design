"""
AgentGateway + LaneQueue — Session Lane inspirado en OpenClaw.

LaneQueue:
  - concurrency=1 por session_key
  - Mensajes concurrentes → modo collect (se acumulan y procesan juntos)
  - Generation tracking para evitar race conditions

AgentGateway:
  - Orquesta estado por sesión, persistence, LangGraph y LaneQueue
  - API: send() / get_state() / close_session() / shutdown()
  - Base para integraciones futuras (Telegram, etc.)

Referencia: OpenClaw src/process/command-queue.ts
"""
import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from graph import create_supervisor_graph
from memory import distill_memory, load_memory_context
import persistence


# ==================== DEBUG LOG ====================

def _log(session_key: str, msg: str) -> None:
    if os.getenv("DEBUG_GATEWAY", "false").lower() == "true":
        print(f"[gateway] session={session_key} {msg}")


# ==================== LANE QUEUE ====================

@dataclass
class _LaneState:
    generation: int = 0
    running: bool = False
    collecting: list = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class LaneQueue:
    """
    Cola FIFO por session_key con concurrency=1.

    Implementa el Session Lane de OpenClaw:
    - Un run activo por session_key
    - Mensajes concurrentes → modo collect
    - Procesamiento batch en followup turn
    - Generation tracking para evitar race conditions
    """

    def __init__(self) -> None:
        self._lanes: dict[str, _LaneState] = {}

    def _get_lane(self, session_key: str) -> _LaneState:
        if session_key not in self._lanes:
            self._lanes[session_key] = _LaneState()
        return self._lanes[session_key]

    async def send(self, session_key: str, message: str, handler) -> str:
        """
        Envía un mensaje al handler del session_key.

        Si NO hay run activo: ejecuta handler y retorna respuesta real.
        Si HAY run activo: aplica modo collect y retorna "" (ack silencioso).
        """
        lane = self._get_lane(session_key)

        async with lane.lock:
            if lane.running:
                lane.collecting.append(message)
                _log(session_key, f"collect msg_added queue={len(lane.collecting)}")
                return ""
            lane.running = True
            my_gen = lane.generation

        response = await self._execute(session_key, message, handler, my_gen)
        return response

    async def _execute(self, session_key: str, message: str, handler, my_gen: int) -> str:
        """Ejecuta el handler y procesa mensajes acumulados en modo collect."""
        lane = self._get_lane(session_key)
        t0 = time.monotonic()
        _log(session_key, "run_start")

        try:
            response = await handler(message)
        except asyncio.CancelledError:
            async with lane.lock:
                lane.running = False
            return ""
        except Exception as e:
            response = f"Error: {e}"

        elapsed = time.monotonic() - t0

        # Generation check — run cancelado no actualiza estado
        async with lane.lock:
            if lane.generation != my_gen:
                lane.running = False
                _log(session_key, f"run_invalid gen_mismatch expected={my_gen} current={lane.generation}")
                return ""

        _log(session_key, f"run_end {elapsed:.2f}s collect_pending={len(lane.collecting)}")

        # Procesar mensajes acumulados como followup turns
        while True:
            async with lane.lock:
                if not lane.collecting:
                    lane.running = False
                    break
                combined = "\n".join(lane.collecting)
                n_msgs = len(lane.collecting)
                lane.collecting.clear()
                my_gen = lane.generation

            _log(session_key, f"followup_turn msgs={n_msgs} combined_chars={len(combined)}")

            try:
                await handler(combined)
            except Exception as e:
                _log(session_key, f"followup_turn_error msgs={n_msgs} error={e}")

            async with lane.lock:
                if lane.generation != my_gen:
                    lane.running = False
                    break

        return response

    async def close(self, session_key: str) -> None:
        """Cierra la sesión: incrementa generation e invalida runs activos."""
        lane = self._get_lane(session_key)
        async with lane.lock:
            lane.generation += 1
            lane.collecting.clear()
            lane.running = False
        _log(session_key, f"closed gen={lane.generation}")

    async def shutdown(self) -> None:
        """Cierra todas las sesiones activas."""
        for session_key in list(self._lanes):
            await self.close(session_key)
        self._lanes.clear()


# ==================== AGENT GATEWAY ====================

class AgentGateway:
    """
    Orquesta la interacción entre clientes y LangGraph.

    Responsabilidades:
    - Estado por sesión (AgentState)
    - Integración con persistence (load/save mensajes)
    - Carga de memoria destilada (MEMORY.md)
    - Ejecución del grafo vía LaneQueue (concurrency=1 por sesión)
    """

    def __init__(self) -> None:
        self._graph = create_supervisor_graph()
        self._states: dict[str, dict] = {}
        self._queue = LaneQueue()

    def _get_state(self, session_key: str) -> dict:
        """Carga o retorna el estado existente de la sesión."""
        if session_key not in self._states:
            prior_messages = persistence.load_messages(session_key)
            memory = load_memory_context(session_key)
            if memory:
                prior_messages = [
                    SystemMessage(content=f"[Memoria de sesiones previas]\n{memory}")
                ] + prior_messages
            self._states[session_key] = {
                "messages":       prior_messages,
                "next_agent":     "",
                "risk_flag":      False,
                "blocked":        False,
                "request_id":     "",
                "scrape_tracker": {},
            }
        return self._states[session_key]

    def load_session_info(self, session_key: str) -> tuple[int, bool]:
        """
        Pre-carga la sesión y retorna (n_mensajes_humanos_ai, tiene_memoria).
        Útil para mostrar info al usuario antes del primer send().
        """
        state = self._get_state(session_key)
        msgs = state["messages"]
        has_memory = any(
            getattr(m, "type", "") == "system" and "[Memoria" in getattr(m, "content", "")
            for m in msgs
        )
        n = sum(1 for m in msgs if getattr(m, "type", "") in ("human", "ai"))
        return n, has_memory

    async def _handle(self, session_key: str, message: str) -> str:
        """Handler interno: agrega mensaje, invoca grafo, persiste resultado."""
        state = self._get_state(session_key)
        request_id = str(uuid.uuid4())
        state["request_id"] = request_id

        persistence.save_message(session_key, "human", message, request_id=request_id)
        state["messages"].append(HumanMessage(content=message))

        try:
            result = await self._graph.ainvoke(state)
            self._states[session_key] = result

            if result["messages"]:
                last = result["messages"][-1]
                persistence.save_message(session_key, "ai", last.content, request_id=request_id)
                return last.content
            return ""
        except Exception as e:
            return f"Error: {e}"

    async def send(self, session_key: str, message: str) -> str:
        """
        Punto de entrada principal.

        Retorna la respuesta del agente, o "" si el mensaje fue acumulado
        en modo collect (run activo para esta sesión).
        """
        return await self._queue.send(
            session_key,
            message,
            lambda msg: self._handle(session_key, msg),
        )

    async def get_state(self, session_key: str) -> dict | None:
        """Retorna el estado actual de la sesión, o None si no existe."""
        return self._states.get(session_key)

    async def close_session(self, session_key: str) -> None:
        """
        Cierra la sesión: invalida runs activos, persiste y destila memoria.
        """
        await self._queue.close(session_key)

        state = self._states.get(session_key)
        if state:
            persistence.save_session(session_key, state["messages"])
            try:
                await asyncio.wait_for(distill_memory(state, session_key), timeout=10)
            except asyncio.TimeoutError:
                pass

    async def shutdown(self) -> None:
        """Cierra todas las sesiones y el queue."""
        await self._queue.shutdown()
