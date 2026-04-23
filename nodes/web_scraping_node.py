"""Nodo fino del agente de web scraping.

Toda la coordinación vive en `application/use_cases/web_scraping_flow.py`.
Este módulo solo adapta dependencias concretas al caso de uso.
"""
import re
from typing import Callable, Awaitable, Any, cast

from langchain_core.messages import AIMessage
from application.policies.agentdog import evaluate_trajectory_safe, _should_evaluate_guard
from features.web_scraping.api import run_web_scraping_flow
from ports.confirmation_port import ConfirmationPort
from ports.llm_port import LLMFactory
import application.policies.hitl_flow as hitl_flow
from application.policies.scrape_tracker import get_runtime_policy
from domain.models import AgentState


def _extract_block_title(para: str) -> str:
    """Extrae un título corto del primer fragmento de un párrafo."""
    first_sent = re.split(r'[.!?,]', para, maxsplit=1)[0].strip()
    words = first_sent.split()
    return " ".join(words[:10]) + ("..." if len(words) > 10 else "")


def _format_news_blocks(content: str) -> str:
    """Reformatea el contenido final en bloques con título ## y fuente inline.

    Maneja dos formatos de entrada:
    1. Fuentes inline (nuevo): párrafo + \\nFuente: ... — agrega títulos ## por bloque.
       Usa re.split para tolerar tanto \\n como \\n\\n antes de la línea Fuente:.
    2. Sources: al final (legado): redistribuye 1:1 si counts coinciden, sino
       agrega títulos y deja Sources: al fondo.
    """
    # ── Formato nuevo: fuentes inline ────────────────────────────────────────
    if re.search(r'(?:^|\n)Fuente:\s*\S', content, re.MULTILINE):
        # Dividir por líneas "Fuente: ..." capturando el texto de la fuente.
        # re.split con grupo capturador alterna: [pre1, fuente1, pre2, fuente2, ...]
        split_parts = re.split(r'\n+(Fuente:\s*.+)', content)
        blocks: list[str] = []
        i = 0
        while i < len(split_parts):
            para_raw = split_parts[i].strip()
            fuente_line: str | None = None
            if i + 1 < len(split_parts) and split_parts[i + 1].startswith('Fuente:'):
                fuente_line = split_parts[i + 1].strip()
                i += 2
            else:
                i += 1

            if not para_raw:
                continue

            # Omitir título global en negrita (ej: **Seguridad en Argentina**)
            para_stripped = re.sub(r'^[•\-\*]\s+', '', para_raw)
            if re.match(r'^\*\*[^*\n]+\*\*$', para_stripped):
                if fuente_line:
                    blocks.append(f"{para_raw}\n\n{fuente_line}")
                else:
                    blocks.append(para_raw)
                continue

            title = _extract_block_title(para_stripped)
            if fuente_line:
                blocks.append(f"## {title}\n\n{para_stripped}\n\n{fuente_line}")
            else:
                blocks.append(f"## {title}\n\n{para_stripped}")

        return "\n\n".join(blocks) if blocks else content

    # ── Formato legado: Sources: al final ────────────────────────────────────
    parts = re.split(r'\n[ \t]*[Ss]ources[ \t]*:[ \t]*\n', content, maxsplit=1)
    if len(parts) != 2:
        return content

    body_text, sources_text = parts

    source_items: list[dict[str, str]] = []
    for line in sources_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^-\s*\[(.+?)\]\((.+?)\)', line)
        if m:
            source_items.append({"name": m.group(1), "url": m.group(2)})
        elif line.startswith('-'):
            source_items.append({"name": line.lstrip('- '), "url": ""})

    if len(source_items) < 2:
        return content

    paragraphs: list[str] = []
    for chunk in re.split(r'\n\n+', body_text.strip()):
        chunk = chunk.strip()
        if not chunk:
            continue
        if re.match(r'^\*\*[^*\n]+\*\*$', chunk):
            continue
        chunk = re.sub(r'^[•\-\*]\s+', '', chunk)
        if chunk:
            paragraphs.append(chunk)

    if len(paragraphs) < 2:
        return content

    if len(paragraphs) != len(source_items):
        # Counts distintos: agrega títulos pero deja Sources: al fondo (sin reasignar)
        titled: list[str] = []
        for para in paragraphs:
            title = _extract_block_title(para)
            titled.append(f"## {title}\n\n{para}")
        sources_block = "Sources:\n" + sources_text.strip()
        return "\n\n".join(titled) + "\n\n" + sources_block

    # Counts iguales: asignación 1:1 garantizada
    result_blocks: list[str] = []
    for para, src in zip(paragraphs, source_items):
        title = _extract_block_title(para)
        source_str = (
            f"Fuente: [{src['name']}]({src['url']})" if src['url']
            else f"Fuente: {src['name']}"
        )
        result_blocks.append(f"## {title}\n\n{para}\n\n{source_str}")

    return "\n\n".join(result_blocks)


# ==================== FACTORY ====================

def make_web_scraping_node(
    agent,
    get_llm_fn: LLMFactory,
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Retorna un adaptador fino que delega toda la lógica al caso de uso."""

    async def web_scraping_node(state: AgentState) -> dict[str, Any]:
        class _PatchedConfirmationAdapter:
            async def confirm(self, prompt: str) -> bool:
                return await hitl_flow.ask_confirmation(prompt)

        result = await run_web_scraping_flow(
            state,
            agent,
            get_llm_fn,
            hitl_enabled=hitl_flow.HITL_ENABLED,
            confirmation_handler=cast(ConfirmationPort, _PatchedConfirmationAdapter()) if hitl_flow.HITL_ENABLED else None,
            ask_confirmation_compat=hitl_flow.ask_confirmation if hitl_flow.HITL_ENABLED else None,
            get_runtime_policy=get_runtime_policy,
            evaluate_trajectory_safe_fn=evaluate_trajectory_safe,
            should_evaluate_guard_fn=_should_evaluate_guard,
        )

        msgs = result.get("messages")
        if msgs and hasattr(msgs[0], "content") and isinstance(msgs[0].content, str):
            formatted = _format_news_blocks(msgs[0].content)
            if formatted is not msgs[0].content:
                result["messages"] = [AIMessage(content=formatted)] + list(msgs[1:])

        return result

    return web_scraping_node


__all__ = ["make_web_scraping_node"]
