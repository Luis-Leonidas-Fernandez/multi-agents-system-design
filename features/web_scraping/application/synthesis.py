"""Síntesis LLM de resultados de búsqueda web."""
from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import HumanMessage


async def _synthesize_search_summary(
    raw_summary: str,
    query: str,
    get_llm_fn: Callable,
    sources: list[dict[str, str]],
    has_labeled_content: bool = False,
) -> str:
    from features.web_scraping.application import flow as _flow

    try:
        llm = get_llm_fn()
        sources_block = _flow._format_sources(sources)
        clean_lines = []
        for line in raw_summary.splitlines():
            stripped = line.strip()
            if not stripped or stripped in ("...", "[...]"):
                continue
            if stripped.startswith("#"):
                continue
            if any(word.count("-") >= 3 for word in stripped.split()):
                continue
            clean_lines.append(stripped)
        import datetime
        today_str = datetime.date.today().strftime("%d de %B de %Y")
        clean_content = "\n\n".join(clean_lines[:40])
        query_terms_for_dedup = _flow._extract_generic_query_terms(query)
        query_horizon_local = _flow.detect_recent_query_horizon(query) if _flow._is_recent_web_information_query(query) else None
        translation_prefix = ""
        if has_labeled_content:
            translation_prefix = (
                "⚠️ TRADUCCIÓN OBLIGATORIA: El contenido de abajo puede estar en otro idioma "
                "(italiano, japonés, etc.). DEBES traducir TODO a español rioplatense. "
                "NUNCA copies texto en otro idioma — siempre traducí.\n\n"
            )
        prompt = (
            f"{translation_prefix}"
            f"Fecha actual: {today_str}\n"
            f"Consulta del usuario: {query}\n\n"
            f"Información recopilada de la web:\n{clean_content}\n\n"
            "Sintetizá una respuesta clara respondiendo ÚNICAMENTE con lo que está en el texto de arriba. "
            "PROHIBIDO usar conocimiento propio o información que no esté en el texto provisto.\n\n"
            "Reglas de contenido:\n"
            "- IDIOMA: respondé siempre en el mismo idioma que la consulta del usuario\n"
            "- IGNORÁ completamente: pie de fotos, descripciones de imágenes, nombres de personas sin contexto noticioso, títulos de anime/manga, fragmentos sin información útil\n"
            "- PRIORIZÁ: artículos con hechos concretos, cifras, eventos, decisiones o noticias verificables\n"
            "- Si el contenido disponible no responde bien la consulta, indicalo brevemente\n\n"
            "Reglas de formato:\n"
            "- Cada punto DEBE comenzar con '•' seguido de un espacio\n"
            "- Cada artículo/fuente del texto = UN punto separado, pero TODOS deben responder al mismo tema solicitado.\n"
            "- NUNCA combines dos artículos en un solo punto. Cada punto viene de UNA sola fuente y no debe repetir la misma noticia.\n"
            "- Si un artículo tiene información irrelevante para la consulta (noticias de otro país, entretenimiento, deportes sin relación), omitilo.\n"
            "- OBLIGATORIO: dejá UNA línea en blanco entre cada punto\n"
            "- Cada punto tiene 2-3 oraciones con el hecho concreto, quiénes están involucrados y por qué importa\n"
            "- NO uses títulos ni headers (##, ###) dentro de la respuesta\n"
        )
        if has_labeled_content:
            prompt += (
                "- OBLIGATORIO: después del texto de cada punto, en una nueva línea escribí exactamente "
                "'Fuente: [titulo]' usando el título entre corchetes del texto de arriba\n"
                "- NO incluyas una sección Sources al final"
            )
        else:
            prompt += "- NO incluyas una sección Sources — se agrega automáticamente"
        if query_horizon_local == "week":
            cutoff = (datetime.date.today() - datetime.timedelta(days=30)).strftime('%d/%m/%Y')
            prompt += f"\n- Solo incluí eventos de los últimos 30 días (desde el {cutoff}). Descartá cualquier evento más antiguo aunque esté en el texto."
        elif query_horizon_local:
            prompt += "\n- Solo incluí eventos ocurridos en los últimos 30 días. Descartá cualquier evento más antiguo."
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        synthesized = getattr(response, "content", str(response)).strip()
        synthesized = __import__("re").split(r"\n\s*sources\s*:", synthesized, maxsplit=1, flags=__import__("re").IGNORECASE)[0].strip()
        synthesized = _flow._enforce_synthesis_format(synthesized)
        synthesized = _flow._dedup_synthesis_bullets(synthesized, query_terms_for_dedup)
        if not has_labeled_content and sources_block:
            synthesized = f"{synthesized}\n\n{sources_block}"
        return synthesized
    except Exception as _synth_exc:
        import logging
        logging.warning(f"_synthesize_search_summary FAILED: {type(_synth_exc).__name__}: {_synth_exc}")
        return _flow._enforce_synthesis_format(raw_summary)


__all__ = ["_synthesize_search_summary"]
