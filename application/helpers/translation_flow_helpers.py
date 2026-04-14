"""
Traducción de términos de búsqueda para prensa local en idiomas no latinos.

Usa el mismo LLM del sistema con temperatura configurable via TRANSLATION_TEMPERATURE (default 0) para máxima consistencia.
El resultado se cachea en memoria por sesión — la traducción solo se hace
una vez por combinación (geography, topic, language).
"""
from __future__ import annotations

import json
from typing import Optional

# ── Cache de sesión ───────────────────────────────────────────────────────────
# Clave: (geography_en, topic, language_code) → list[list[str]]
_NATIVE_VARIANTS_CACHE: dict[tuple[str, str, str], list[list[str]]] = {}

# Idiomas ya cubiertos por las variantes hardcodeadas (español + inglés).
# Para estos no se hace traducción.
_SKIP_LANGUAGES = {"es", "en"}

# Descripción en inglés de cada topic para el prompt de traducción.
_TOPIC_DESCRIPTIONS: dict[str, str] = {
    "security": "security, crime, public safety, police",
    "economy":  "economy, market, finance, business",
    "politics": "politics, government, policy",
    "default":  "general news, latest news",
}


async def get_native_search_variants(
    geography_en: str,
    topic: str,
    target_language: str,
) -> list[list[str]]:
    """Devuelve variantes de búsqueda en el idioma nativo del país.

    Ejemplo::

        await get_native_search_variants("South Korea", "security", "ko")
        # → [["한국", "보안"], ["한국", "치안"]]

    - Retorna ``[]`` si el idioma es "es" o "en" (ya cubiertos).
    - Retorna ``[]`` si faltan argumentos.
    - Retorna ``[]`` ante cualquier error del LLM (nunca rompe el flujo).
    - Los resultados se cachean por sesión.
    """
    if not geography_en or not target_language:
        return []
    lang = target_language.strip().lower()
    if lang in _SKIP_LANGUAGES:
        return []

    cache_key = (geography_en, topic or "default", lang)
    if cache_key in _NATIVE_VARIANTS_CACHE:
        return _NATIVE_VARIANTS_CACHE[cache_key]

    result = await _fetch_native_variants(geography_en, topic or "default", lang)
    _NATIVE_VARIANTS_CACHE[cache_key] = result
    return result


async def _fetch_native_variants(
    geography_en: str,
    topic: str,
    target_language: str,
) -> list[list[str]]:
    """Llama al LLM para obtener las variantes nativas.

    Retorna ``[]`` ante cualquier fallo — la función nunca propaga excepciones.
    """
    try:
        from application.helpers.config_flow_helpers import get_llm
        from langchain_core.messages import HumanMessage

        topic_desc = _TOPIC_DESCRIPTIONS.get(topic, _TOPIC_DESCRIPTIONS["default"])

        prompt = (
            f"You are a multilingual news search assistant.\n"
            f"Generate exactly 2 short search term pairs in language '{target_language}' (ISO 639-1) "
            f"for finding news articles about:\n"
            f"  Country: {geography_en}\n"
            f"  Topic: {topic_desc}\n\n"
            f"Rules:\n"
            f"- Each pair = [country_name_in_native_script, topic_keyword_in_native_script]\n"
            f"- Use native script (Hangul for Korean, Kanji/Kana for Japanese, etc.)\n"
            f"- Keep each term 1–3 words max, suitable for a news site search query\n"
            f"- Return ONLY a JSON array of 2 arrays, no explanation, no markdown\n\n"
            f"Example for South Korea + security:\n"
            f'[["\ud55c\uad6d", "\ubcf4\uc548"], ["\ud55c\uad6d", "\uce58\uc548"]]\n'
        )

        import os
        translation_temp = float(os.getenv("TRANSLATION_TEMPERATURE", "0"))
        llm = get_llm(temperature=translation_temp)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = str(getattr(response, "content", response)).strip()

        # Extraer el primer array JSON válido de la respuesta
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return []

        parsed = json.loads(raw[start:end])
        if not isinstance(parsed, list):
            return []

        # Validar estructura: lista de listas de strings no vacíos
        result: list[list[str]] = []
        for item in parsed:
            if (
                isinstance(item, list)
                and len(item) >= 2
                and all(isinstance(s, str) and s.strip() for s in item)
            ):
                result.append([s.strip() for s in item])

        return result

    except Exception:
        return []
