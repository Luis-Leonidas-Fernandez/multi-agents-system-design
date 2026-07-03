"""Construcción localizada de términos y queries para noticias por país."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


DebugHook = Optional[Callable[[str], None]]


@dataclass(frozen=True)
class QueryLocalizationContext:
    geography: str
    geo_en: str
    topic: str
    horizon: Optional[str]
    query_source_group: Optional[str]
    public_safety_query: bool = False


@dataclass(frozen=True)
class QuerySpec:
    language: str
    query: str


class GeoLanguageResolver:
    """Resuelve prioridad idiomática por país y por dominio."""

    _LANGUAGE_PRIORITY: dict[str, list[str]] = {
        "south_korea": ["ko", "en", "es"],
        "japan": ["ja", "en", "es"],
        "china": ["zh", "en", "es"],
        "united_states": ["en", "es"],
        "argentina": ["es", "en"],
        "germany": ["de", "en", "es"],
        "default": ["en", "es"],
    }

    _DOMAIN_LANGUAGE_PROFILE: dict[str, list[str]] = {
        "arirang.com": ["en", "ko"],
        "chosun.com": ["ko", "en"],
        "donga.com": ["ko", "en"],
        "edaily.co.kr": ["ko", "en"],
        "hani.co.kr": ["ko", "en"],
        "english.hani.co.kr": ["en", "ko"],
        "news.hankooki.com": ["ko", "en"],
        "hankyung.com": ["ko", "en"],
        "biz.heraldcorp.com": ["en", "ko"],
        "joongang.co.kr": ["ko", "en"],
        "news.kbs.co.kr": ["ko", "en"],
    }

    _TLD_LANGUAGE_PRIORITY: dict[str, list[str]] = {
        ".kr": ["ko", "en"],
        ".jp": ["ja", "en"],
        ".cn": ["zh", "en"],
        ".tw": ["zh", "en"],
        ".fr": ["fr", "en"],
        ".de": ["de", "en"],
        ".it": ["it", "en"],
        ".pt": ["pt", "en"],
        ".br": ["pt", "en"],
        ".es": ["es", "en"],
        ".mx": ["es", "en"],
        ".ar": ["es", "en"],
        ".cl": ["es", "en"],
        ".co": ["es", "en"],
        ".pe": ["es", "en"],
        ".uy": ["es", "en"],
    }

    def resolve(self, *, country_group: Optional[str], domain: str) -> list[str]:
        normalized_domain = (domain or "").strip().lower()
        domain_priority = list(self._DOMAIN_LANGUAGE_PROFILE.get(normalized_domain, []))
        tld_priority: list[str] = []
        for tld, languages in self._TLD_LANGUAGE_PRIORITY.items():
            if normalized_domain.endswith(tld):
                tld_priority = list(languages)
                break
        country_priority = list(self._LANGUAGE_PRIORITY.get(country_group or "", self._LANGUAGE_PRIORITY["default"]))
        ordered: list[str] = []
        seen: set[str] = set()
        for language in [*domain_priority, *tld_priority, *country_priority]:
            if language and language not in seen:
                seen.add(language)
                ordered.append(language)
        return ordered


class LocalizedNewsQueryBuilder:
    """Expande tópicos de noticias a queries monolingües por país/idioma."""

    _MAX_LANGUAGES_PER_DOMAIN = 2
    _MAX_TERMS_PER_LANGUAGE = 3
    _MAX_QUERY_SPECS_PER_DOMAIN = 6

    _TOPIC_TERMS: dict[str, dict[str, dict[str, list[str]]]] = {
        "security": {
            "public_safety": {
                "es": ["seguridad", "inseguridad", "policiales", "sucesos", "policia"],
                "en": ["security", "crime", "police", "public safety", "security incidents"],
                "fr": ["sécurité", "criminalité", "police"],
                "de": ["sicherheit", "kriminalität", "polizei"],
                "it": ["sicurezza", "cronaca", "polizia"],
                "pt": ["segurança", "crime", "polícia"],
            },
            "default": {
                "es": ["seguridad"],
                "en": ["security"],
                "fr": ["sécurité"],
                "de": ["sicherheit"],
                "it": ["sicurezza"],
                "pt": ["segurança"],
            },
        },
        "economy": {
            "default": {
                "es": ["economia", "mercado", "finanzas"],
                "en": ["economy", "market", "finance"],
                "fr": ["économie", "marché", "finance"],
                "de": ["wirtschaft", "markt", "finanzen"],
                "it": ["economia", "mercato", "finanza"],
                "pt": ["economia", "mercado", "finanças"],
            },
        },
        "politics": {
            "default": {
                "es": ["politica", "gobierno", "parlamento"],
                "en": ["politics", "government", "parliament"],
                "fr": ["politique", "gouvernement", "parlement"],
                "de": ["politik", "regierung", "parlament"],
                "it": ["politica", "governo", "parlamento"],
                "pt": ["política", "governo", "parlamento"],
            },
        },
        "default": {
            "default": {
                "es": ["noticias"],
                "en": ["news"],
                "fr": ["actualités"],
                "de": ["nachrichten"],
                "it": ["notizie"],
                "pt": ["notícias"],
            },
        },
    }

    _LOCALIZED_TOPIC_TERMS: dict[str, dict[str, dict[str, list[str]]]] = {
        "south_korea": {
            "security": {"ko": ["치안", "경찰", "범죄", "사건", "안전", "수사"]},
            "economy": {"ko": ["경제", "시장", "투자", "기업"]},
            "politics": {"ko": ["정치", "정부", "의회", "선거"]},
        },
        "japan": {
            "security": {"ja": ["治安", "警察", "犯罪", "事件", "捜査"]},
            "economy": {"ja": ["経済", "市場", "投資", "企業"]},
            "politics": {"ja": ["政治", "政府", "国会", "選挙"]},
        },
    }

    _LOCALIZED_GEOGRAPHY: dict[str, dict[str, str]] = {
        "south_korea": {"ko": "한국"},
        "japan": {"ja": "日本"},
    }

    _HORIZON_TERMS: dict[str, dict[str, str]] = {
        "week": {
            "es": "esta semana",
            "en": "this week",
            "fr": "cette semaine",
            "de": "diese woche",
            "it": "questa settimana",
            "pt": "esta semana",
            "south_korea": "이번 주",
            "japan": "今週",
        },
        "today": {
            "es": "hoy",
            "en": "today",
            "fr": "aujourd'hui",
            "de": "heute",
            "it": "oggi",
            "pt": "hoje",
            "south_korea": "오늘",
            "japan": "今日",
        },
    }

    def __init__(self, debug_hook: Optional[Callable[..., None]] = None) -> None:
        self._debug_hook = debug_hook
        self._language_resolver = GeoLanguageResolver()

    def _debug(self, label: str, **data: object) -> None:
        if self._debug_hook is None:
            return
        self._debug_hook(label, **data)

    def build_terms(self, context: QueryLocalizationContext) -> list[str]:
        topic_bucket = self._TOPIC_TERMS.get(context.topic, self._TOPIC_TERMS["default"])
        mode = "public_safety" if context.topic == "security" and context.public_safety_query else "default"
        language_terms = topic_bucket[mode]
        base_terms = list(language_terms.get("es", [])) + list(language_terms.get("en", []))
        local_terms = self._local_terms_for_context(context)

        terms: list[str] = []
        seen: set[str] = set()
        for value in [context.geography, context.geo_en, *base_terms, *local_terms, *self._horizon_terms(context)]:
            cleaned = str(value or "").strip()
            lowered = cleaned.lower()
            if cleaned and lowered not in seen:
                seen.add(lowered)
                terms.append(cleaned)

        self._debug(
            "query_localizer.build_terms",
            query_source_group=context.query_source_group,
            topic=context.topic,
            public_safety_query=context.public_safety_query,
            local_terms=local_terms,
            terms=terms,
        )
        return terms

    def build_queries(
        self,
        *,
        domain: str,
        press_name: str,
        context: QueryLocalizationContext,
    ) -> list[str]:
        return [spec.query for spec in self.build_query_specs(domain=domain, press_name=press_name, context=context)]

    def build_query_specs(
        self,
        *,
        domain: str,
        press_name: str,
        context: QueryLocalizationContext,
    ) -> list[QuerySpec]:
        terms = self.build_terms(context)
        topical_variants = self._topic_variants(context, domain=domain)
        queries: list[str] = []
        query_specs: list[QuerySpec] = []
        seen: set[str] = set()
        query_families: list[dict[str, object]] = []

        for language, *variant_parts in topical_variants:
            parts = [f"site:{domain}", *[part for part in variant_parts if part]]
            query = " ".join(str(part).strip() for part in parts if str(part).strip())
            if query and query not in seen:
                seen.add(query)
                queries.append(query)
                query_families.append({"language": language, "query": query})
                query_specs.append(QuerySpec(language=language, query=query))

        if not queries:
            fallback = " ".join([f"site:{domain}", *terms]).strip()
            queries.append(fallback)
            query_families.append({"language": "fallback", "query": fallback})
            query_specs.append(QuerySpec(language="fallback", query=fallback))

        self._debug(
            "query_localizer.build_queries",
            domain=domain,
            press_name=press_name,
            query_source_group=context.query_source_group,
            topic=context.topic,
            public_safety_query=context.public_safety_query,
            language_priority=self._language_resolver.resolve(
                country_group=context.query_source_group,
                domain=domain,
            ),
            query_families=query_families,
            queries=queries,
        )
        return query_specs

    def _topic_variants(self, context: QueryLocalizationContext, *, domain: str) -> list[list[str]]:
        language_priority = self._language_resolver.resolve(
            country_group=context.query_source_group,
            domain=domain,
        )
        language_priority = language_priority[: self._MAX_LANGUAGES_PER_DOMAIN]
        variants: list[list[str]] = []
        for language in language_priority:
            variants.extend(self._variants_for_language(context, language))
            if len(variants) >= self._MAX_QUERY_SPECS_PER_DOMAIN:
                break
        return variants[: self._MAX_QUERY_SPECS_PER_DOMAIN]

    def _variants_for_language(self, context: QueryLocalizationContext, language: str) -> list[list[str]]:
        geography = self._geography_for_language(context, language)
        horizon = self._horizon_term(context, language)
        topic_terms = self._topic_terms_for_language(context, language)[: self._MAX_TERMS_PER_LANGUAGE]
        return [[language, geography, term, horizon] for term in topic_terms if geography and term]

    def _topic_terms_for_language(self, context: QueryLocalizationContext, language: str) -> list[str]:
        topic_bucket = self._TOPIC_TERMS.get(context.topic, self._TOPIC_TERMS["default"])
        mode = "public_safety" if context.topic == "security" and context.public_safety_query else "default"
        terms = list(topic_bucket[mode].get(language, []))
        localized_terms = self._localized_terms_map(context).get(language, [])
        if localized_terms:
            terms.extend(localized_terms)
        deduped: list[str] = []
        seen: set[str] = set()
        for term in terms:
            lowered = term.lower()
            if term and lowered not in seen:
                seen.add(lowered)
                deduped.append(term)
        return deduped

    def _geography_for_language(self, context: QueryLocalizationContext, language: str) -> str:
        if language == "en":
            return context.geo_en
        if language == "es":
            return context.geography
        if language in {"fr", "de", "it", "pt", "ko", "ja", "zh"}:
            return self._LOCALIZED_GEOGRAPHY.get(context.query_source_group or "", {}).get(language, context.geo_en)
        return self._LOCALIZED_GEOGRAPHY.get(context.query_source_group or "", {}).get(language, context.geo_en)

    def _horizon_terms(self, context: QueryLocalizationContext) -> list[str]:
        if not context.horizon:
            return []
        values = self._HORIZON_TERMS.get(context.horizon, {})
        ordered = [values.get("es"), values.get("en")]
        localized_terms = self._localized_terms_map(context)
        for language in localized_terms:
            if language not in {"es", "en"}:
                ordered.append(values.get(context.query_source_group or ""))
        return [value for value in ordered if value]

    def _horizon_term(self, context: QueryLocalizationContext, language: str) -> str:
        values = self._HORIZON_TERMS.get(context.horizon or "", {})
        if language == "es":
            return values.get("es", "")
        if language == "en":
            return values.get("en", "")
        if language in {"fr", "de", "it", "pt"}:
            return values.get(language, "")
        return values.get(context.query_source_group or "", "")

    def _localized_terms_map(self, context: QueryLocalizationContext) -> dict[str, list[str]]:
        return self._LOCALIZED_TOPIC_TERMS.get(context.query_source_group or "", {}).get(context.topic, {})

    def _local_terms_for_context(self, context: QueryLocalizationContext) -> list[str]:
        flattened: list[str] = []
        for terms in self._localized_terms_map(context).values():
            flattened.extend(terms)
        return flattened


__all__ = ["GeoLanguageResolver", "LocalizedNewsQueryBuilder", "QueryLocalizationContext", "QuerySpec"]
