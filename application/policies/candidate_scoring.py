"""Política de scoring y ranking de candidatos web.

Centraliza los criterios de puntuación para que no queden atrapados
dentro del use_case web_scraping_flow.
"""
from dataclasses import dataclass
import re
from typing import Optional
from urllib.parse import urlparse

from features.web_scraping.domain.classifier import _is_hub_like_candidate, _query_targets_public_safety
from features.web_scraping.domain.models import CandidateDict
from features.web_scraping.domain.text_utils import _strip_accents
from application.policies.web_source_policy import score_domain_boost, get_source_domain_priority

# --- Scores positivos ---
SCORE_TERM_MATCH = 3          # término de la query encontrado en blob
SCORE_PRICE_RANGE = 2         # patrón numérico tipo "100 - 200" en blob
SCORE_DATE_IN_URL = 4         # fecha YYYYMMDD o YYYY/MM/DD en path
SCORE_DEEP_URL = 2            # URL con 3 o más segmentos de path

# --- Penalizaciones ---
PENALTY_SHALLOW_URL = -3      # URL corta sin fecha → probable hub/listing
PENALTY_NAV_SEGMENT = -4      # segmento de navegación (tag, category, archive…)
PENALTY_NOISE_WORD = -2       # palabra de ruido (login, cookie, privacy…)
PENALTY_NO_TITLE_MATCH = -6   # título sin ningún término significativo de la query
PENALTY_HOMEPAGE_FALLBACK = -8
PENALTY_SECTION_FALLBACK = -3
PENALTY_HUB_LIKE = -12        # candidato que parece portada/hub de sitio
BOOST_PUBLIC_SAFETY_SIGNAL = 5
BOOST_PUBLIC_SAFETY_URL = 3
PENALTY_GEOPOLITICAL_SECURITY = -5
PENALTY_EDITORIAL_SECURITY = -6

_NAV_SEGMENTS = {"topic", "topics", "tag", "tags", "category", "categories", "archive", "author"}
_NOISE_WORDS = {"login", "signin", "cookie", "privacy", "archive", "perfil"}
_PUBLIC_SAFETY_TERMS = {
    "inseguridad", "policial", "policiales", "policia", "police", "crime", "crimen",
    "delito", "delincuencia", "robo", "robbery", "homicid", "murder", "asesinat",
    "operativo", "raid", "detenid", "arrest", "sucesos", "justicia", "tribunal",
    "fiscal", "allanamiento", "tiroteo", "apuñal", "amenaza", "evacuacion", "evacuación",
}
_GEOPOLITICAL_SECURITY_TERMS = {
    "defensa", "defense", "militar", "military", "alliance", "alianza", "diplom",
    "foreign policy", "politica exterior", "summit", "cumbre", "tariff", "tarifa",
    "submarine", "submarin", "nuclear", "nato", "washington", "japan", "japon",
}
_EDITORIAL_TERMS = {"editorial", "column", "columna", "opinion", "analysis", "analisis", "análisis"}

_COUNTRY_LANGUAGE_HINTS: dict[str, tuple[str, ...]] = {
    "south_korea": ("ko", "en", "es"),
    "japan": ("ja", "en", "es"),
    "china": ("zh", "en", "es"),
    "default": ("en", "es"),
}

_SECURITY_SUBTYPES: dict[str, tuple[str, ...]] = {
    "public_safety": (
        "seguridad", "security", "inseguridad", "sicurezza", "policia", "police", "polizia",
        "policial", "policiales", "crime", "crimen", "delito", "delincuencia", "robo",
        "homicidio", "sucesos", "cronaca", "public safety", "orden publico", "orden público",
    ),
    "national_security": (
        "defensa", "defense", "militar", "military", "misil", "missile", "nuclear", "guerra",
        "war", "border", "frontera", "army", "navy", "air force", "intel", "intelligence",
    ),
    "cybersecurity": (
        "ciberseguridad", "cybersecurity", "hackeo", "hacking", "ransomware", "phishing",
        "malware", "data breach", "brecha de datos",
    ),
}

_INTENT_PROFILES: dict[str, dict[str, tuple[str, ...]]] = {
    "security.public_safety": {
        "positive_concepts": (
            "police", "crime", "incident", "investigation", "court", "arrest", "murder",
            "assault", "fraud", "public_safety", "justice", "safety", "emergency",
        ),
        "preferred_section_concepts": ("society", "national", "local", "crime", "police", "court", "justice"),
        "tangential_concepts": (
            "international", "world", "foreign_policy", "diplomacy", "military",
            "missile", "nuclear", "war", "summit", "north_korea", "defense",
        ),
        "broad_section_concepts": ("latest", "breaking", "all_news"),
    },
    "security.national_security": {
        "positive_concepts": ("military", "defense", "missile", "nuclear", "border", "intelligence", "war"),
        "preferred_section_concepts": ("defense", "military", "politics", "international", "world"),
        "tangential_concepts": ("local_crime", "traffic_accident", "celebrity_scandal"),
        "broad_section_concepts": ("latest", "all_news"),
    },
    "security.cybersecurity": {
        "positive_concepts": ("cyber", "hack", "ransomware", "data_breach", "malware", "phishing", "digital_crime"),
        "preferred_section_concepts": ("technology", "security", "business", "crime"),
        "tangential_concepts": ("physical_crime", "traffic_accident", "military_drill"),
        "broad_section_concepts": ("latest", "all_news"),
    },
}

_CONCEPT_TERMS: dict[str, dict[str, tuple[str, ...]]] = {
    "police": {"es": ("policia", "policía"), "en": ("police",), "ko": ("경찰",), "ja": ("警察",), "zh": ("警察",)},
    "crime": {"es": ("crimen", "delito", "criminal"), "en": ("crime", "criminal"), "ko": ("범죄",), "ja": ("犯罪",), "zh": ("犯罪",)},
    "incident": {"es": ("incidente", "hecho"), "en": ("incident",), "ko": ("사건",), "ja": ("事件",), "zh": ("事件",)},
    "investigation": {"es": ("investigacion", "investigación", "investiga"), "en": ("investigation", "investigating"), "ko": ("수사", "조사"), "ja": ("捜査", "調査"), "zh": ("调查", "调查中")},
    "court": {"es": ("tribunal", "juzgado"), "en": ("court",), "ko": ("법원",), "ja": ("裁判所",), "zh": ("法院",)},
    "arrest": {"es": ("detenido", "arrestado", "capturado"), "en": ("arrest", "arrested", "detained"), "ko": ("체포", "구속"), "ja": ("逮捕",), "zh": ("逮捕",)},
    "murder": {"es": ("homicidio", "asesinato"), "en": ("murder", "homicide"), "ko": ("살인",), "ja": ("殺人",), "zh": ("谋杀", "杀人")},
    "assault": {"es": ("asalto", "agresion", "agresión"), "en": ("assault",), "ko": ("폭행",), "ja": ("暴行",), "zh": ("袭击",)},
    "fraud": {"es": ("fraude", "estafa"), "en": ("fraud", "scam"), "ko": ("사기",), "ja": ("詐欺",), "zh": ("诈骗",)},
    "public_safety": {"es": ("seguridad publica", "seguridad pública", "inseguridad"), "en": ("public safety", "safety"), "ko": ("치안", "안전"), "ja": ("治安", "安全"), "zh": ("治安", "安全")},
    "justice": {"es": ("justicia", "fiscal"), "en": ("justice", "prosecutor"), "ko": ("사법", "검찰"), "ja": ("司法", "検察"), "zh": ("司法", "检察")},
    "safety": {"es": ("seguridad", "seguro"), "en": ("safety", "security"), "ko": ("안전",), "ja": ("安全",), "zh": ("安全",)},
    "emergency": {"es": ("emergencia", "evacuacion", "evacuación"), "en": ("emergency", "evacuation"), "ko": ("비상", "대피"), "ja": ("緊急", "避難"), "zh": ("紧急", "疏散")},
    "society": {"es": ("sociedad",), "en": ("society", "social affairs"), "ko": ("사회",), "ja": ("社会",), "zh": ("社会",)},
    "national": {"es": ("nacional",), "en": ("national",), "ko": ("국내", "전국"), "ja": ("国内",), "zh": ("国内",)},
    "local": {"es": ("local",), "en": ("local", "metro"), "ko": ("지역", "로컬"), "ja": ("地域", "地方"), "zh": ("地方", "本地")},
    "international": {"es": ("internacional", "mundo"), "en": ("international", "world"), "ko": ("국제",), "ja": ("国際",), "zh": ("国际",)},
    "world": {"es": ("mundo",), "en": ("world",), "ko": ("세계",), "ja": ("世界",), "zh": ("世界",)},
    "foreign_policy": {"es": ("politica exterior", "política exterior"), "en": ("foreign policy",), "ko": ("외교",), "ja": ("外交",), "zh": ("外交",)},
    "diplomacy": {"es": ("diplomacia",), "en": ("diplomacy", "diplomatic"), "ko": ("외교",), "ja": ("外交",), "zh": ("外交",)},
    "military": {"es": ("militar",), "en": ("military",), "ko": ("군사", "군"), "ja": ("軍事", "軍"), "zh": ("军事", "军方")},
    "missile": {"es": ("misil", "misiles"), "en": ("missile", "missiles"), "ko": ("미사일",), "ja": ("ミサイル",), "zh": ("导弹",)},
    "nuclear": {"es": ("nuclear",), "en": ("nuclear",), "ko": ("핵", "비핵화"), "ja": ("核", "非核化"), "zh": ("核", "无核化")},
    "war": {"es": ("guerra",), "en": ("war",), "ko": ("전쟁",), "ja": ("戦争",), "zh": ("战争",)},
    "summit": {"es": ("cumbre",), "en": ("summit",), "ko": ("정상회담",), "ja": ("首脳会談",), "zh": ("峰会",)},
    "north_korea": {"es": ("corea del norte",), "en": ("north korea",), "ko": ("북한",), "ja": ("北朝鮮",), "zh": ("朝鲜", "北韩")},
    "defense": {"es": ("defensa",), "en": ("defense",), "ko": ("국방",), "ja": ("防衛",), "zh": ("国防",)},
    "latest": {"es": ("ultimas", "últimas", "ultima hora"), "en": ("latest", "breaking"), "ko": ("최신기사", "속보"), "ja": ("最新", "速報"), "zh": ("最新", "快讯")},
    "breaking": {"es": ("urgente", "último momento"), "en": ("breaking",), "ko": ("속보",), "ja": ("速報",), "zh": ("突发",)},
    "all_news": {"es": ("todas las noticias",), "en": ("all news",), "ko": ("전체기사", "모든기사"), "ja": ("すべての記事",), "zh": ("全部新闻",)},
    "politics": {"es": ("politica", "gobierno"), "en": ("politics", "government"), "ko": ("정치", "정부"), "ja": ("政治", "政府"), "zh": ("政治", "政府")},
    "technology": {"es": ("tecnologia", "tecnología"), "en": ("technology", "tech"), "ko": ("기술", "테크"), "ja": ("技術", "テック"), "zh": ("科技", "技术")},
    "business": {"es": ("negocios", "empresa"), "en": ("business",), "ko": ("비즈", "기업"), "ja": ("ビジネス", "企業"), "zh": ("商业", "企业")},
    "cyber": {"es": ("ciber", "ciberseguridad"), "en": ("cyber", "cybersecurity"), "ko": ("사이버",), "ja": ("サイバー",), "zh": ("网络", "网络安全")},
    "hack": {"es": ("hackeo", "hack"), "en": ("hack", "hacking"), "ko": ("해킹",), "ja": ("ハッキング",), "zh": ("黑客", "入侵")},
    "ransomware": {"es": ("ransomware",), "en": ("ransomware",), "ko": ("랜섬웨어",), "ja": ("ランサムウェア",), "zh": ("勒索软件",)},
    "data_breach": {"es": ("brecha de datos", "filtracion de datos", "filtración de datos"), "en": ("data breach",), "ko": ("정보 유출", "데이터 유출"), "ja": ("情報漏えい",), "zh": ("数据泄露",)},
    "malware": {"es": ("malware",), "en": ("malware",), "ko": ("악성코드",), "ja": ("マルウェア",), "zh": ("恶意软件",)},
    "phishing": {"es": ("phishing",), "en": ("phishing",), "ko": ("피싱",), "ja": ("フィッシング",), "zh": ("网络钓鱼",)},
    "digital_crime": {"es": ("delito digital",), "en": ("digital crime",), "ko": ("디지털 범죄",), "ja": ("デジタル犯罪",), "zh": ("数字犯罪",)},
    "local_crime": {"es": ("delito local",), "en": ("local crime",), "ko": ("지역 범죄",), "ja": ("地域犯罪",), "zh": ("本地犯罪",)},
    "traffic_accident": {"es": ("accidente de transito", "accidente de tránsito"), "en": ("traffic accident",), "ko": ("교통사고",), "ja": ("交通事故",), "zh": ("交通事故",)},
    "celebrity_scandal": {"es": ("escandalo", "escándalo"), "en": ("celebrity scandal",), "ko": ("연예 스캔들",), "ja": ("芸能スキャンダル",), "zh": ("明星丑闻",)},
    "physical_crime": {"es": ("crimen fisico", "crimen físico"), "en": ("physical crime",), "ko": ("물리 범죄",), "ja": ("物理犯罪",), "zh": ("实体犯罪",)},
    "military_drill": {"es": ("ejercicio militar",), "en": ("military drill",), "ko": ("군사 훈련",), "ja": ("軍事訓練",), "zh": ("军事演习",)},
}


@dataclass(frozen=True)
class CandidateRelevanceContext:
    topic: str
    intent_subtype: str
    query_source_group: Optional[str]
    languages: tuple[str, ...]
    positive_terms: tuple[str, ...]
    preferred_section_terms: tuple[str, ...]
    tangential_terms: tuple[str, ...]
    broad_section_terms: tuple[str, ...]
    min_relevance_score: int = 8


def _detect_security_subtype(query: str) -> str:
    normalized = _strip_accents((query or "").lower())
    if any(signal in normalized for signal in _SECURITY_SUBTYPES["cybersecurity"]):
        return "cybersecurity"
    if any(signal in normalized for signal in _SECURITY_SUBTYPES["national_security"]):
        return "national_security"
    if any(signal in normalized for signal in _SECURITY_SUBTYPES["public_safety"]) or "security" in normalized or "seguridad" in normalized:
        return "public_safety"
    return "public_safety"


def _detect_query_topic(query: str) -> str:
    normalized = _strip_accents((query or "").lower())
    if any(signal in normalized for signal in ("economia", "economy", "mercado", "market", "finance", "finanzas")):
        return "economy"
    if any(signal in normalized for signal in ("politica", "politics", "gobierno", "government", "election", "eleccion", "elección")):
        return "politics"
    if any(signal in normalized for signal in ("seguridad", "security", "crime", "crimen", "police", "policia", "policía", "cybersecurity", "ciberseguridad", "military", "militar", "defense", "defensa")):
        return "security"
    return "default"


def _query_country_languages(query_source_group: Optional[str]) -> tuple[str, ...]:
    normalized_group = (query_source_group or "default").split(":", 1)[-1] or "default"
    return _COUNTRY_LANGUAGE_HINTS.get(normalized_group, _COUNTRY_LANGUAGE_HINTS["default"])


def _resolve_concept_terms(concepts: tuple[str, ...], languages: tuple[str, ...]) -> tuple[str, ...]:
    ordered_languages = tuple(dict.fromkeys((*languages, "en", "es")))
    resolved: list[str] = []
    seen: set[str] = set()
    for concept in concepts:
        language_map = _CONCEPT_TERMS.get(concept, {})
        values: tuple[str, ...] = tuple(language_map.get("en", ())) if language_map else (concept.replace("_", " "),)
        localized_values: list[str] = []
        if language_map:
            for language in ordered_languages:
                localized_values.extend(language_map.get(language, ()))
        for raw_value in (*values, *localized_values):
            normalized = _strip_accents(raw_value.lower()).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                resolved.append(normalized)
    return tuple(resolved)


def _build_relevance_context(query: str, query_source_group: Optional[str]) -> CandidateRelevanceContext:
    topic = _detect_query_topic(query)
    intent_subtype = _detect_security_subtype(query) if topic == "security" else "default"
    profile_key = f"{topic}.{intent_subtype}"
    profile = _INTENT_PROFILES.get(profile_key, {
        "positive_concepts": (),
        "preferred_section_concepts": (),
        "tangential_concepts": (),
        "broad_section_concepts": (),
    })
    languages = _query_country_languages(query_source_group)
    min_relevance_score = 8 if profile_key == "security.public_safety" else 6 if topic == "security" else 4
    return CandidateRelevanceContext(
        topic=topic,
        intent_subtype=intent_subtype,
        query_source_group=query_source_group,
        languages=languages,
        positive_terms=_resolve_concept_terms(profile["positive_concepts"], languages),
        preferred_section_terms=_resolve_concept_terms(profile["preferred_section_concepts"], languages),
        tangential_terms=_resolve_concept_terms(profile["tangential_concepts"], languages),
        broad_section_terms=_resolve_concept_terms(profile["broad_section_concepts"], languages),
        min_relevance_score=min_relevance_score,
    )


def _normalize_candidate_blob(candidate: CandidateDict) -> tuple[str, str]:
    title = str(candidate.get("title") or "")
    snippet = str(candidate.get("snippet") or "")
    url = str(candidate.get("url") or "")
    source_label = str(candidate.get("source_label") or "")
    blob = _strip_accents(" ".join([title, snippet, url, source_label]).lower())
    section_blob = _strip_accents(" ".join([title, source_label, url]).lower())
    return blob, section_blob


def _candidate_from_broad_section(candidate: CandidateDict, context: CandidateRelevanceContext) -> bool:
    _, section_blob = _normalize_candidate_blob(candidate)
    return any(term in section_blob for term in context.broad_section_terms)


def _score_candidate_relevance(
    candidate: CandidateDict,
    query: str,
    query_source_group: Optional[str],
) -> int:
    context = _build_relevance_context(query, query_source_group)
    if context.topic == "default":
        return 0

    blob, section_blob = _normalize_candidate_blob(candidate)
    score = 0

    for term in context.positive_terms:
        if term in blob:
            score += 3
    for term in context.preferred_section_terms:
        if term in section_blob:
            score += 4
    for term in context.tangential_terms:
        if term in blob:
            score -= 5
    for term in context.broad_section_terms:
        if term in section_blob:
            score -= 2

    if _candidate_from_broad_section(candidate, context) and score < context.min_relevance_score:
        score -= 5

    return score


def _is_relevant_candidate_for_query(
    candidate: CandidateDict,
    query: str,
    query_source_group: Optional[str],
) -> bool:
    context = _build_relevance_context(query, query_source_group)
    return _score_candidate_relevance(candidate, query, query_source_group) >= context.min_relevance_score


def _score_generic_candidate(
    candidate: CandidateDict,
    query_terms: list[str],
    query_source_group: Optional[str] = None,
) -> int:
    blob = " ".join([candidate.get("title", ""), candidate.get("snippet", ""), candidate.get("url", "")]).lower()
    score = 0

    for term in query_terms:
        if term in blob:
            score += SCORE_TERM_MATCH

    if re.search(r"\b\d+\s*-\s*\d+\b", blob):
        score += SCORE_PRICE_RANGE

    url = candidate.get("url", "")
    path = urlparse(url).path.lower()
    segments = [s for s in path.split("/") if s]

    if re.search(r"(19|20)\d{2}[/\-]?\d{2}[/\-]?\d{2}", path) or re.search(r"\d{6,8}", path):
        score += SCORE_DATE_IN_URL
    if len(segments) >= 3:
        score += SCORE_DEEP_URL
    if len(segments) <= 2 and not re.search(r"(19|20)\d{2}[/\-]?\d{2}[/\-]?\d{2}", path):
        score += PENALTY_SHALLOW_URL
    if any(seg in _NAV_SEGMENTS for seg in segments):
        score += PENALTY_NAV_SEGMENT
    if any(noise in blob for noise in _NOISE_WORDS):
        score += PENALTY_NOISE_WORD

    score += score_domain_boost(query_source_group, url)

    # Penalizar candidatos cuyo título no contiene ningún término significativo
    # (longitud ≥ 4 para excluir stopwords cortas). Excepción: section_fallback
    # con el término presente en el snippet.
    title_lower = candidate.get("title", "").lower()
    snippet_lower = candidate.get("snippet", "").lower()
    meaningful_terms = [t for t in query_terms if len(t) >= 4]
    if meaningful_terms and not any(t in title_lower for t in meaningful_terms):
        if candidate.get("source_kind") == "section_fallback" and any(t in snippet_lower for t in meaningful_terms):
            pass
        else:
            score += PENALTY_NO_TITLE_MATCH

    if candidate.get("source_kind") == "homepage_fallback":
        score += PENALTY_HOMEPAGE_FALLBACK
    if candidate.get("source_kind") == "section_fallback":
        score += PENALTY_SECTION_FALLBACK
    if _is_hub_like_candidate(candidate):
        score += PENALTY_HUB_LIKE

    if _query_targets_public_safety(" ".join(query_terms)):
        if any(term in blob for term in _PUBLIC_SAFETY_TERMS):
            score += BOOST_PUBLIC_SAFETY_SIGNAL
        if any(seg in path for seg in ("/policial", "/police", "/crime", "/seguridad", "/inseguridad", "/sucesos", "/justicia")):
            score += BOOST_PUBLIC_SAFETY_URL
        if any(term in blob for term in _GEOPOLITICAL_SECURITY_TERMS):
            score += PENALTY_GEOPOLITICAL_SECURITY
        if any(term in blob for term in _EDITORIAL_TERMS):
            score += PENALTY_EDITORIAL_SECURITY

    score += _score_candidate_relevance(candidate, " ".join(query_terms), query_source_group)

    return score


def _candidate_source_priority(candidate: CandidateDict, query_source_group: Optional[str]) -> int:
    return get_source_domain_priority(query_source_group, candidate.get("url", ""))


def _rank_candidates_by_source_policy(
    candidates: list[CandidateDict],
    query_terms: list[str],
    query_source_group: Optional[str],
) -> list[CandidateDict]:
    if not candidates:
        return []
    if not query_source_group:
        return sorted(
            candidates,
            key=lambda c: _score_generic_candidate(c, query_terms, query_source_group),
            reverse=True,
        )
    return sorted(
        candidates,
        key=lambda c: (
            _candidate_source_priority(c, query_source_group),
            -_score_generic_candidate(c, query_terms, query_source_group),
        ),
    )
