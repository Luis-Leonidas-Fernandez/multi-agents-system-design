"""Clasificación y ranking de candidatos web.

Contiene lógica de dominio para determinar tipo, recencia y especificidad
de candidatos de búsqueda. Sin dependencias de capa de aplicación.
"""
import re
from typing import Optional
from urllib.parse import urlparse

from features.web_scraping.domain.text_utils import (
    _TITLE_STOPWORDS,
    _text_keywords,
    _strip_accents,
    _candidate_url_has_date,
    _candidate_url_is_recent,
)
from features.web_scraping.domain.models import (
    EvidenceKind,
    Recency,
    SourceKind,
    Specificity,
    WebCandidate,
)

_NON_NEWS_DOMAINS = {
    "travel", "tourism", "tripadvisor", "lonelyplanet", "fodors", "frommers",
    "wikivoyage", "wikipedia", "wikitravel", "about.com", "tripsavvy",
    "smartertravel", "booking.com", "expedia", "airbnb",
    # statistics / data aggregators — evergreen content, never news
    "numbeo.com", "statista.com", "macrotrends.net", "worldometers.info",
    "tradingeconomics.com", "indexmundi.com", "globaleconomy.com",
    "countrymeters.info", "globalterrorismindex.org", "visionofhumanity.org",
    # think tanks / advocacy orgs — policy analysis, not news reporting
    "brennancenter.org", "aclu.org", "cato.org", "heritage.org",
    "brookings.edu", "cfr.org", "chathamhouse.org", "sipri.org",
    "amnesty.org", "hrw.org", "freedomhouse.org",
    "dialogopolitico.org", "csis.org", "rand.org", "wilsoncenter.org",
    # government travel advisory portals — evergreen safety ratings, not news
    "osac.gov", "travel.state.gov", "smartraveller.gov.au",
    "travel.gc.ca", "gov.uk/foreign-travel-advice",
    # travel/community forums — threads stay active for years, not current news
    "losviajeros.com", "foro.travel", "viajeros.com", "tripadvisor.com",
    "lonelyplanet.com/thorntree", "reddit.com/r/travel",
}

_FORUM_PATH_SEGMENTS = {
    "/foros/", "/forum/", "/forums/", "/thread/", "/threads/",
    "/topic/", "/topics/", "/post/", "/posts/", "/discussion/",
    "/comunidad/", "/community/", "/board/", "/boards/",
}


def _is_non_news_candidate(candidate: dict[str, str]) -> bool:
    """Returns True if the candidate looks like evergreen/travel/wiki/government-PR content rather than news."""
    url = candidate.get("url", "").lower()
    title = candidate.get("title", "").lower()
    snippet = candidate.get("snippet", "").lower()

    if any(domain in url for domain in _NON_NEWS_DOMAINS):
        return True

    if any(seg in url for seg in _FORUM_PATH_SEGMENTS):
        return True

    _GOV_TLD = (".gob.", ".gov.", "/gob.", "/gov.")
    _PR_PATHS = ("/prensa/", "/comunicado", "/nota-de-prensa", "/press-release", "/sala-de-prensa")
    if any(tld in url for tld in _GOV_TLD) and any(path in url for path in _PR_PATHS):
        return True

    _LEGAL_PATHS = (
        "/legal-update", "/client-alert", "/client-advisory", "/legal-alert",
        "/publications/", "/publication/", "/insights/", "/knowledge/",
        "/briefing/", "/memorandum/", "/legal-news/",
    )
    if any(seg in url for seg in _LEGAL_PATHS):
        return True

    evergreen_signals = [
        "se recomienda a los viajeros", "se aconseja a los viajeros",
        "para los turistas", "consejos de seguridad", "guía de viaje",
        "recomendaciones para viajeros", "baja tasa de criminalidad",
        "travel advisory", "safety tips for travelers",
        "ejercer mayor precaución", "ejercer precaución",
        "se desaconseja viajar", "no se recomienda viajar",
        "nivel de alerta de viaje", "travel level", "do not travel",
        "reconsider travel", "exercise increased caution",
        "high threat location", "ubicación de alta amenaza",
    ]
    return any(signal in snippet or signal in title for signal in evergreen_signals)


def _same_event(
    candidate_a: dict[str, str],
    candidate_b: dict[str, str],
    query_terms: Optional[list[str]] = None,
) -> bool:
    """Returns True if two candidates appear to describe the same event.

    Title overlap ≥ 3 shared keywords (excluding query terms) → same event.
    Full-text (title+snippet) overlap ≥ 5 non-query keywords → same event.

    Thresholds intentionally conservative: for topic-rich queries (e.g. Japan security),
    many articles share 2 generic words (china, tensiones) without covering the same story.
    Requiring 3 title-keyword overlap prevents false deduplication across genuinely
    distinct events, which is what produces 4 diverse bullets.
    """
    excluded = set(t.lower() for t in (query_terms or []))
    excluded.update(_TITLE_STOPWORDS)

    def keywords(text: str) -> set[str]:
        return {w.lower() for w in text.split() if len(w) > 4 and w.lower() not in excluded}

    title_kw_a = keywords(candidate_a.get("title", ""))
    title_kw_b = keywords(candidate_b.get("title", ""))
    if len(title_kw_a & title_kw_b) >= 3:
        return True

    full_a = keywords(f"{candidate_a.get('title', '')} {candidate_a.get('snippet', '')}")
    full_b = keywords(f"{candidate_b.get('title', '')} {candidate_b.get('snippet', '')}")
    return len(full_a & full_b) >= 5


def _dedup_candidates_by_event(
    candidates: list[dict[str, str]],
    query_terms: Optional[list[str]] = None,
) -> list[dict[str, str]]:
    """Keep one candidate per event — drop articles that appear to cover the same story."""
    accepted: list[dict[str, str]] = []
    for candidate in candidates:
        if not any(_same_event(candidate, a, query_terms) for a in accepted):
            accepted.append(candidate)
    return accepted


def _extract_generic_search_candidates(search_text: str) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    current: Optional[dict[str, str]] = None

    for line in [line.rstrip() for line in (search_text or "").splitlines() if line.strip()]:
        item_match = re.match(r"^\d+\. (?:\[(article|hub)\]\s*)?\[(.+?)\]\((https?://[^)]+)\)", line.strip())
        if item_match:
            if current:
                candidates.append(current)
            tag = item_match.group(1) or ""
            current = {
                "title": item_match.group(2).strip(),
                "url": item_match.group(3).strip(),
                "snippet": "",
                "hit_type": tag,
            }
            continue

        if current is not None and not line.startswith("Sources:") and not line.startswith("-") and not line.startswith("Call web_fetch") and not line.startswith("Next step"):
            snippet = line.strip()
            if snippet:
                current["snippet"] = (current.get("snippet", "") + " " + snippet).strip()

    if current:
        candidates.append(current)

    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for candidate in candidates:
        url = candidate.get("url", "")
        if url and url not in seen:
            seen.add(url)
            deduped.append(candidate)
    return deduped


def _candidate_snippet_lines(candidate: dict[str, str]) -> list[str]:
    snippet = (candidate.get("snippet") or "").strip()
    if not snippet:
        return []
    snippet = re.sub(r"^#+\s+", "", snippet)
    snippet = re.sub(r"\s+#+\s+", " ", snippet).strip()
    if len(snippet.split()) < 4:
        return []
    return [snippet]


def _is_hub_like_candidate(candidate: dict[str, str]) -> bool:
    url = (candidate.get("url") or "").lower()
    hit_type = (candidate.get("hit_type") or "").lower()
    if hit_type == "hub":
        return True
    path = urlparse(url).path.lower().rstrip("/")
    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        return True
    structurally_invalid_segments = {
        "edizioni",
        "editioni",
        "dalle_sezioni_mobile.html",
        "gli-inserti-del-foglio",
        "conosci-i-foglianti",
        "ultima-ora",
    }
    if any(segment in structurally_invalid_segments for segment in segments):
        return True
    if any(token in path for token in ("/edizioni/", "/dalle_sezioni_mobile", "/gli-inserti-del-foglio", "/conosci-i-foglianti", "/t/")):
        return True
    if any(token in path for token in ("/tag/", "/tags/", "/autori/", "/authors/", "/argomenti/", "/rubriche/")):
        return True
    if path.endswith(("/news.shtml", "/index.shtml", "/cronaca.shtml", "/politica.shtml", "/economia.shtml")):
        return True
    if segments[-1] in {"politica", "cronaca", "economia", "sport", "archive", "archivio", "topnews", "ultimaora"}:
        return True
    if (
        len(segments) <= 2
        and any(seg in {"politica", "cronaca", "economia", "sport"} for seg in segments)
        and not any("-" in seg or "_" in seg or re.search(r"\d", seg) for seg in segments)
    ):
        return True
    return False


def _query_targets_public_safety(query: str) -> bool:
    normalized = _strip_accents((query or "").lower())
    signals = (
        "seguridad",
        "security",
        "sicurezza",
        "policia",
        "police",
        "polizia",
        "crime",
        "crimen",
        "cronaca",
        "public safety",
        "orden publico",
        "orden público",
    )
    return any(signal in normalized for signal in signals)


def _is_tangential_vertical_candidate(candidate: dict[str, str], query: str) -> bool:
    if not _query_targets_public_safety(query):
        return False
    path = urlparse(candidate.get("url", "")).path.lower()
    title = (candidate.get("title") or "").lower()
    blob = f"{path} {title}"
    return any(
        token in blob
        for token in (
            "/canale_motori/",
            "/motori/",
            "/auto/",
            "/sicurezza-informatica",
            "sicurezza informatica",
            "cybersecurity",
            "ciberseguridad",
            "motori",
            "automotive",
            "sicurezza stradale",
            "sicurezza vial",
            "road safety",
        )
    )


def _is_invalid_news_candidate(candidate: dict[str, str], query: str) -> bool:
    return _is_hub_like_candidate(candidate) or _is_tangential_vertical_candidate(candidate, query)


def _classify_candidate_source_kind(candidate: dict[str, str]) -> SourceKind:
    if _is_hub_like_candidate(candidate):
        return SourceKind.HUB
    if candidate.get("source_kind") == "section_fallback":
        return SourceKind.SECTION
    if candidate.get("source_kind") == "homepage_fallback":
        return SourceKind.HOMEPAGE
    if _is_specific_article_hit(candidate):
        return SourceKind.ARTICLE
    return SourceKind.TOPIC


def _classify_candidate_recency(candidate: dict[str, str], query_horizon: Optional[str]) -> Recency:
    url = str(candidate.get("url") or "")
    if _candidate_url_has_date(url):
        threshold = 45 if query_horizon == "month" else 14 if query_horizon == "week" else 2 if query_horizon == "today" else 30
        return Recency.DATED_RECENT if _candidate_url_is_recent(url, threshold) else Recency.DATED_OLD
    if candidate.get("source_kind") == "section_fallback":
        return Recency.DATED_RECENT
    return Recency.UNDATED


def _classify_candidate_specificity(candidate: dict[str, str], query: str) -> Specificity:
    if _is_invalid_news_candidate(candidate, query):
        return Specificity.STRUCTURAL
    if candidate.get("source_kind") == "section_fallback" or _is_specific_article_hit(candidate):
        return Specificity.CONCRETE
    return Specificity.BROAD


def _candidate_record_from_dict(candidate: dict[str, str], *, query: str, query_horizon: Optional[str]) -> WebCandidate:
    source_kind = _classify_candidate_source_kind(candidate)
    evidence_kind = EvidenceKind.SECTION_LINES if source_kind == SourceKind.SECTION else EvidenceKind.SEARCH_SNIPPET
    recency = _classify_candidate_recency(candidate, query_horizon)
    specificity = _classify_candidate_specificity(candidate, query)
    return WebCandidate(
        title=str(candidate.get("title") or candidate.get("url") or "result"),
        url=str(candidate.get("url") or ""),
        snippet=str(candidate.get("snippet") or ""),
        source_kind=source_kind,
        evidence_kind=evidence_kind,
        recency=recency,
        specificity=specificity,
        source_label=str(candidate.get("source_label") or ""),
    )


def _candidate_strategy_priority(candidate: dict[str, str], *, query: str, query_horizon: Optional[str]) -> tuple[int, int, int, int]:
    record = _candidate_record_from_dict(candidate, query=query, query_horizon=query_horizon)
    source_rank = {
        "section_hit": 0,
        "article_hit": 1,
        "homepage_hit": 2,
        "topic_hit": 3,
        "hub_hit": 4,
    }.get(record.source_kind, 5)
    specificity_rank = {"concrete": 0, "broad": 1, "structural": 2}.get(record.specificity, 3)
    recency_rank = {"dated_recent": 0, "undated": 1, "dated_old": 2}.get(record.recency, 3)
    return (source_rank, specificity_rank, recency_rank, -len(record.snippet.split()))


def _hit_path_segments(link: str) -> list[str]:
    path = urlparse(link or "").path.strip("/")
    if not path:
        return []
    return [segment for segment in path.split("/") if segment]


def _hit_path_tokens(link: str) -> list[str]:
    tokens: list[str] = []
    for segment in _hit_path_segments(link):
        for token in re.split(r"[-_]+", segment):
            token = token.strip().lower()
            if token:
                tokens.append(token)
    return tokens


def _path_has_date_or_slug(link: str) -> bool:
    path = urlparse(link or "").path.lower()
    segments = _hit_path_segments(link)
    return bool(
        re.search(r"(19|20)\d{2}[/\-]?\d{2}[/\-]?\d{2}", path)
        or re.search(r"\d{6,8}", path)
        or any("-" in seg or "_" in seg or re.search(r"\d", seg) for seg in segments)
    )


def _blob_has_article_signal(blob: str) -> bool:
    article_signals = (
        "security", "safety", "economy", "economic", "politics", "political", "election",
        "breaking", "update", "reported", "report", "announced", "announcement", "court",
        "attack", "disaster", "conflict", "strike", "result", "results", "resultado", "resultados",
        "match", "game", "policy", "strategy", "minister", "president", "prime minister",
    )
    return any(term in blob for term in article_signals)


def _is_topic_or_hub_hit(hit: dict[str, str]) -> bool:
    link = str(hit.get("url") or hit.get("link") or "")
    if not link:
        return False
    segments = _hit_path_segments(link)
    blob = " ".join(str(hit.get(field) or "") for field in ("title", "url", "link", "content", "snippet")).lower()
    tokens = _hit_path_tokens(link)
    hub_terms = {"topic", "topics", "tag", "tags", "category", "categories", "archive", "author", "world", "mundo", "index", "home", "partidos", "resultados", "ultima-ora"}
    if "/t/" in urlparse(link).path.lower():
        return True
    if any(tok in hub_terms for tok in tokens):
        return True
    if len(segments) >= 1 and segments[0] in {"news", "noticias", "world", "mundo", "partidos", "resultados", "home", "index"} and not _path_has_date_or_slug(link):
        if segments[0] in {"news", "noticias"} and _blob_has_article_signal(blob):
            return False
        return True
    has_slug = any("-" in seg or "_" in seg or re.search(r"\d", seg) for seg in segments)
    if len(segments) <= 2 and not _path_has_date_or_slug(link) and not has_slug and not _blob_has_article_signal(blob):
        return True
    return False


def _is_specific_article_hit(hit: dict[str, str]) -> bool:
    link = str(hit.get("url") or hit.get("link") or "")
    if not link:
        return False
    path = urlparse(link).path.lower()
    segments = _hit_path_segments(link)
    tokens = _hit_path_tokens(link)
    blob = " ".join(str(hit.get(field) or "") for field in ("title", "url", "link", "content", "snippet")).lower()
    if re.search(r"(19|20)\d{2}[/\-]?\d{2}[/\-]?\d{2}", path) or re.search(r"\d{6,8}", path):
        return True
    if _is_topic_or_hub_hit(hit):
        return False
    if any(tok in {"news", "noticias"} for tok in tokens) and _blob_has_article_signal(blob):
        return True
    if any(tok in {"topic", "topics", "tag", "tags", "category", "categories", "archive", "author", "world", "mundo", "index", "home", "partidos", "resultados"} for tok in tokens):
        return False
    if len(segments) >= 2 and _blob_has_article_signal(blob):
        return True
    if len(segments) == 1 and any("-" in seg or "_" in seg for seg in segments) and _blob_has_article_signal(blob):
        return True
    if any("-" in seg or "_" in seg or re.search(r"\d", seg) for seg in segments):
        return True
    return len(segments) >= 3
