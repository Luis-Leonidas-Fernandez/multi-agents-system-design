"""Modelos de dominio para candidatos y fuentes web."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypedDict


class CandidateDict(TypedDict, total=False):
    """Shape of a raw search candidate (Tavily/SearXNG result)."""
    url: str
    title: str
    snippet: str
    source_kind: str
    source_label: str


class SourceDict(TypedDict, total=False):
    """Shape of a citation source reference."""
    url: str
    title: str


class WebDigestSection(TypedDict, total=False):
    title: str
    topic: str
    source: SourceDict
    bullets: list[str]


class WebDigestContract(TypedDict, total=False):
    version: str
    intro: str
    sections: list[WebDigestSection]
    conclusion: str
    sources: list[SourceDict]


class SourceKind(str, Enum):
    ARTICLE = "article_hit"
    SECTION = "section_hit"
    HOMEPAGE = "homepage_hit"
    TOPIC = "topic_hit"
    HUB = "hub_hit"


class EvidenceKind(str, Enum):
    SEARCH_SNIPPET = "search_snippet"
    FETCHED_ARTICLE = "fetched_article"
    SECTION_LINES = "section_lines"


class Recency(str, Enum):
    DATED_RECENT = "dated_recent"
    DATED_OLD = "dated_old"
    UNDATED = "undated"


class Specificity(str, Enum):
    CONCRETE = "concrete"
    BROAD = "broad"
    STRUCTURAL = "structural"


@dataclass(frozen=True)
class WebCandidate:
    title: str
    url: str
    snippet: str = ""
    source_kind: SourceKind = SourceKind.TOPIC
    evidence_kind: EvidenceKind = EvidenceKind.SEARCH_SNIPPET
    recency: Recency = Recency.UNDATED
    specificity: Specificity = Specificity.BROAD
    source_label: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source_kind": self.source_kind.value,
            "source_label": self.source_label,
        }

    def as_candidate(self) -> dict[str, str]:
        return self.as_dict()
