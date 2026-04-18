"""Modelos de dominio para candidatos y fuentes web."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
class WebSource:
    title: str
    url: str
    domain: str = ""
    source_kind: SourceKind | None = None


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
