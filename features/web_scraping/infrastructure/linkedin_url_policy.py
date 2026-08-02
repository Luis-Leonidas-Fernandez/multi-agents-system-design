"""Política URL cerrada para el vertical LinkedIn Jobs."""
from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from core.helpers.url_helpers import _validate_public_http_url


_ALLOWED_HOSTS = {"linkedin.com", "www.linkedin.com"}
_JOB_PATH_RE = re.compile(r"^/jobs/(?:$|search/?$|view/(?:\d+|[^/?#]+)/?$)")
_ALLOWED_SEARCH_QUERY_KEYS = {
    "currentJobId",
    "f_TPR",
    "geoId",
    "keywords",
    "location",
    "origin",
    "refresh",
    "sortBy",
    "start",
}
_TRACKING_QUERY_KEYS = {
    "refid",
    "trackingid",
    "trk",
    "trkinfo",
    "lipi",
    "midtoken",
    "ebp",
}


def validate_linkedin_jobs_url(url: str) -> str:
    normalized, reason = _validate_public_http_url(url)
    if not normalized:
        raise ValueError(f"URL LinkedIn rechazada: {reason or 'invalid_url'}")
    parsed = urlparse(normalized)
    if parsed.scheme.lower() != "https":
        raise ValueError("URL LinkedIn rechazada: scheme_not_allowed")
    host = (parsed.hostname or "").lower()
    if host not in _ALLOWED_HOSTS:
        raise ValueError("URL LinkedIn rechazada: host_not_allowed")
    if not _JOB_PATH_RE.match(parsed.path or "/"):
        raise ValueError("URL LinkedIn rechazada: path_not_allowed")
    if parsed.username or parsed.password:
        raise ValueError("URL LinkedIn rechazada: credentials_not_allowed")
    if (parsed.path or "").rstrip("/") == "/jobs/search":
        query_keys = {
            key
            for key, _value in parse_qsl(parsed.query, keep_blank_values=True)
        }
        if query_keys - _ALLOWED_SEARCH_QUERY_KEYS:
            raise ValueError("URL LinkedIn rechazada: query_parameter_not_allowed")
    return normalized


def canonicalize_linkedin_job_url(url: str) -> str:
    normalized = validate_linkedin_jobs_url(url)
    parsed = urlparse(normalized)
    path = parsed.path.rstrip("/") or "/jobs"
    if path.startswith("/jobs/view/"):
        query = ""
    else:
        query_pairs = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key.lower() not in _TRACKING_QUERY_KEYS
        ]
        query = urlencode(query_pairs)
    return urlunparse(("https", "www.linkedin.com", path, "", query, ""))


def linkedin_job_id_from_url(url: str) -> str:
    try:
        path = urlparse(canonicalize_linkedin_job_url(url)).path
    except ValueError:
        return ""
    match = re.search(r"/jobs/view/(?:[^/?#]*-)?(\d+)$", path)
    return match.group(1) if match else ""


def is_linkedin_auth_checkpoint(url: str) -> bool:
    parsed = urlparse((url or "").strip())
    host = (parsed.hostname or "").lower()
    if host not in _ALLOWED_HOSTS:
        return False
    path = (parsed.path or "").lower()
    return any(token in path for token in ("/login", "/checkpoint", "/challenge", "/uas/login"))


__all__ = [
    "canonicalize_linkedin_job_url",
    "is_linkedin_auth_checkpoint",
    "linkedin_job_id_from_url",
    "validate_linkedin_jobs_url",
]
