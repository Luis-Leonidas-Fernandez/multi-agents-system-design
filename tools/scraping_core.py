"""Helpers compartidos para scraping y fetch web."""
from __future__ import annotations

import os
from typing import Optional
from urllib.parse import urlparse


# Curated legacy allowlist preserved from the original scraping facade.
# It intentionally includes path-scoped entries for supported docs/code hubs
# so existing web fetch flows keep working without broadening host access.
_PREAPPROVED_HOSTS = {
    "platform.claude.com",
    "code.claude.com",
    "modelcontextprotocol.io",
    "github.com/anthropics",
    "agentskills.io",
    "docs.python.org",
    "en.cppreference.com",
    "docs.oracle.com",
    "learn.microsoft.com",
    "developer.mozilla.org",
    "go.dev",
    "pkg.go.dev",
    "www.php.net",
    "docs.swift.org",
    "kotlinlang.org",
    "ruby-doc.org",
    "doc.rust-lang.org",
    "www.typescriptlang.org",
    "react.dev",
    "angular.io",
    "vuejs.org",
    "nextjs.org",
    "expressjs.com",
    "nodejs.org",
    "bun.sh",
    "jquery.com",
    "getbootstrap.com",
    "tailwindcss.com",
    "d3js.org",
    "threejs.org",
    "redux.js.org",
    "webpack.js.org",
    "jestjs.io",
    "reactrouter.com",
    "docs.djangoproject.com",
    "flask.palletsprojects.com",
    "fastapi.tiangolo.com",
    "pandas.pydata.org",
    "numpy.org",
    "www.tensorflow.org",
    "pytorch.org",
    "scikit-learn.org",
    "matplotlib.org",
    "requests.readthedocs.io",
    "jupyter.org",
    "laravel.com",
    "symfony.com",
    "wordpress.org",
    "docs.spring.io",
    "hibernate.org",
    "tomcat.apache.org",
    "gradle.org",
    "maven.apache.org",
    "asp.net",
    "dotnet.microsoft.com",
    "nuget.org",
    "blazor.net",
    "reactnative.dev",
    "docs.flutter.dev",
    "developer.apple.com",
    "developer.android.com",
    "keras.io",
    "spark.apache.org",
    "huggingface.co",
    "www.kaggle.com",
    "www.mongodb.com",
    "redis.io",
    "www.postgresql.org",
    "dev.mysql.com",
    "www.sqlite.org",
    "graphql.org",
    "prisma.io",
    "docs.aws.amazon.com",
    "cloud.google.com",
    "kubernetes.io",
    "www.docker.com",
    "www.terraform.io",
    "www.ansible.com",
    "vercel.com/docs",
    "docs.netlify.com",
    "devcenter.heroku.com",
    "cypress.io",
    "selenium.dev",
    "docs.unity.com",
    "docs.unrealengine.com",
    "git-scm.com",
    "nginx.org",
    "httpd.apache.org",
}

_PREAPPROVED_HOSTNAME_ONLY: set[str] = set()
_PREAPPROVED_PATH_PREFIXES: dict[str, list[str]] = {}

for _entry in _PREAPPROVED_HOSTS:
    _slash = _entry.find("/")
    if _slash == -1:
        _PREAPPROVED_HOSTNAME_ONLY.add(_entry)
    else:
        _host = _entry[:_slash]
        _path = _entry[_slash:]
        _PREAPPROVED_PATH_PREFIXES.setdefault(_host, []).append(_path)


def _split_domain_list(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _domain_matches(domain: str, pattern: str) -> bool:
    domain = domain.lower().lstrip("www.")
    pattern = pattern.lower().lstrip("www.")
    return domain == pattern or domain.endswith("." + pattern)


def _domain_allowed(url: str, *, allowed: Optional[list[str]] = None, blocked: Optional[list[str]] = None) -> bool:
    host = urlparse(url).hostname or ""
    if not host:
        return False

    env_allowed = _split_domain_list(os.getenv("WEB_ALLOWED_DOMAINS"))
    env_blocked = _split_domain_list(os.getenv("WEB_BLOCKED_DOMAINS"))
    allowed = (allowed or []) + env_allowed
    blocked = (blocked or []) + env_blocked

    if blocked and any(_domain_matches(host, pattern) for pattern in blocked):
        return False
    if allowed:
        return any(_domain_matches(host, pattern) for pattern in allowed)
    return True


def _filter_search_hits_by_domains(hits: list[dict[str, str]], allowed_domains: Optional[list[str]], blocked_domains: Optional[list[str]]) -> list[dict[str, str]]:
    filtered_hits: list[dict[str, str]] = []
    for hit in hits:
        url = str(hit.get("url") or hit.get("link") or "").strip()
        if not url:
            continue
        if not _domain_allowed(url, allowed=allowed_domains, blocked=blocked_domains):
            continue
        filtered_hits.append(hit)
    return filtered_hits


def _is_preapproved_host(hostname: str, pathname: str) -> bool:
    host = hostname.lower().lstrip("www.")
    if host in _PREAPPROVED_HOSTNAME_ONLY:
        return True
    prefixes = _PREAPPROVED_PATH_PREFIXES.get(host, [])
    for prefix in prefixes:
        if pathname == prefix or pathname.startswith(prefix + "/"):
            return True
    return False


def _is_preapproved_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    return bool(host) and _is_preapproved_host(host, parsed.path or "/")


def _html_to_markdownish_text(url: str, title: str, text: str, links: list[dict[str, str]]) -> str:
    parts = [f"# {title or url}", ""]
    if text.strip():
        parts.append(text.strip())
        parts.append("")
    if links:
        parts.append("## Links")
        for link in links[:12]:
            link_text = (link.get("text") or link.get("href") or "link").strip()
            href = (link.get("href") or "").strip()
            if href:
                parts.append(f"- [{link_text}]({href})")
        parts.append("")
    parts.append(f"Source URL: {url}")
    return "\n".join(parts).strip()


def _build_web_fetch_prompt(markdown_content: str, prompt: str, is_preapproved_domain: bool) -> str:
    guidelines = (
        "Provide a moderately detailed response based on the content above. Include relevant details, code examples, and documentation excerpts as needed."
        if is_preapproved_domain
        else "Provide a moderately detailed response based only on the content above. In your response:\n"
        " - Enforce a strict 125-character maximum for quotes from any source document. Open Source Software is ok as long as we respect the license.\n"
        " - Use quotation marks for exact language from articles; any language outside of the quotation should never be word-for-word the same.\n"
        " - You are not a lawyer and never comment on the legality of your own prompts and responses.\n"
        " - Never produce or reproduce exact song lyrics."
    )

    return f"""
Web page content:
---
{markdown_content}
---

{prompt}

{guidelines}
""".strip()


def _build_redirect_message(original_url: str, redirect_url: str, status_code: int, prompt: str) -> str:
    status_text = {301: "Moved Permanently", 307: "Temporary Redirect", 308: "Permanent Redirect"}.get(status_code, "Found")
    return (
        "REDIRECT DETECTED: The URL redirects to a different host.\n\n"
        f"Original URL: {original_url}\n"
        f"Redirect URL: {redirect_url}\n"
        f"Status: {status_code} {status_text}\n\n"
        "To complete your request, I need to fetch content from the redirected URL. "
        "Please use WebFetch again with these parameters:\n"
        f'- url: "{redirect_url}"\n'
        f'- prompt: "{prompt}"'
    )


__all__ = [
    "_split_domain_list",
    "_domain_matches",
    "_domain_allowed",
    "_filter_search_hits_by_domains",
    "_is_preapproved_host",
    "_is_preapproved_url",
    "_html_to_markdownish_text",
    "_build_web_fetch_prompt",
    "_build_redirect_message",
]
