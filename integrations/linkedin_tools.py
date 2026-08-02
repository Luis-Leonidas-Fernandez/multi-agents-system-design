"""Tools LangChain del vertical LinkedIn Jobs read-only."""
from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import tool

from features.web_scraping.application.linkedin_service import run_linkedin_jobs_vertical


def prepare_linkedin_jobs_payload(
    query: str,
    location: str = "",
    max_results: Optional[int] = None,
) -> dict[str, object]:
    result = run_linkedin_jobs_vertical(
        query,
        location=location,
        max_results=max_results,
    )
    return result.model_dump(mode="json")


@tool
def scrape_linkedin_jobs_authenticated(
    query: str,
    location: str = "",
    max_results: Optional[int] = None,
) -> str:
    """Busca vacantes LinkedIn read-only de las últimas 24 horas con sesión manual."""
    try:
        payload = prepare_linkedin_jobs_payload(query, location, max_results)
    except Exception as exc:
        return f"No pude ejecutar LinkedIn Jobs de forma segura: {type(exc).__name__}."
    safe_payload = {
        "status": payload.get("status"),
        "job_uid": payload.get("job_uid"),
        "records": payload.get("records"),
        "warnings": payload.get("warnings"),
        "user_summary": payload.get("user_summary"),
    }
    return json.dumps(safe_payload, ensure_ascii=False, indent=2)


__all__ = ["prepare_linkedin_jobs_payload", "scrape_linkedin_jobs_authenticated"]
