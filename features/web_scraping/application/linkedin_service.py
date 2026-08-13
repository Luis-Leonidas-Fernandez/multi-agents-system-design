"""Caso de uso del vertical LinkedIn Jobs autenticado y read-only."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pydantic import ValidationError

from features.web_scraping.application.linkedin_audit import (
    current_linkedin_audit_dir,
    new_linkedin_job_uid,
    persist_linkedin_audit_snapshot,
)
from features.web_scraping.application.linkedin_intent import (
    extract_linkedin_locations,
)
from features.web_scraping.domain.linkedin_models import LinkedInJobsRequest, LinkedInJobsResult
from features.web_scraping.infrastructure.linkedin_detail_diagnostics import (
    detail_diagnostics_context,
)
from features.web_scraping.infrastructure.linkedin_query_navigation import (
    search_hydration_diagnostics_context,
)
from features.web_scraping.infrastructure.linkedin_static_probe_diagnostics import (
    static_probe_diagnostics_context,
)
from features.web_scraping.infrastructure.linkedin_visual_diagnostics import (
    visual_diagnostics_context,
)
from features.web_scraping.infrastructure.authenticated_browser import (
    BrowserProfileInUseError,
    close_reusable_authenticated_contexts,
)
from features.web_scraping.infrastructure.linkedin_jobs_pipeline import (
    scrape_linkedin_jobs,
)
from features.web_scraping.infrastructure.linkedin_scraper import (
    LinkedInAuthRequiredError,
    LinkedInBlockedError,
    configured_linkedin_max_results,
)
from features.web_scraping.infrastructure.linkedin_session_store import (
    LinkedInSessionStore,
)


ROOT = Path(__file__).resolve().parents[3]
_LINKEDIN_DEV_AUTO_BOOTSTRAP_ENV = "LINKEDIN_AUTO_BOOTSTRAP_ON_AUTH_FAILURE"
_LINKEDIN_BOOTSTRAP_CMD = [
    sys.executable,
    "scripts/bootstrap_linkedin_session.py",
    "--browser",
    "brave",
    "--executable-path",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "--observe-ready",
    "--ready-timeout-seconds",
    "300",
]


def _run_linkedin_bootstrap_for_retry(warnings: list[str]) -> bool:
    if os.getenv(_LINKEDIN_DEV_AUTO_BOOTSTRAP_ENV) != "1":
        return False

    warnings.append("auth_refresh_bootstrap_started")
    try:
        store = LinkedInSessionStore()
        metadata = store.load_browser_metadata()
        if metadata is not None:
            profile_path = store.resolve_profile_path(
                persisted_profile_path=metadata.get("profile_path"),
                create=False,
            )
            if close_reusable_authenticated_contexts(profile_path=profile_path):
                warnings.append("auth_refresh_runtime_session_closed")
        completed = subprocess.run(_LINKEDIN_BOOTSTRAP_CMD, cwd=ROOT, check=False)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        warnings.append(f"auth_refresh_bootstrap_failed:{type(exc).__name__}")
        return False

    if completed.returncode != 0:
        warnings.append(f"auth_refresh_bootstrap_failed:exit_{completed.returncode}")
        return False

    warnings.append("auth_refresh_bootstrap_completed")
    return True


def _split_location_fallback(value: str) -> tuple[str, ...]:
    locations = [
        item.strip()
        for item in re.split(r"[,;\n]+", value or "")
        if item.strip()
    ]
    return tuple(dict.fromkeys(locations))


def _record_country_group(record) -> str:
    location_hint = record.location or ""
    visible_location = location_hint.casefold()
    apac_or_remote = any(
        token in visible_location
        for token in (
            "asia-pacific",
            "asia pacific",
            "asia-pacífico",
            "asia pacífico",
            "apac",
            "remote",
            "remoto",
        )
    )
    visible_has_korea = any(
        token in visible_location
        for token in ("south korea", "corea del sur", "korea", "seoul", "서울", "대한민국")
    )
    visible_has_japan = any(
        token in visible_location
        for token in ("japan", "japón", "tokyo", "日本", "東京")
    )
    if apac_or_remote and not visible_has_korea:
        return "Asia-Pacífico / Remote"
    if visible_has_korea:
        return "Corea del Sur"
    if visible_has_japan:
        return "Japón"
    try:
        source_location = parse_qs(urlparse(record.source_url).query).get(
            "location",
            [""],
        )[0]
    except Exception:
        source_location = ""
    normalized = source_location.casefold()
    if any(token in normalized for token in ("south korea", "korea", "seoul", "서울", "대한민국")):
        return "Corea del Sur"
    if any(token in normalized for token in ("japan", "tokyo", "日本", "東京")):
        return "Japón"
    return "Otras ubicaciones"


def _status_label(value: str, *, kind: str) -> str:
    labels = {
        "yes": "sí",
        "no": "no",
        "unknown": "no informado",
        "ambiguous": "ambiguo",
        "sponsorship": "sponsorship",
        "no_sponsorship": "sin sponsorship",
    }
    return labels.get(value, value or f"{kind} no informado")


def _markdown_blockquote(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return "> no informado"
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def _detailed_record_block(record) -> str:
    title = (record.title or "Sin título").strip()
    company = (record.company_name or "No informada").strip()
    location = (record.location or "No informada").strip()
    workplace = (record.workplace_type or "no informada").strip()
    posted = (record.posted_at_text or "No informada").strip()
    languages = ", ".join(record.language_requirements) or "no informado"
    experience = "; ".join(record.experience_requirements) or "no informado"
    skills_list = [*record.hard_skills, *record.soft_skills]
    skills = ", ".join(skills_list) if skills_list else "no informadas"
    expectations = "; ".join(record.candidate_expectations) or "no informado"
    responsibilities = "; ".join(record.responsibilities) or "no informado"
    body = record.description_full_text or record.description_excerpt or "no informado"

    foreigner = _status_label(record.foreigner_acceptance, kind="extranjeros")
    visa = _status_label(record.visa_status, kind="visa")
    relocation = _status_label(record.relocation_support, kind="relocation")
    url = record.canonical_url or record.source_url or ""

    return (
        f"#### 📌 {title}\n"
        f"##### Metadata\n"
        f"- **Empresa:** {company}\n"
        f"- **Ubicación:** {location}\n"
        f"- **Modalidad:** {workplace}\n"
        f"- **Fecha:** {posted}\n"
        f"- **Skills:** {skills}\n"
        f"- **Idioma:** {languages}\n"
        f"- **Experiencia:** {experience}\n"
        f"- **Expectativas:** {expectations}\n"
        f"- **Responsabilidades:** {responsibilities}\n"
        f"- **Extranjeros / Visa:** Extranjeros: {foreigner} | Visa: {visa} | Relocation: {relocation}\n"
        f"- **Enlace:** {url}\n\n"
        f"##### Body completo\n"
        f"{_markdown_blockquote(body)}\n\n"
    )


def _render_grouped_records(result: LinkedInJobsResult) -> str:
    groups: dict[str, list] = {
        "Corea del Sur": [],
        "Japón": [],
        "Asia-Pacífico / Remote": [],
        "Otras ubicaciones": [],
    }
    for record in result.records:
        groups[_record_country_group(record)].append(record)

    lines = [
        f"### {len(result.records)} vacantes verificadas de las últimas 24 horas:\n",
    ]
    nonempty_groups = [
        (group_name, group_records)
        for group_name, group_records in groups.items()
        if group_records
    ]

    for group_name, group_records in nonempty_groups:
        flag = (
            "🇰🇷"
            if "Corea" in group_name
            else "🇯🇵"
            if "Japón" in group_name
            else "🌏"
            if "Asia-Pacífico" in group_name
            else "📍"
        )
        lines.append(f"### {flag} {group_name}\n")
        for record in group_records:
            lines.append(_detailed_record_block(record))

    return "\n".join(lines)


def _render_user_summary(result: LinkedInJobsResult) -> str:
    if result.status == "validation_error":
        return (
            "La búsqueda de LinkedIn no se ejecutó porque los parámetros son inválidos. "
            "`max_results` debe ser un número entre 1 y 50."
        )
    if result.status == "auth_required":
        suffix = (
            " Abrí automáticamente el bootstrap de LinkedIn en modo dev; completá "
            "el login manual en la ventana de Brave. El observador continuará cuando "
            "LinkedIn Jobs esté listo."
            if "auth_refresh_bootstrap_started" in result.warnings
            else ""
        )
        return (
            "LinkedIn no pudo validar o abrir el perfil persistente dedicado. El perfil "
            "y el snapshot storage_state se conservaron; no se expusieron cookies. "
            "Cerrá cualquier bootstrap/búsqueda LinkedIn todavía abierta y reintentá. "
            "Si persiste, ejecutá `python scripts/bootstrap_linkedin_session.py` y "
            "completá el login manual en el mismo perfil dedicado."
            f"{suffix}"
        )
    if result.status == "blocked":
        suffix = (
            " Abrí automáticamente el bootstrap de LinkedIn en modo dev; completá "
            "la verificación manual en la ventana de Brave. El observador continuará "
            "cuando LinkedIn Jobs esté listo."
            if "auth_refresh_bootstrap_started" in result.warnings
            else ""
        )
        return (
            "LinkedIn solicitó una verificación, checkpoint o redirigió fuera del área "
            "permitida. El scraper read-only se detuvo sin intentar evadir el control."
            f"{suffix}"
        )
    if result.status == "error":
        return (
            "No pude completar la búsqueda autenticada de LinkedIn. "
            "El error quedó registrado sin exponer cookies ni datos de sesión."
        )
    if result.status == "extraction_incomplete":
        if any(
            warning.startswith("extraction_incomplete:linkedin_hydration_unusable")
            for warning in result.warnings
        ):
            return (
                "LinkedIn no hidrató la lista ni entregó detalles verificables aun "
                "después de los probes read-only autenticados. La extracción quedó "
                "indeterminada; revisá el audit seguro del job."
            )
        if any(
            warning.startswith("extraction_incomplete:query_rate_limited")
            for warning in result.warnings
        ):
            return (
                "LinkedIn limitó temporalmente las consultas antes de cargar resultados. "
                "La extracción quedó incompleta y no se afirmó que hubiera cero vacantes."
            )
        if any(
            warning.startswith("extraction_incomplete:query_access_rejected")
            for warning in result.warnings
        ):
            return (
                "LinkedIn rechazó el acceso a la búsqueda antes de cargar resultados. "
                "La extracción quedó incompleta; revisá el audit seguro y la sesión."
            )
        if any(
            warning.startswith("extraction_incomplete:query_upstream_failure")
            for warning in result.warnings
        ):
            return (
                "LinkedIn devolvió una falla temporal de su servicio antes de cargar "
                "resultados. No se pudo determinar si existen vacantes."
            )
        if any(
            warning.startswith("extraction_incomplete:query_navigation_failure")
            for warning in result.warnings
        ):
            return (
                "LinkedIn no pudo completar la navegación de búsqueda antes de cargar "
                "resultados. No se pudo determinar si existen vacantes."
            )
        if any(
            warning.startswith("extraction_incomplete:query_hydration")
            for warning in result.warnings
        ):
            return (
                "LinkedIn cargó la búsqueda, pero no mostró cards ni un estado explícito "
                "de cero resultados antes del timeout de hidratación. La extracción quedó "
                "indeterminada; el audit no la registra como cero vacantes."
            )
        if any(
            warning.startswith("extraction_incomplete:detail_network")
            for warning in result.warnings
        ):
            return (
                "LinkedIn devolvió candidatos relevantes, pero la navegación de red falló "
                "al verificar sus detalles y fechas. El presupuesto seguro NO se agotó: "
                "la extracción quedó incompleta por una falla de red registrada en el audit."
            )
        if any(
            warning.startswith("extraction_incomplete:detail_fetch")
            for warning in result.warnings
        ):
            return (
                "LinkedIn devolvió candidatos relevantes, pero no pude completar la "
                "verificación técnica de sus detalles. El audit registra la categoría "
                "segura del fallo; no se afirmó que hubiera cero vacantes."
            )
        if any(
            warning.startswith("extraction_incomplete:posted_date")
            for warning in result.warnings
        ):
            return (
                "LinkedIn devolvió candidatos, pero no pude verificar su fecha de publicación "
                "dentro del presupuesto seguro de detalle. No puedo afirmar que haya cero "
                "vacantes recientes. El audit distingue fechas ausentes, presupuesto agotado "
                "y publicaciones fuera de las últimas 24 horas."
            )
        return (
            "La extracción de LinkedIn quedó incompleta: la página autenticada no produjo "
            "candidatos parseables. No puedo afirmar que haya cero vacantes. Revisá el audit "
            "del job para ver conteos de selectores, enlaces detectados y razones de descarte."
        )
    if not result.records:
        return (
            "No encontré vacantes verificables de las últimas 24 horas para AI Engineering, "
            "Data Science, Machine Learning, Deep Learning o productos con IA."
        )

    return _render_grouped_records(result)


def _classify_linkedin_scrape_result(records, rejected, timings, warnings: list[str]) -> str:
    status = "ok"
    parseable_candidates = sum(
        max(
            timing.discovered_count,
            timing.diagnostics.parseable_candidate_count,
        )
        for timing in timings
    )
    if not records and parseable_candidates == 0:
        hydration_unusable = "linkedin_hydration_unusable" in warnings
        hydration_timed_out = any(
            warning.startswith("query_hydration_timeout:")
            for warning in warnings
        )
        explicit_empty_queries = {
            warning.split(":", 1)[1]
            for warning in warnings
            if warning.startswith("query_empty_results_explicit:")
        }
        successful_timings = [timing for timing in timings if not timing.error]
        all_successful_queries_explicitly_empty = bool(successful_timings) and all(
            timing.query in explicit_empty_queries
            for timing in successful_timings
        )
        has_query_failures = any(timing.error for timing in timings)
        failure_priority = (
            "query_rate_limited",
            "query_access_rejected",
            "query_upstream_failure",
            "query_navigation_failure",
        )
        query_failure_category = next(
            (
                category
                for category in failure_priority
                if any(
                    category in warning
                    for warning in warnings
                    if warning.startswith("query_probe_result:")
                )
                or any(timing.error.endswith(category) for timing in timings)
            ),
            "",
        )
        no_query_dom = bool(timings) and all(
            timing.diagnostics.href_count == 0
            and timing.diagnostics.candidate_count == 0
            and not timing.diagnostics.selector_counts
            for timing in timings
        )
        if hydration_unusable:
            status = "extraction_incomplete"
            warnings.append("extraction_incomplete:linkedin_hydration_unusable")
        elif query_failure_category:
            status = "extraction_incomplete"
            warnings.append(f"extraction_incomplete:{query_failure_category}")
        elif has_query_failures and no_query_dom:
            status = "extraction_incomplete"
            warnings.append("extraction_incomplete:query_navigation_failure")
        elif hydration_timed_out:
            status = "extraction_incomplete"
            warnings.append("extraction_incomplete:query_hydration_timeout")
        elif all_successful_queries_explicitly_empty and not has_query_failures:
            status = "ok"
        else:
            status = "extraction_incomplete"
            warnings.append("extraction_incomplete:no_parseable_candidates")
    elif not records:
        rejection_reasons = {item.reason for item in rejected}
        hydration_unusable = "linkedin_hydration_unusable" in warnings
        has_detail_network_failure = "detail_network_failure" in rejection_reasons
        has_detail_fetch_failure = "detail_fetch_failed" in rejection_reasons
        has_unverified_dates = bool(
            rejection_reasons & {"unverified_posted_date", "detail_budget_exhausted"}
        )
        has_verified_outside = "outside_24_hours" in rejection_reasons
        if hydration_unusable:
            status = "extraction_incomplete"
            warnings.append("extraction_incomplete:linkedin_hydration_unusable")
        elif has_detail_network_failure:
            status = "extraction_incomplete"
            warnings.append("extraction_incomplete:detail_network_failure")
        elif has_detail_fetch_failure:
            status = "extraction_incomplete"
            warnings.append("extraction_incomplete:detail_fetch_failure")
        elif has_unverified_dates and not has_verified_outside:
            status = "extraction_incomplete"
            warnings.append("extraction_incomplete:posted_date_unverified")
    return status


class _LinkedInScrapeExecutionError(Exception):
    """Carries bounded diagnostics captured before a scrape exception escaped."""

    def __init__(
        self,
        original: Exception,
        *,
        detail_diagnostics,
        search_hydration_diagnostics,
        static_probe_diagnostics,
        visual_diagnostics,
    ) -> None:
        super().__init__(str(original))
        self.original = original
        self.detail_diagnostics = list(detail_diagnostics)
        self.search_hydration_diagnostics = list(search_hydration_diagnostics)
        self.static_probe_diagnostics = list(static_probe_diagnostics)
        self.visual_diagnostics = list(visual_diagnostics)


def _execute_linkedin_scrape(request: LinkedInJobsRequest, *, job_uid: str = "local"):
    with (
        detail_diagnostics_context() as detail_diagnostics,
        search_hydration_diagnostics_context() as search_hydration_diagnostics,
        static_probe_diagnostics_context() as static_probe_diagnostics,
        visual_diagnostics_context(
            current_linkedin_audit_dir(),
            job_uid=job_uid,
            diagnostics_root=ROOT / "data" / "private" / "linkedin" / "diagnostics",
        ) as visual_diagnostics,
    ):
        try:
            records, rejected, timings, warnings, queries = scrape_linkedin_jobs(request)
        except Exception as exc:
            raise _LinkedInScrapeExecutionError(
                exc,
                detail_diagnostics=detail_diagnostics.events,
                search_hydration_diagnostics=search_hydration_diagnostics.events,
                static_probe_diagnostics=static_probe_diagnostics.events,
                visual_diagnostics=visual_diagnostics.events,
            ) from exc
    status = _classify_linkedin_scrape_result(records, rejected, timings, warnings)
    return (
        status,
        records,
        rejected,
        timings,
        warnings,
        queries,
        detail_diagnostics.events,
        search_hydration_diagnostics.events,
        static_probe_diagnostics.events,
        visual_diagnostics.events,
    )


def _apply_linkedin_execution_error(
    exc: _LinkedInScrapeExecutionError,
    *,
    detail_diagnostics: list,
    search_hydration_diagnostics: list,
    static_probe_diagnostics: list,
    visual_diagnostics: list,
    warnings: list[str],
) -> str:
    detail_diagnostics.extend(exc.detail_diagnostics)
    search_hydration_diagnostics.extend(exc.search_hydration_diagnostics)
    static_probe_diagnostics.extend(exc.static_probe_diagnostics)
    visual_diagnostics.extend(exc.visual_diagnostics)
    original = exc.original
    if isinstance(
        original,
        (
            FileNotFoundError,
            ValueError,
            BrowserProfileInUseError,
            LinkedInAuthRequiredError,
        ),
    ):
        warnings.append(f"auth_required:{type(original).__name__}")
        return "auth_required"
    if isinstance(original, LinkedInBlockedError):
        warnings.append(f"blocked:{type(original).__name__}")
        return "blocked"
    warnings.append(f"runtime_error:{type(original).__name__}")
    return "error"


def run_linkedin_jobs_vertical(
    original_query: str,
    *,
    location: str = "",
    max_results: int | None = None,
) -> LinkedInJobsResult:
    status = "ok"
    records = []
    rejected = []
    timings = []
    warnings: list[str] = []
    queries: list[str] = []
    detail_diagnostics = []
    search_hydration_diagnostics = []
    static_probe_diagnostics = []
    visual_diagnostics = []
    job_uid = new_linkedin_job_uid()
    try:
        prompt_locations = extract_linkedin_locations(original_query)
        fallback_locations = _split_location_fallback(
            location or (os.getenv("LINKEDIN_LOCATION") or "")
        )
        resolved_locations = prompt_locations or fallback_locations
        request = LinkedInJobsRequest(
            query=original_query,
            location=resolved_locations[0] if resolved_locations else "",
            locations=list(resolved_locations),
            max_results=(
                max_results
                if max_results is not None
                else configured_linkedin_max_results()
            ),
        )
    except (ValidationError, ValueError) as exc:
        status = "validation_error"
        warnings.append(f"validation_error:{type(exc).__name__}")
    else:
        try:
            (
                status,
                records,
                rejected,
                timings,
                warnings,
                queries,
                detail_diagnostics,
                search_hydration_diagnostics,
                static_probe_diagnostics,
                visual_diagnostics,
            ) = _execute_linkedin_scrape(request, job_uid=job_uid)
        except _LinkedInScrapeExecutionError as exc:
            status = _apply_linkedin_execution_error(
                exc,
                detail_diagnostics=detail_diagnostics,
                search_hydration_diagnostics=search_hydration_diagnostics,
                static_probe_diagnostics=static_probe_diagnostics,
                visual_diagnostics=visual_diagnostics,
                warnings=warnings,
            )
        except (
            FileNotFoundError,
            ValueError,
            BrowserProfileInUseError,
            LinkedInAuthRequiredError,
        ) as exc:
            status = "auth_required"
            warnings.append(f"auth_required:{type(exc).__name__}")
        except LinkedInBlockedError as exc:
            status = "blocked"
            warnings.append(f"blocked:{type(exc).__name__}")
        except Exception as exc:
            status = "error"
            warnings.append(f"runtime_error:{type(exc).__name__}")

        if status in {"auth_required", "blocked"} and _run_linkedin_bootstrap_for_retry(warnings):
            retry_warnings = ["auth_refresh_retry_started"]
            try:
                (
                    status,
                    records,
                    rejected,
                    timings,
                    scrape_warnings,
                    queries,
                    retry_detail_diagnostics,
                    retry_search_hydration_diagnostics,
                    retry_static_probe_diagnostics,
                    retry_visual_diagnostics,
                ) = _execute_linkedin_scrape(request, job_uid=job_uid)
                detail_diagnostics.extend(retry_detail_diagnostics)
                search_hydration_diagnostics.extend(
                    retry_search_hydration_diagnostics
                )
                static_probe_diagnostics.extend(retry_static_probe_diagnostics)
                visual_diagnostics.extend(retry_visual_diagnostics)
                retry_warnings.extend(scrape_warnings)
            except _LinkedInScrapeExecutionError as exc:
                status = _apply_linkedin_execution_error(
                    exc,
                    detail_diagnostics=detail_diagnostics,
                    search_hydration_diagnostics=search_hydration_diagnostics,
                    static_probe_diagnostics=static_probe_diagnostics,
                    visual_diagnostics=visual_diagnostics,
                    warnings=retry_warnings,
                )
            except (
                FileNotFoundError,
                ValueError,
                BrowserProfileInUseError,
                LinkedInAuthRequiredError,
            ) as exc:
                status = "auth_required"
                retry_warnings.append(f"auth_required:{type(exc).__name__}")
            except LinkedInBlockedError as exc:
                status = "blocked"
                retry_warnings.append(f"blocked:{type(exc).__name__}")
            except Exception as exc:
                status = "error"
                retry_warnings.append(f"runtime_error:{type(exc).__name__}")
            warnings.extend(retry_warnings)

    paths = persist_linkedin_audit_snapshot(
        original_query=original_query.strip() or "(empty query)",
        queries=queries,
        timings=timings,
        vacancies=records,
        rejected=rejected,
        warnings=warnings,
        detail_diagnostics=detail_diagnostics,
        search_hydration_diagnostics=search_hydration_diagnostics,
        static_probe_diagnostics=static_probe_diagnostics,
        visual_diagnostics=visual_diagnostics,
        job_uid=job_uid,
    )
    result = LinkedInJobsResult(
        status=status,
        job_uid=paths.job_uid,
        records=records,
        rejected=rejected,
        warnings=warnings,
        queries=queries,
        timings=timings,
        audit_json_path=str(paths.json_path),
        audit_schema_path=str(paths.schema_path),
        audit_summary_path=str(paths.summary_path),
    )
    return result.model_copy(update={"user_summary": _render_user_summary(result)})


__all__ = ["run_linkedin_jobs_vertical"]
