"""Construcción de árbol renderizable para auditorías Moodle."""
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from features.web_scraping.domain.moodle_audit_models import (
    MoodleAuditAttachment,
    MoodleAuditExternalResource,
    MoodleAuditImage,
    MoodleAuditLink,
    MoodleAuditPage,
    MoodleAuditSnapshot,
    MoodleAuditSubmissionFile,
    MoodleAuditSubmissionState,
    MoodleAuditVideo,
)


MoodleAuditTreeNode = dict[str, Any]
MoodleAuditTreePayload = dict[str, Any]

_NOISE_LINK_TITLES = {
    "área personal",
    "area personal",
    "perfil",
    "calificaciones",
    "mensajes",
    "preferencias",
    "cerrar sesión",
    "cerrar sesion",
    "nuevo mensaje",
    "ver todo",
    "english ‎(en)‎",
    "español - internacional ‎(es)‎",
    "espanol - internacional ‎(es)‎",
    "página principal",
    "pagina principal",
    "participantes",
    "insignias",
    "competencias",
    "calendario",
    "archivos privados",
}

_NOISE_LINK_PATH_PREFIXES = (
    "/my/",
    "/user/profile.php",
    "/grade/report/",
    "/message/",
    "/user/preferences.php",
    "/login/logout.php",
    "/badges/view.php",
    "/admin/tool/lp/",
    "/calendar/view.php",
    "/user/files.php",
    "/user/index.php",
    "/user/view.php",
)

_NOISE_QUERY_TOKENS = (
    "lang=",
    "sesskey=",
    "contactsfirst=",
    "notificationpreferences",
)

_NOISE_FORUM_PATH_TOKENS = (
    "/mod/forum/subscribe.php",
    "/mod/forum/post.php",
)

_NOISE_PROVIDER_URL_TOKENS = (
    "about:blank",
    "drivesharing/clientmodel",
    "contacts.google.com/widget/hovercard",
    "clients6.google.com/static/proxy.html",
    "youtube.googleapis.com/embed/",
)


def _normalized_url(value: str) -> str:
    return (value or "").strip()


def _node_url_key(url: str) -> str:
    normalized = _normalized_url(url)
    if not normalized:
        return ""
    parsed = urlparse(normalized)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    path = parsed.path or ""
    query = parsed.query or ""
    suffix = f"?{query}" if query else ""
    return f"{host}{path}{suffix}" if host or path else normalized


def _host(url: str) -> str:
    try:
        return (urlparse(_normalized_url(url)).hostname or "").lower().removeprefix("www.")
    except Exception:
        return ""


def _query_param(url: str, key: str) -> str:
    try:
        from urllib.parse import parse_qs

        parsed = urlparse(_normalized_url(url))
        return (parse_qs(parsed.query).get(key) or [""])[0].strip()
    except Exception:
        return ""


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _pick_title(*values: str) -> str:
    for value in values:
        cleaned = _stringify(value)
        if cleaned:
            return cleaned
    return "Sin título"


def _trim_description(value: str, *, limit: int = 280) -> str:
    text = _stringify(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _trim_text(value: str, *, limit: int) -> str:
    text = _stringify(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _looks_like_intro_prompt(text: str) -> bool:
    lowered = _stringify(text).lower()
    signals = (
        "para comenzar, los invito a presentarse",
        "indiquen su nombre",
        "qué esperan aprender",
        "que esperan aprender",
        "experiencia previa tienen",
    )
    return any(signal in lowered for signal in signals)


def _looks_like_navigation_dump(text: str) -> bool:
    lowered = _stringify(text).lower()
    signals = (
        "salta al contenido principal",
        "área personal",
        "area personal",
        "cerrar sesión",
        "cerrar sesion",
        "nuevo mensaje",
        "no hay mensaje",
        "preferencias",
    )
    return sum(1 for signal in signals if signal in lowered) >= 2


def _looks_like_student_presentation(text: str) -> bool:
    lowered = _stringify(text).lower()
    signals = (
        "mi nombre es",
        "me gusta mucho esta carrera",
        "qué esperan aprender",
        "que esperan aprender",
        "experiencia previa",
        "actualmente curso la carrera",
        "para comenzar, los invito a presentarse",
    )
    return any(signal in lowered for signal in signals)


def _redirect_target(url: str, final_url: str, redirect_target: str = "") -> str:
    candidates = [_stringify(redirect_target), _stringify(final_url)]
    origin = _stringify(url)
    for candidate in candidates:
        if candidate and candidate != origin:
            return candidate
    return ""


def _resource_kind(resource_type: str, fallback_kind: str = "unknown") -> str:
    normalized = _stringify(resource_type).lower()
    if normalized in {
        "course",
        "page",
        "course_section",
        "section",
        "forum",
        "quiz",
        "assignment",
        "document",
        "image",
        "video",
        "preview",
        "link",
        "google_slides",
        "google_drive",
        "external_redirect",
        "unknown",
    }:
        return normalized
    if normalized in {"folder", "resource", "file"}:
        return "document"
    if normalized in {"url", "redirect_link", "external_tool"}:
        return "link"
    return fallback_kind


def _resource_identity_key(node: MoodleAuditTreeNode) -> str:
    canonical = _stringify(node.get("canonicalUrl"))
    url = _stringify(node.get("url"))
    if canonical:
        return _node_url_key(canonical)
    if url:
        return _node_url_key(url)
    return ""


def _resource_identity_keys(node: MoodleAuditTreeNode) -> list[str]:
    keys: list[str] = []
    for raw_value in (
        _stringify(node.get("canonicalUrl")),
        _stringify(node.get("url")),
        _stringify(node.get("redirectUrl")),
        _stringify(node.get("downloadUrl")),
    ):
        if not raw_value:
            continue
        key = _node_url_key(raw_value)
        if key and key not in keys:
            keys.append(key)
    return keys


def _is_course_section_identity(url: str, root_course_id: str) -> bool:
    if not url or not root_course_id:
        return False
    lowered = url.lower()
    if "/course/view.php" not in lowered:
        return False
    return _query_param(url, "id") == root_course_id and bool(_query_param(url, "section"))


def _page_kind(page: MoodleAuditPage) -> str:
    if page.page_kind == "course_home":
        return "section"
    if page.page_kind == "course_section":
        return "section"
    if page.resource_type:
        return _resource_kind(page.resource_type, "page")
    if page.page_kind in {"assignment_detail", "submission_page"}:
        return "assignment"
    return "page"


def _node_id(prefix: str, seed: str, fallback: str) -> str:
    value = _stringify(seed) or fallback
    safe = value.replace("https://", "").replace("http://", "").replace("/", "::")
    return f"{prefix}:{safe}"


def _compact_metadata(**kwargs: Any) -> dict[str, str | int | float | bool]:
    result: dict[str, str | int | float | bool] = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, bool):
            result[key] = value
            continue
        if isinstance(value, (int, float)):
            result[key] = value
            continue
        cleaned = _stringify(value)
        if cleaned:
            result[key] = cleaned
    return result


def _redirect_child(
    *,
    parent_id: str,
    title: str,
    origin_url: str,
    redirect_url: str,
) -> MoodleAuditTreeNode | None:
    origin = _stringify(origin_url)
    target = _stringify(redirect_url)
    if not origin or not target or origin == target:
        return None
    return {
        "id": _node_id("redirect", f"{parent_id}:{origin}:{target}", parent_id),
        "kind": "external_redirect",
        "title": title,
        "url": origin,
        "canonicalUrl": origin,
        "redirectUrl": target,
        "previewUrl": "",
        "downloadUrl": "",
        "mimeType": "",
        "subtitle": "",
        "description": f"{origin} → {target}",
        "badges": ["redirect"],
        "metadata": _compact_metadata(origin_host=_host(origin), target_host=_host(target)),
        "children": [],
    }


def _link_node(link: MoodleAuditLink, parent_id: str, index: int) -> MoodleAuditTreeNode:
    final_url = _stringify(link.final_url or link.url)
    redirect_url = _redirect_target(link.url, final_url, link.redirect_target)
    node: MoodleAuditTreeNode = {
        "id": _node_id("link", final_url or link.url, f"{parent_id}:link:{index}"),
        "kind": _resource_kind(link.resource_type, "link"),
        "title": _pick_title(link.label, final_url or link.url, f"Link {index}"),
        "url": _stringify(link.url),
        "canonicalUrl": final_url,
        "previewUrl": "",
        "downloadUrl": "",
        "redirectUrl": redirect_url,
        "mimeType": "",
        "subtitle": "",
        "description": "",
        "badges": [badge for badge in ([link.resource_type] if link.resource_type else []) + (["submission"] if link.is_submission_target else []) if badge],
        "metadata": _compact_metadata(
            final_url=final_url if final_url and final_url != link.url else "",
            redirect_chain=len(link.redirect_chain),
            submission_target=link.is_submission_target,
        ),
        "children": [],
    }
    redirect_child = _redirect_child(
        parent_id=node["id"],
        title="Redirect detectado",
        origin_url=_stringify(link.url),
        redirect_url=redirect_url,
    )
    if redirect_child:
        node["children"].append(redirect_child)
    return node


def _attachment_kind(attachment: MoodleAuditAttachment) -> str:
    content_type = _stringify(attachment.content_type or attachment.mime_hint).lower()
    if content_type.startswith("image/"):
        return "image"
    if content_type.startswith("video/"):
        return "video"
    return "document"


def _attachment_node(attachment: MoodleAuditAttachment, parent_id: str, index: int) -> MoodleAuditTreeNode:
    final_url = _stringify(attachment.final_url or attachment.url)
    redirect_url = _redirect_target(attachment.url, final_url, attachment.redirect_target)
    kind = _attachment_kind(attachment)
    node: MoodleAuditTreeNode = {
        "id": _node_id("attachment", final_url or attachment.url, f"{parent_id}:attachment:{index}"),
        "kind": kind,
        "title": _pick_title(attachment.filename, attachment.label, final_url or attachment.url, f"Archivo {index}"),
        "url": _stringify(attachment.url),
        "canonicalUrl": final_url,
        "previewUrl": final_url if kind == "image" else "",
        "downloadUrl": final_url if attachment.is_download or kind == "document" else "",
        "redirectUrl": redirect_url,
        "mimeType": _stringify(attachment.content_type or attachment.mime_hint),
        "subtitle": "",
        "description": "",
        "badges": [badge for badge in [attachment.kind, "download" if attachment.is_download else ""] if badge],
        "metadata": _compact_metadata(
            filename=attachment.filename,
            content_length=attachment.content_length,
            status_code=attachment.status_code,
            content_disposition=attachment.content_disposition,
        ),
        "children": [],
    }
    redirect_child = _redirect_child(
        parent_id=node["id"],
        title="Redirect detectado",
        origin_url=_stringify(attachment.url),
        redirect_url=redirect_url,
    )
    if redirect_child:
        node["children"].append(redirect_child)
    return node


def _video_node(video: MoodleAuditVideo, parent_id: str, index: int) -> MoodleAuditTreeNode:
    url = _stringify(video.watch_url or video.embed_url)
    return {
        "id": _node_id("video", url, f"{parent_id}:video:{index}"),
        "kind": "video",
        "title": _pick_title(video.label, video.watch_url, video.embed_url, f"Video {index}"),
        "url": url,
        "canonicalUrl": url,
        "previewUrl": _stringify(video.preview_url),
        "downloadUrl": "",
        "redirectUrl": "",
        "mimeType": "",
        "subtitle": _stringify(video.provider),
        "description": "",
        "badges": [badge for badge in [video.provider] if badge],
        "metadata": _compact_metadata(embed_url=video.embed_url),
        "children": [],
    }


def _image_node(image: MoodleAuditImage, parent_id: str, index: int) -> MoodleAuditTreeNode:
    url = _stringify(image.url)
    label = _stringify(image.label)
    title = _pick_title(label, image.url, f"Imagen {index}")
    if _looks_like_raw_url_title(title):
        title = "Preview"
    return {
        "id": _node_id("image", url, f"{parent_id}:image:{index}"),
        "kind": "image",
        "title": title,
        "url": url,
        "canonicalUrl": url,
        "previewUrl": url,
        "downloadUrl": "",
        "redirectUrl": "",
        "mimeType": "",
        "subtitle": "",
        "description": "",
        "badges": [badge for badge in [image.kind] if badge],
        "metadata": {},
        "children": [],
    }


def _external_resource_node(resource: MoodleAuditExternalResource, parent_id: str) -> MoodleAuditTreeNode:
    primary_url = _stringify(resource.canonical_url or resource.access_url or resource.preview_url or resource.download_url)
    description = ""
    if resource.content_blocks:
        description = _trim_description(" · ".join(block for block in resource.content_blocks if _stringify(block)))
    node: MoodleAuditTreeNode = {
        "id": _node_id("external", primary_url, f"{parent_id}:external"),
        "kind": _resource_kind(resource.provider or resource.resource_type, "link"),
        "title": _pick_title(resource.content_blocks[0] if resource.content_blocks else "", resource.provider or resource.resource_type, "Recurso externo"),
        "url": _stringify(resource.access_url or resource.canonical_url),
        "canonicalUrl": _stringify(resource.canonical_url),
        "previewUrl": _stringify(resource.preview_url or resource.htmlpresent_url),
        "downloadUrl": _stringify(resource.download_url),
        "redirectUrl": "",
        "mimeType": "",
        "subtitle": _stringify(resource.resource_type),
        "description": description,
        "badges": [badge for badge in [resource.provider, "login" if resource.requires_login else ""] if badge],
        "metadata": _compact_metadata(
            resource_id=resource.resource_id,
            htmlpresent_url=resource.htmlpresent_url,
            slide_count=resource.slide_count,
            requires_login=resource.requires_login,
        ),
        "children": [],
    }
    if resource.access_url and resource.canonical_url and resource.access_url != resource.canonical_url:
        redirect_child = _redirect_child(
            parent_id=node["id"],
            title="Acceso → destino canónico",
            origin_url=resource.access_url,
            redirect_url=resource.canonical_url,
        )
        if redirect_child:
            node["children"].append(redirect_child)
    return node


def _submitted_file_node(file: MoodleAuditSubmissionFile, parent_id: str, index: int) -> MoodleAuditTreeNode:
    final_url = _stringify(file.final_url or file.url)
    redirect_url = _redirect_target(file.url, final_url, file.redirect_target)
    node: MoodleAuditTreeNode = {
        "id": _node_id("submission-file", final_url or file.url, f"{parent_id}:submission-file:{index}"),
        "kind": "document",
        "title": _pick_title(file.filename, file.label, final_url or file.url, f"Entrega {index}"),
        "url": _stringify(file.url),
        "canonicalUrl": final_url,
        "previewUrl": "",
        "downloadUrl": final_url if file.is_download or final_url else "",
        "redirectUrl": redirect_url,
        "mimeType": _stringify(file.content_type or file.mime_hint),
        "subtitle": "Archivo entregado",
        "description": "",
        "badges": ["submitted"],
        "metadata": _compact_metadata(
            content_length=file.content_length,
            status_code=file.status_code,
            content_disposition=file.content_disposition,
        ),
        "children": [],
    }
    redirect_child = _redirect_child(
        parent_id=node["id"],
        title="Redirect detectado",
        origin_url=_stringify(file.url),
        redirect_url=redirect_url,
    )
    if redirect_child:
        node["children"].append(redirect_child)
    return node


def _submission_badges(state: MoodleAuditSubmissionState | None) -> list[str]:
    if state is None:
        return []
    badges: list[str] = []
    if state.can_submit:
        badges.append("can_submit")
    if state.is_submitted:
        badges.append("submitted")
    if state.is_graded:
        badges.append("graded")
    if state.is_locked:
        badges.append("locked")
    return badges


def _submission_metadata(state: MoodleAuditSubmissionState | None) -> dict[str, str | int | float | bool]:
    if state is None:
        return {}
    return _compact_metadata(
        submission_status=state.submission_status,
        grading_status=state.grading_status,
        due_date=state.due_date_text,
        time_remaining=state.time_remaining_text,
        last_modified=state.last_modified_text,
        attempt=state.attempt_text,
        instructions_count=len(state.instructions),
        available_actions=", ".join(item for item in state.available_actions if _stringify(item)),
    )


def _page_node(page: MoodleAuditPage, index: int) -> MoodleAuditTreeNode:
    canonical = _stringify(page.final_url or page.url)
    preferred_title = page.title
    if page.page_kind in {"course_section", "linked_resource"} and _stringify(page.source_link_label):
        normalized_title = _stringify(page.title).lower()
        if not normalized_title or normalized_title == "hardware y sistemas operativos":
            preferred_title = page.source_link_label
    node: MoodleAuditTreeNode = {
        "id": _node_id("page", page.dedupe_key or canonical or page.url, f"page:{index}"),
        "kind": _page_kind(page),
        "title": _pick_title(preferred_title, page.source_link_label, canonical or page.url, f"Página {index}"),
        "url": _stringify(page.url),
        "canonicalUrl": canonical,
        "previewUrl": "",
        "downloadUrl": "",
        "redirectUrl": _redirect_target(page.url, canonical),
        "mimeType": "",
        "subtitle": _stringify(page.subtitle),
        "description": _trim_description(page.description or page.text_excerpt),
        "badges": list(
            dict.fromkeys(
                [badge for badge in [page.resource_type, page.page_kind, *_submission_badges(page.submission_state)] if _stringify(badge)]
            )
        ),
        "metadata": _compact_metadata(
            page_kind=page.page_kind,
            visit_order=page.visit_order,
            crawl_depth=page.crawl_depth,
            confidence_score=page.confidence_score,
            source_link_label=page.source_link_label,
            breadcrumbs=" / ".join(item for item in page.breadcrumbs if _stringify(item)),
            html_snapshot_path=page.html_snapshot_path,
            extracted_items_count=page.extracted_items_count,
            **_submission_metadata(page.submission_state),
        ),
        "children": [],
    }
    redirect_child = _redirect_child(
        parent_id=node["id"],
        title="Redirect detectado",
        origin_url=_stringify(page.url),
        redirect_url=_redirect_target(page.url, canonical),
    )
    if redirect_child:
        node["children"].append(redirect_child)

    for idx, link in enumerate(page.links, start=1):
        node["children"].append(_link_node(link, node["id"], idx))
    for idx, attachment in enumerate(page.attachments, start=1):
        node["children"].append(_attachment_node(attachment, node["id"], idx))
    for idx, video in enumerate(page.videos, start=1):
        node["children"].append(_video_node(video, node["id"], idx))
    for idx, image in enumerate(page.images, start=1):
        node["children"].append(_image_node(image, node["id"], idx))
    if page.external_resource is not None:
        node["children"].append(_external_resource_node(page.external_resource, node["id"]))
    if page.submission_state is not None:
        for idx, submission_file in enumerate(page.submission_state.submitted_files, start=1):
            node["children"].append(_submitted_file_node(submission_file, node["id"], idx))
    return node


def _clone_tree_node(node: MoodleAuditTreeNode) -> MoodleAuditTreeNode:
    cloned = dict(node)
    cloned["badges"] = [*node.get("badges", [])]
    cloned["metadata"] = dict(node.get("metadata", {}) or {})
    cloned["children"] = [_clone_tree_node(child) for child in node.get("children", [])]
    return cloned


def _node_signature(node: MoodleAuditTreeNode) -> str:
    kind = _stringify(node.get("kind"))
    url = _stringify(node.get("canonicalUrl") or node.get("url"))
    redirect = _stringify(node.get("redirectUrl"))
    title = _stringify(node.get("title"))
    if kind == "external_redirect":
        return f"{kind}|{_node_url_key(_stringify(node.get('url')))}|{_node_url_key(redirect)}"
    if url:
        return f"{kind}|{_node_url_key(url)}"
    return f"{kind}|{title.lower()}"


def _node_score(node: MoodleAuditTreeNode) -> int:
    kind = _stringify(node.get("kind"))
    priority = {
        "assignment": 100,
        "quiz": 95,
        "google_slides": 92,
        "google_drive": 92,
        "document": 90,
        "forum": 85,
        "page": 80,
        "section": 75,
        "video": 70,
        "image": 20,
        "link": 10,
        "external_redirect": 5,
    }.get(kind, 0)
    metadata = node.get("metadata") or {}
    return (
        priority
        + len(node.get("children") or []) * 4
        + len(node.get("badges") or []) * 2
        + (8 if _stringify(node.get("description")) else 0)
        + (8 if _stringify(node.get("downloadUrl")) else 0)
        + (6 if _stringify(node.get("previewUrl")) else 0)
        + (4 if _stringify(node.get("mimeType")) else 0)
        + min(len(metadata), 8)
    )


def _merge_preferred_node(preferred: MoodleAuditTreeNode, candidate: MoodleAuditTreeNode) -> MoodleAuditTreeNode:
    for field in ("description", "subtitle", "previewUrl", "downloadUrl", "redirectUrl", "mimeType", "canonicalUrl", "url"):
        if not _stringify(preferred.get(field)) and _stringify(candidate.get(field)):
            preferred[field] = candidate[field]
    preferred_kind = _stringify(preferred.get("kind"))
    candidate_kind = _stringify(candidate.get("kind"))
    if preferred_kind in {"page", "section"} and candidate_kind in {"link", "course_section"} and _stringify(candidate.get("title")):
        preferred["title"] = candidate["title"]
    preferred["badges"] = list(dict.fromkeys([*preferred.get("badges", []), *candidate.get("badges", [])]))
    preferred_metadata = dict(preferred.get("metadata") or {})
    for key, value in dict(candidate.get("metadata") or {}).items():
        if key not in preferred_metadata or preferred_metadata[key] in {"", None, 0, False}:
            preferred_metadata[key] = value
    preferred["metadata"] = preferred_metadata
    preferred["children"] = [*preferred.get("children", []), *candidate.get("children", [])]
    return preferred


def _dedupe_children(children: list[MoodleAuditTreeNode]) -> list[MoodleAuditTreeNode]:
    deduped: list[MoodleAuditTreeNode] = []
    index_by_signature: dict[str, int] = {}
    for child in children:
        signature = _node_signature(child)
        current_index = index_by_signature.get(signature)
        if current_index is None:
            index_by_signature[signature] = len(deduped)
            deduped.append(child)
            continue
        existing = deduped[current_index]
        if _node_score(child) > _node_score(existing):
            preferred, candidate = child, existing
        else:
            preferred, candidate = existing, child
        deduped[current_index] = _merge_preferred_node(preferred, candidate)
    return deduped


def _semantic_duplicate_key(node: MoodleAuditTreeNode) -> str:
    kind = _stringify(node.get("kind"))
    if kind == "external_redirect":
        return ""
    identity = _resource_identity_key(node)
    if not identity:
        return ""
    return f"resource|{identity}"


def _semantic_dedupe_children(children: list[MoodleAuditTreeNode]) -> list[MoodleAuditTreeNode]:
    deduped: list[MoodleAuditTreeNode] = []
    index_by_key: dict[str, int] = {}
    for child in children:
        semantic_keys = [key for key in _resource_identity_keys(child) if key]
        if _stringify(child.get("kind")) == "external_redirect":
            semantic_keys = []
        if not semantic_keys:
            semantic_key = _semantic_duplicate_key(child)
            semantic_keys = [semantic_key] if semantic_key else []
        if not semantic_keys:
            deduped.append(child)
            continue
        current_index = next((index_by_key[key] for key in semantic_keys if key in index_by_key), None)
        if current_index is None:
            new_index = len(deduped)
            for semantic_key in semantic_keys:
                index_by_key[semantic_key] = new_index
            deduped.append(child)
            continue
        existing = deduped[current_index]
        if _node_score(child) > _node_score(existing):
            preferred, candidate = child, existing
        else:
            preferred, candidate = existing, child
        merged = _merge_preferred_node(preferred, candidate)
        deduped[current_index] = merged
        for semantic_key in _resource_identity_keys(merged):
            index_by_key[semantic_key] = current_index
    return deduped


def _looks_like_raw_url_title(title: str) -> bool:
    lowered = _stringify(title).lower()
    return lowered.startswith("http://") or lowered.startswith("https://")


def _is_noise_link(node: MoodleAuditTreeNode, *, root_host: str, course_name: str, root_course_id: str) -> bool:
    title = _stringify(node.get("title")).lower()
    url = _stringify(node.get("canonicalUrl") or node.get("url"))
    parsed = urlparse(url) if url else None
    path = (parsed.path or "") if parsed else ""
    host = (parsed.hostname or "").lower().removeprefix("www.") if parsed else ""
    if title in _NOISE_LINK_TITLES:
        return True
    if title == course_name.lower():
        return True
    if host == root_host and "/course/view.php" in path and root_course_id and _query_param(url, "id") and _query_param(url, "id") != root_course_id:
        return True
    if any(path.startswith(prefix) for prefix in _NOISE_LINK_PATH_PREFIXES):
        return True
    query = (parsed.query or "") if parsed else ""
    if any(token in query for token in _NOISE_QUERY_TOKENS):
        return True
    if any(token in (url or "").lower() for token in _NOISE_FORUM_PATH_TOKENS):
        return True
    if host == root_host and (_looks_like_raw_url_title(title) or title in {"(0362) 4570624"}):
        return True
    if host in {"facebook.com", "www.facebook.com"} and _looks_like_raw_url_title(title):
        return True
    if host == "www.uep165.edu.ar" and title in {"pagina principal", "página principal"}:
        return True
    if title in {"acceder", "suscribir", "suscrito", "darse de baja de este foro", "responder", "finalizar revisión", "finalizar revision"}:
        return True
    return False


def _is_noise_image(node: MoodleAuditTreeNode) -> bool:
    title = _stringify(node.get("title")).lower()
    url = _stringify(node.get("canonicalUrl") or node.get("url")).lower()
    preview = _stringify(node.get("previewUrl")).lower()
    if title in {"cargando", "klass"}:
        return True
    if "/theme/image.php/" in url or "/theme_klass/" in url:
        return True
    if "/user/icon/" in url or "/mod_label/intro/" in url:
        return True
    if "drive.google.com/file/" in url and (not preview or preview == url):
        return True
    return False


def _is_noise_video(node: MoodleAuditTreeNode) -> bool:
    title = _stringify(node.get("title")).lower()
    url = _stringify(node.get("canonicalUrl") or node.get("url")).lower()
    subtitle = _stringify(node.get("subtitle")).lower()
    if any(token in url for token in _NOISE_PROVIDER_URL_TOKENS):
        return True
    if title in {"about:blank", "tarjeta de información", "tarjeta de informacion"}:
        return True
    if subtitle == "iframe" and ("google" in url or "about:blank" in url):
        return True
    return False


def _is_noise_forum(node: MoodleAuditTreeNode) -> bool:
    title = _stringify(node.get("title")).lower()
    url = _stringify(node.get("canonicalUrl") or node.get("url")).lower()
    subtitle = _stringify(node.get("subtitle")).lower()
    description = _stringify(node.get("description"))
    breadcrumbs = _stringify((node.get("metadata") or {}).get("breadcrumbs"))
    if (
        "foro de presentación" in title
        or "foro de presentacion" in title
        or "foro de presentación" in subtitle
        or "foro de presentacion" in subtitle
        or "avisos y debates" in title
        or "avisos y debates" in subtitle
    ):
        return True
    if (
        "english" in title
        or "español - internacional" in title
        or "espanol - internacional" in title
        or title in {"suscribir", "suscrito", "darse de baja de este foro", "responder"}
    ):
        return True
    if any(token in url for token in _NOISE_FORUM_PATH_TOKENS):
        return True
    if "/user/view.php" in url:
        return True
    if _looks_like_raw_url_title(title):
        return True
    if (
        ("foro de presentación" in subtitle or "foro de presentacion" in subtitle or "presentación" in title or "presentacion" in title)
        and _looks_like_student_presentation(description)
    ):
        return True
    if ("foro de presentación" in breadcrumbs.lower() or "foro de presentacion" in breadcrumbs.lower()) and _looks_like_student_presentation(description):
        return True
    return False


def _is_noise_google_resource(node: MoodleAuditTreeNode) -> bool:
    title = _stringify(node.get("title")).lower()
    url = _stringify(node.get("canonicalUrl") or node.get("url")).lower()
    if any(token in url for token in _NOISE_PROVIDER_URL_TOKENS):
        return True
    if title in {"acceder", "vista html de la presentación", "vista html de la presentacion"}:
        return True
    if title.startswith("https://docs.google.com/presentation/?"):
        return True
    return False


def _is_noise_quiz(node: MoodleAuditTreeNode) -> bool:
    title = _stringify(node.get("title")).lower()
    url = _stringify(node.get("canonicalUrl") or node.get("url")).lower()
    if "lang=en" in url or "lang=es" in url:
        return True
    if title in {"english ‎(en)‎", "español - internacional ‎(es)‎", "espanol - internacional ‎(es)‎"}:
        return True
    if title in {"revisión", "revision"}:
        return True
    return False


def _is_noise_node(node: MoodleAuditTreeNode, *, root_host: str, course_name: str, root_course_id: str) -> bool:
    kind = _stringify(node.get("kind"))
    if kind in {"link", "course_section"}:
        return _is_noise_link(node, root_host=root_host, course_name=course_name, root_course_id=root_course_id)
    if kind == "image":
        return _is_noise_image(node)
    if kind == "video":
        return _is_noise_video(node)
    if kind == "forum":
        return _is_noise_forum(node)
    if kind == "quiz":
        return _is_noise_quiz(node)
    if kind in {"google_slides", "google_drive"}:
        return _is_noise_google_resource(node)
    return False


def _normalize_display_title(node: MoodleAuditTreeNode, *, course_name: str) -> MoodleAuditTreeNode:
    title = _stringify(node.get("title"))
    kind = _stringify(node.get("kind"))
    subtitle = _stringify(node.get("subtitle"))
    metadata = dict(node.get("metadata") or {})
    source_link_label = _stringify(metadata.get("source_link_label"))
    if title.lower() == course_name.lower():
        if kind in {"page", "section"} and source_link_label:
            node["title"] = source_link_label
        elif kind in {"quiz", "forum", "assignment"} and subtitle:
            node["title"] = subtitle
    return node


def _presentable_metadata(node: MoodleAuditTreeNode) -> dict[str, Any]:
    metadata = dict(node.get("metadata") or {})
    noisy_keys = {
        "source_link_label",
        "html_snapshot_path",
        "page_kind",
        "origin_host",
        "target_host",
        "breadcrumbs",
        "visit_order",
        "crawl_depth",
        "confidence_score",
        "extracted_items_count",
        "instructions_count",
        "redirect_chain",
        "content_disposition",
        "content_length",
        "status_code",
        "resource_id",
        "slide_count",
        "requires_login",
        "submission_target",
        "final_url",
    }
    for key in noisy_keys:
        metadata.pop(key, None)
    return metadata


def _promote_primary_asset(node: MoodleAuditTreeNode) -> MoodleAuditTreeNode:
    kind = _stringify(node.get("kind"))
    if kind not in {"document", "video", "image", "google_slides", "google_drive"}:
        return node
    children = list(node.get("children", []))
    asset_child = next(
        (
            child
            for child in children
            if _stringify(child.get("kind")) == kind and _node_url_key(_stringify(child.get("canonicalUrl") or child.get("url")))
            == _node_url_key(_stringify(node.get("canonicalUrl") or node.get("url")))
        ),
        None,
    )
    if asset_child is None:
        return node
    for field in ("previewUrl", "downloadUrl", "mimeType", "subtitle", "description"):
        if not _stringify(node.get(field)) and _stringify(asset_child.get(field)):
            node[field] = asset_child[field]
    node["badges"] = list(dict.fromkeys([*node.get("badges", []), *asset_child.get("badges", [])]))
    promoted_metadata = dict(node.get("metadata") or {})
    for key, value in dict(asset_child.get("metadata") or {}).items():
        if key not in promoted_metadata or promoted_metadata[key] in {"", None, 0, False}:
            promoted_metadata[key] = value
    node["metadata"] = promoted_metadata
    node["children"] = [child for child in children if child is not asset_child]
    return node


def _collapse_single_login_duplicate(node: MoodleAuditTreeNode) -> MoodleAuditTreeNode:
    kind = _stringify(node.get("kind"))
    if kind not in {"google_slides", "google_drive"}:
        return node
    children = list(node.get("children", []))
    canonical_key = _node_url_key(_stringify(node.get("canonicalUrl") or node.get("url")))
    duplicate_child = next(
        (
            child
            for child in children
            if _stringify(child.get("kind")) == kind
            and _node_url_key(_stringify(child.get("canonicalUrl") or child.get("url"))) == canonical_key
        ),
        None,
    )
    if duplicate_child is None:
        return node
    merged = _merge_preferred_node(node, duplicate_child)
    merged["children"] = [
        child
        for child in children
        if child is not duplicate_child
    ] + [
        child
        for child in duplicate_child.get("children", [])
    ]
    return merged


def _remove_redundant_self_redirects(node: MoodleAuditTreeNode) -> MoodleAuditTreeNode:
    canonical_key = _node_url_key(_stringify(node.get("canonicalUrl") or node.get("url")))
    redirect_key = _node_url_key(_stringify(node.get("redirectUrl")))
    if not canonical_key or not redirect_key:
        return node
    filtered_children: list[MoodleAuditTreeNode] = []
    seen_signatures: set[str] = set()
    for child in node.get("children", []):
        if _stringify(child.get("kind")) == "external_redirect":
            child_origin = _node_url_key(_stringify(child.get("url")))
            child_target = _node_url_key(_stringify(child.get("redirectUrl")))
            if child_origin == _node_url_key(_stringify(node.get("url"))) and child_target == redirect_key:
                signature = _node_signature(child)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
        filtered_children.append(child)
    node["children"] = filtered_children
    return node


def _remove_redundant_self_children(node: MoodleAuditTreeNode) -> MoodleAuditTreeNode:
    parent_kind = _stringify(node.get("kind"))
    parent_url_key = _node_url_key(_stringify(node.get("canonicalUrl") or node.get("url")))
    if not parent_url_key:
        return node

    filtered_children: list[MoodleAuditTreeNode] = []
    for child in node.get("children", []):
        child_kind = _stringify(child.get("kind"))
        child_url_key = _node_url_key(_stringify(child.get("canonicalUrl") or child.get("url")))
        if not child_url_key:
            filtered_children.append(child)
            continue
        same_target = child_url_key == parent_url_key
        if not same_target:
            filtered_children.append(child)
            continue
        if child_kind == parent_kind:
            continue
        if child_kind in {"link", "quiz", "document", "google_slides", "google_drive", "page", "section"}:
            continue
        filtered_children.append(child)
    node["children"] = filtered_children
    return node


def _remove_nested_course_navigation(node: MoodleAuditTreeNode, *, root_course_id: str) -> MoodleAuditTreeNode:
    if _stringify(node.get("kind")) == "course":
        return node
    filtered_children: list[MoodleAuditTreeNode] = []
    for child in node.get("children", []):
        child_identity = _resource_identity_key(child)
        if _is_course_section_identity(child_identity, root_course_id):
            continue
        filtered_children.append(child)
    node["children"] = filtered_children
    return node


def _normalize_node_descendants(node: MoodleAuditTreeNode, *, root_course_id: str) -> MoodleAuditTreeNode:
    normalized_children: list[MoodleAuditTreeNode] = []
    for child in node.get("children", []):
        normalized_children.append(_normalize_node_descendants(child, root_course_id=root_course_id))
    node["children"] = _semantic_dedupe_children(_dedupe_children(normalized_children))
    node["children"] = _semantic_dedupe_children(_dedupe_children(list(node.get("children", []))))
    node = _remove_nested_course_navigation(node, root_course_id=root_course_id)
    node = _remove_redundant_self_children(node)
    node = _remove_redundant_self_redirects(node)
    parent_preview_key = _node_url_key(_stringify(node.get("previewUrl")))
    parent_canonical_key = _node_url_key(_stringify(node.get("canonicalUrl") or node.get("url")))
    filtered_children: list[MoodleAuditTreeNode] = []
    for child in node.get("children", []):
        child_kind = _stringify(child.get("kind"))
        child_canonical_key = _node_url_key(_stringify(child.get("canonicalUrl") or child.get("url")))
        if child_kind == "image" and (
            (parent_preview_key and child_canonical_key == parent_preview_key)
            or (parent_canonical_key and child_canonical_key == parent_canonical_key)
        ):
            continue
        filtered_children.append(child)
    node["children"] = filtered_children
    node["children"] = _semantic_dedupe_children(_dedupe_children(list(node.get("children", []))))
    return node


def _build_presentable_tree(node: MoodleAuditTreeNode, *, root_host: str, course_name: str, root_course_id: str) -> MoodleAuditTreeNode | None:
    cloned = _clone_tree_node(node)
    presented_children: list[MoodleAuditTreeNode] = []
    for child in cloned.get("children", []):
        presented_child = _build_presentable_tree(child, root_host=root_host, course_name=course_name, root_course_id=root_course_id)
        if presented_child is not None:
            presented_children.append(presented_child)
    cloned["children"] = _dedupe_children(presented_children)
    cloned = _normalize_display_title(cloned, course_name=course_name)
    description = _stringify(cloned.get("description"))
    if _looks_like_intro_prompt(description) or _looks_like_navigation_dump(description):
        description = ""
    cloned["description"] = _trim_text(description, limit=220)
    cloned = _promote_primary_asset(cloned)
    cloned = _collapse_single_login_duplicate(cloned)
    cloned = _normalize_node_descendants(cloned, root_course_id=root_course_id)
    cloned["metadata"] = _presentable_metadata(cloned)
    if _is_noise_node(cloned, root_host=root_host, course_name=course_name, root_course_id=root_course_id):
        return None
    return cloned


def _root_course_name(snapshot: MoodleAuditSnapshot) -> str:
    for page in snapshot.pages:
        if page.page_kind == "course_home" and _stringify(page.title):
            return _stringify(page.title)
    for page in snapshot.pages:
        if page.crawl_depth == 0 and _stringify(page.title):
            return _stringify(page.title)
    if snapshot.assignments:
        return _pick_title(snapshot.assignments[0].course, "Materia Moodle")
    return "Materia Moodle"


def build_moodle_audit_tree(
    snapshot: MoodleAuditSnapshot,
    *,
    audit_json_path: str = "",
    summary_path: str = "",
) -> MoodleAuditTreePayload:
    course_name = _root_course_name(snapshot)
    root_url = ""
    for page in snapshot.pages:
        if page.page_kind == "course_home":
            root_url = _stringify(page.final_url or page.url)
            break
    root: MoodleAuditTreeNode = {
        "id": _node_id("course", snapshot.meta.job_uid or course_name, "course:root"),
        "kind": "course",
        "title": course_name,
        "url": root_url,
        "canonicalUrl": root_url,
        "previewUrl": "",
        "downloadUrl": "",
        "redirectUrl": "",
        "mimeType": "",
        "subtitle": "",
        "description": "",
        "badges": ["course"],
        "metadata": _compact_metadata(
            schema_version=snapshot.schema_version,
            warning_count=len(snapshot.warnings),
            assignment_count=len(snapshot.assignments),
        ),
        "children": [],
    }

    nodes_by_key: dict[str, MoodleAuditTreeNode] = {}
    root_children: list[MoodleAuditTreeNode] = []

    for index, page in enumerate(snapshot.pages, start=1):
        if page.page_kind == "course_home":
            # Usamos course_home como metadata del root, no como nodo duplicado.
            root["subtitle"] = root["subtitle"] or _stringify(page.subtitle)
            root["description"] = root["description"] or _trim_description(page.description or page.text_excerpt)
            root["metadata"].update(
                _compact_metadata(
                    breadcrumbs=" / ".join(item for item in page.breadcrumbs if _stringify(item)),
                    confidence_score=page.confidence_score,
                )
            )
            for child in _page_node(page, index)["children"]:
                root_children.append(child)
            continue

        node = _page_node(page, index)
        canonical_key = _node_url_key(page.final_url or page.url)
        url_key = _node_url_key(page.url)
        dedupe_key = _stringify(page.dedupe_key)
        for key in {canonical_key, url_key, dedupe_key}:
            if key:
                nodes_by_key[key] = node
        parent_key = _node_url_key(page.parent_url)
        parent_node = nodes_by_key.get(parent_key) if parent_key else None
        if parent_node is None:
            root_children.append(node)
        else:
            parent_node["children"].append(node)

    root["children"].extend(root_children)
    root_host = _host(root_url)
    root_course_id = _query_param(root_url, "id")
    presented_root = _build_presentable_tree(
        root,
        root_host=root_host,
        course_name=course_name,
        root_course_id=root_course_id,
    ) or root

    return {
        "jobUid": snapshot.meta.job_uid,
        "courseName": course_name,
        "auditPath": audit_json_path,
        "summaryPath": summary_path,
        "stats": {
            "pageCount": len(snapshot.pages),
            "retainedPageCount": int(snapshot.meta.stats.get("retained_page_count", len(snapshot.pages))),
            "externalRedirectCount": int(snapshot.meta.stats.get("external_redirect_count", 0)),
            "downloadDocumentCount": int(snapshot.meta.stats.get("download_document_count", 0)),
            "assignmentLikeCount": int(snapshot.meta.stats.get("assignment_like_count", len(snapshot.assignments))),
            "resourceTypeCounts": dict(snapshot.meta.resource_type_counts),
        },
        "root": presented_root,
    }


__all__ = ["build_moodle_audit_tree"]
