from types import SimpleNamespace


def test_frontend_bridge_extracts_moodle_tree_from_relative_audit_path(tmp_path, monkeypatch):
    import sys
    import types

    websockets_module = types.ModuleType("websockets")
    websockets_exceptions = types.ModuleType("websockets.exceptions")
    websockets_server = types.ModuleType("websockets.server")

    class _ConnectionClosed(Exception):
        pass

    class _WebSocketServerProtocol:
        pass

    async def _serve(*args, **kwargs):
        raise RuntimeError("test stub")

    websockets_exceptions.ConnectionClosed = _ConnectionClosed
    websockets_server.WebSocketServerProtocol = _WebSocketServerProtocol
    websockets_server.serve = _serve

    monkeypatch.setitem(sys.modules, "websockets", websockets_module)
    monkeypatch.setitem(sys.modules, "websockets.exceptions", websockets_exceptions)
    monkeypatch.setitem(sys.modules, "websockets.server", websockets_server)

    from application.frontend_bridge import server as bridge_server

    audit_path = tmp_path / "data" / "sessions" / "sess-1" / "moodle" / "req-1" / "audit" / "moodle-job-1__moodle_audit_snapshot.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("{}", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        bridge_server,
        "load_moodle_audit_snapshot",
        lambda path: {"pages": [], "stats": {}, "course_name": "Hardware y Sistemas Operativos"},
    )
    monkeypatch.setattr(
        bridge_server,
        "build_moodle_audit_tree",
        lambda snapshot, audit_json_path, summary_path: {
            "jobUid": "job-123",
            "courseName": "Hardware y Sistemas Operativos",
            "auditPath": audit_json_path,
            "summaryPath": summary_path,
            "stats": {
                "pageCount": 3,
                "retainedPageCount": 3,
                "externalRedirectCount": 1,
                "downloadDocumentCount": 2,
                "assignmentLikeCount": 0,
                "resourceTypeCounts": {"document": 2, "google_slides": 1},
            },
            "root": {
                "id": "course-root",
                "kind": "course",
                "title": "Hardware y Sistemas Operativos",
                "children": [],
            },
        },
    )

    final_response = (
        "Audité la materia.\n"
        "- JSON audit: data/sessions/sess-1/moodle/req-1/audit/moodle-job-1__moodle_audit_snapshot.json\n"
        "- Resumen: data/sessions/sess-1/moodle/req-1/audit/moodle-job-1__moodle_audit_summary.md"
    )

    tree = bridge_server._extract_moodle_audit_tree(final_response)

    assert tree is not None
    assert tree.jobUid == "job-123"
    assert tree.courseName == "Hardware y Sistemas Operativos"
    assert tree.auditPath.endswith("__moodle_audit_snapshot.json")


def test_persist_moodle_audit_snapshot_writes_validated_json_and_schema(tmp_path, monkeypatch):
    from features.web_scraping.application import moodle_artifacts
    from features.web_scraping.application import moodle_audit
    from features.web_scraping.application.moodle_audit import (
        load_moodle_audit_snapshot,
        persist_moodle_audit_snapshot,
    )

    monkeypatch.setattr(
        moodle_artifacts,
        "get_request_runtime_config",
        lambda: SimpleNamespace(session_id="sess-1", request_id="req-1"),
    )
    monkeypatch.setattr(
        moodle_audit,
        "get_request_runtime_config",
        lambda: SimpleNamespace(session_id="sess-1", request_id="req-1"),
    )
    monkeypatch.setattr(
        moodle_audit,
        "get_moodle_artifact_dir",
        lambda: tmp_path / "data" / "sessions" / "sess-1" / "moodle" / "req-1",
    )

    paths = persist_moodle_audit_snapshot(
        [
            {
                "name": "TP 1",
                "date": "16 de abr, 00:00",
                "course": "Historia",
                "url": "https://moodle.local/mod/assign/view.php?id=10",
                "status": "pendiente",
            }
        ],
        base_url="https://moodle.local",
        pages=[
            {
                "page_kind": "assignment_detail",
                "url": "https://moodle.local/mod/assign/view.php?id=10",
                "final_url": "https://moodle.local/mod/assign/view.php?id=10",
                "title": "Trabajo Práctico 1",
                "subtitle": "Unidad 3",
                "description": "Descripción larga",
                "resource_type": "assignment",
                "parent_url": "",
                "source_link_label": "",
                "breadcrumbs": ["Historia", "Unidad 3", "Trabajo Práctico 1"],
                "text_excerpt": "Texto visible",
                "links": [
                    {
                        "label": "Guía",
                        "url": "https://moodle.local/mod/resource/view.php?id=99",
                        "final_url": "https://moodle.local/mod/resource/view.php?id=99",
                        "redirect_target": "",
                        "redirect_chain": ["https://moodle.local/mod/resource/view.php?id=99"],
                        "resource_type": "document",
                        "is_submission_target": False,
                    }
                ],
                "attachments": [
                    {
                        "label": "Consigna.pdf",
                        "filename": "consigna.pdf",
                        "url": "https://moodle.local/pluginfile.php/123/consigna.pdf",
                        "final_url": "https://moodle.local/pluginfile.php/123/consigna.pdf",
                        "mime_hint": "pdf",
                        "kind": "document",
                    }
                ],
                "videos": [
                    {
                        "label": "Clase 1",
                        "embed_url": "https://www.youtube.com/embed/abc",
                        "watch_url": "https://www.youtube.com/watch?v=abc",
                        "provider": "youtube",
                        "preview_url": "https://img.youtube.com/vi/abc/hqdefault.jpg",
                    }
                ],
                "images": [
                    {
                        "label": "Portada",
                        "url": "https://moodle.local/image.png",
                        "kind": "image",
                    }
                ],
                "external_resource": {
                    "provider": "google_slides",
                    "resource_id": "slides-123",
                    "resource_type": "presentation",
                    "canonical_url": "https://docs.google.com/presentation/d/slides-123/edit",
                    "htmlpresent_url": "https://docs.google.com/presentation/d/slides-123/htmlpresent",
                    "preview_url": "https://lh.googleusercontent.com/preview.png",
                    "access_url": "https://accounts.google.com/ServiceLogin?continue=slides-123",
                    "download_url": "",
                    "requires_login": True,
                    "slide_count": 10,
                    "content_blocks": ["Trabajo Práctico 1", "Vista HTML de la presentación"],
                },
                "submission_state": {
                    "submission_status": "No entregado",
                    "grading_status": "No calificado",
                    "due_date_text": "16 de abr, 00:00",
                    "time_remaining_text": "Quedan 2 días",
                    "last_modified_text": "",
                    "attempt_text": "Intento 1",
                    "instructions": ["Subí un PDF con la resolución."],
                    "available_actions": ["Realizar entrega"],
                    "submitted_files": [
                        {
                            "label": "borrador.pdf",
                            "filename": "borrador.pdf",
                            "url": "https://moodle.local/pluginfile.php/999/borrador.pdf",
                            "final_url": "https://moodle.local/pluginfile.php/999/borrador.pdf",
                            "mime_hint": "pdf",
                            "kind": "document",
                        }
                    ],
                    "raw_fields": {
                        "Estado de la entrega": "No entregado",
                        "Estado de calificación": "No calificado",
                    },
                    "field_confidence": {
                        "submission_status": 0.95,
                        "grading_status": 0.95,
                        "due_date_text": 0.9,
                        "time_remaining_text": 0.8,
                        "last_modified_text": 0.0,
                        "attempt_text": 0.75,
                        "instructions": 0.8,
                        "available_actions": 0.9,
                        "submitted_files": 0.95,
                    },
                    "can_submit": True,
                    "is_submitted": True,
                    "is_graded": False,
                    "is_locked": False,
                },
                "raw_html": "<html><body><h1>Trabajo Práctico 1</h1></body></html>",
            },
            {
                "page_kind": "linked_resource",
                "url": "https://moodle.local/mod/assign/view.php?id=10",
                "final_url": "https://moodle.local/mod/assign/view.php?id=10",
                "title": "",
                "subtitle": "",
                "description": "",
                "resource_type": "assignment",
                "parent_url": "https://moodle.local/mod/assign/view.php?id=10",
                "source_link_label": "Duplicado",
                "breadcrumbs": [],
                "text_excerpt": "",
                "links": [],
                "attachments": [],
                "videos": [],
                "images": [],
                "raw_html": "<html><body>duplicado</body></html>",
            }
        ],
    )

    assert paths.audit_dir.exists()
    assert paths.json_path.exists()
    assert paths.schema_path.exists()
    assert paths.summary_path.exists()

    snapshot = load_moodle_audit_snapshot(paths.json_path)
    assert snapshot.snapshot_kind == "moodle_audit"
    assert snapshot.meta.job_uid.startswith("moodle-job-sess-1-req-1-")
    assert paths.job_uid == snapshot.meta.job_uid
    assert paths.json_path.name.startswith(f"{snapshot.meta.job_uid}__")
    assert paths.schema_path.name.startswith(f"{snapshot.meta.job_uid}__")
    assert paths.summary_path.name.startswith(f"{snapshot.meta.job_uid}__")
    assert snapshot.meta.record_count == 1
    assert snapshot.meta.stats["retained_page_count"] == 1
    assert snapshot.meta.resource_type_counts["assignment"] == 1
    assert len(snapshot.pages) == 1
    assert snapshot.assignments[0].name == "TP 1"
    assert snapshot.pages[0].title == "Trabajo Práctico 1"
    assert snapshot.pages[0].resource_type == "assignment"
    assert snapshot.pages[0].dedupe_key == "https://moodle.local/mod/assign/view.php?id=10"
    assert snapshot.pages[0].confidence_score > 0.5
    assert snapshot.pages[0].attachments[0].label == "Consigna.pdf"
    assert snapshot.pages[0].attachments[0].filename == "consigna.pdf"
    assert snapshot.pages[0].videos[0].provider == "youtube"
    assert snapshot.pages[0].videos[0].watch_url == "https://www.youtube.com/watch?v=abc"
    assert snapshot.pages[0].external_resource is not None
    assert snapshot.pages[0].external_resource.provider == "google_slides"
    assert snapshot.pages[0].external_resource.slide_count == 10
    assert snapshot.pages[0].links[0].resource_type == "document"
    assert snapshot.pages[0].submission_state is not None
    assert snapshot.pages[0].submission_state.submission_status == "No entregado"
    assert snapshot.pages[0].submission_state.available_actions[0] == "Realizar entrega"
    assert snapshot.pages[0].submission_state.can_submit is True
    assert snapshot.pages[0].submission_state.is_submitted is True
    assert snapshot.pages[0].submission_state.is_graded is False
    assert snapshot.pages[0].submission_state.submitted_files[0].kind == "document"
    assert snapshot.pages[0].submission_state.field_confidence["submission_status"] == 0.95
    assert snapshot.pages[0].html_snapshot_path.endswith(".html")
    assert f"/{snapshot.meta.job_uid}__" in snapshot.pages[0].html_snapshot_path
    summary = paths.summary_path.read_text(encoding="utf-8")
    assert "Moodle Audit Summary" in summary
    assert "Job UID" in summary
    assert "## Stats" in summary
    assert "can_submit=True" in summary


def test_prepare_moodle_assignments_payload_exposes_audit_paths():
    from integrations.google_calendar_tools import prepare_moodle_assignments_payload

    payload = {
        "valid": [],
        "invalid": [],
        "issues": [],
        "meta": {},
    }

    from unittest.mock import patch

    with (
        patch(
            "integrations.google_calendar_tools.extract_moodle_audit_bundle",
            return_value={"assignments": [], "pages": [], "warnings": []},
        ),
        patch(
            "integrations.google_calendar_tools.persist_moodle_audit_snapshot",
            return_value=SimpleNamespace(
                json_path="/tmp/audit/moodle-job-test-123__moodle_audit_snapshot.json",
                schema_path="/tmp/audit/moodle-job-test-123__moodle_audit_snapshot.schema.json",
                summary_path="/tmp/audit/moodle-job-test-123__moodle_audit_summary.md",
            ),
        ),
        patch(
            "integrations.google_calendar_tools.load_moodle_audit_snapshot",
            return_value=SimpleNamespace(meta=SimpleNamespace(job_uid="moodle-job-test-123")),
        ),
        patch("integrations.google_calendar_tools.enrich_moodle_submission_statuses", return_value=[]),
        patch("integrations.google_calendar_tools.normalize_moodle_assignments", return_value=[]),
        patch(
            "integrations.google_calendar_tools.validate_moodle_assignments",
            return_value=SimpleNamespace(valid=[], invalid=[], issues=[]),
        ),
        patch("integrations.google_calendar_tools.render_moodle_assignments_review", return_value="# review"),
        patch(
            "integrations.google_calendar_tools.persist_moodle_artifacts",
            return_value=SimpleNamespace(json_path="/tmp/moodle_tasks.json", markdown_path="/tmp/moodle_tasks.md"),
        ),
        patch("integrations.google_calendar_tools.render_moodle_assignments_chat", return_value="ok"),
    ):
        result = prepare_moodle_assignments_payload("https://moodle.local")

    assert result["audit_json_path"] == "/tmp/audit/moodle-job-test-123__moodle_audit_snapshot.json"
    assert result["audit_schema_path"] == "/tmp/audit/moodle-job-test-123__moodle_audit_snapshot.schema.json"
    assert result["audit_summary_path"] == "/tmp/audit/moodle-job-test-123__moodle_audit_summary.md"
    assert result["audit_job_uid"] == "moodle-job-test-123"


def test_prepare_moodle_course_audit_payload_exposes_course_audit_paths():
    from integrations.google_calendar_tools import prepare_moodle_course_audit_payload
    from unittest.mock import patch

    with (
        patch(
            "integrations.google_calendar_tools.extract_moodle_course_audit_bundle",
            return_value={
                "course_url": "https://moodle.local/course/view.php?id=123",
                "course_name": "Historia Argentina",
                "assignments": [
                    {
                        "name": "TP 1",
                        "date": "16 de abr, 00:00",
                        "course": "Historia Argentina",
                        "url": "https://moodle.local/mod/assign/view.php?id=10",
                        "status": "pendiente",
                        "source_stage": "course_audit",
                    }
                ],
                "pages": [
                    {
                        "page_kind": "course_home",
                        "url": "https://moodle.local/course/view.php?id=123",
                        "final_url": "https://moodle.local/course/view.php?id=123",
                        "title": "Historia Argentina",
                        "subtitle": "",
                        "description": "",
                        "resource_type": "page",
                        "parent_url": "",
                        "source_link_label": "",
                        "crawl_depth": 0,
                        "visit_order": 1,
                        "breadcrumbs": [],
                        "text_excerpt": "",
                        "links": [],
                        "attachments": [],
                        "videos": [],
                        "images": [],
                    }
                ],
                "warnings": ["depth_limit:https://moodle.local/mod/page/view.php?id=999"],
            },
        ),
        patch(
            "integrations.google_calendar_tools.persist_moodle_audit_snapshot",
            return_value=SimpleNamespace(
                json_path="/tmp/audit/moodle-job-course-123__moodle_audit_snapshot.json",
                schema_path="/tmp/audit/moodle-job-course-123__moodle_audit_snapshot.schema.json",
                summary_path="/tmp/audit/moodle-job-course-123__moodle_audit_summary.md",
            ),
        ),
        patch(
            "integrations.google_calendar_tools.load_moodle_audit_snapshot",
            return_value=SimpleNamespace(
                meta=SimpleNamespace(job_uid="moodle-job-course-123"),
                pages=[SimpleNamespace(title="Historia Argentina")],
                assignments=[SimpleNamespace(name="TP 1")],
                warnings=["depth_limit:https://moodle.local/mod/page/view.php?id=999"],
            ),
        ),
    ):
        result = prepare_moodle_course_audit_payload("/course/view.php?id=123", base_url="https://moodle.local")

    assert result["course_url"] == "https://moodle.local/course/view.php?id=123"
    assert result["course_name"] == "Historia Argentina"
    assert result["audit_json_path"] == "/tmp/audit/moodle-job-course-123__moodle_audit_snapshot.json"
    assert result["audit_job_uid"] == "moodle-job-course-123"
    assert result["page_count"] == 1
    assert result["assignment_count"] == 1
    assert result["warning_count"] == 1
    assert result["root_page_title"] == "Historia Argentina"


def test_prepare_moodle_courses_payload_lists_indexed_courses():
    from integrations.google_calendar_tools import prepare_moodle_courses_payload
    from unittest.mock import patch

    with patch(
        "integrations.google_calendar_tools.list_moodle_courses",
        return_value=[
            {"course_name": "Historia Argentina", "course_url": "https://moodle.local/course/view.php?id=123"},
            {"course_name": "Matemática I", "course_url": "https://moodle.local/course/view.php?id=456"},
        ],
    ):
        result = prepare_moodle_courses_payload(base_url="https://moodle.local")

    assert result["course_count"] == 2
    assert result["courses"][0]["index"] == 1
    assert result["courses"][1]["course_name"] == "Matemática I"


def test_prepare_moodle_course_audit_by_name_payload_resolves_and_audits():
    from integrations.google_calendar_tools import prepare_moodle_course_audit_by_name_payload
    from unittest.mock import patch

    with (
        patch(
            "integrations.google_calendar_tools.resolve_moodle_course_by_name",
            return_value={
                "query": "Historia Argentina",
                "strategy": "exact",
                "matched_course": {
                    "course_name": "Historia Argentina",
                    "course_url": "https://moodle.local/course/view.php?id=123",
                },
                "candidates": [],
                "course_count": 2,
            },
        ),
        patch(
            "integrations.google_calendar_tools.prepare_moodle_course_audit_payload",
            return_value={
                "course_url": "https://moodle.local/course/view.php?id=123",
                "course_name": "Historia Argentina",
                "audit_json_path": "/tmp/audit.json",
                "audit_schema_path": "/tmp/schema.json",
                "audit_summary_path": "/tmp/summary.md",
                "audit_job_uid": "moodle-job-123",
                "page_count": 3,
                "assignment_count": 1,
                "warning_count": 0,
                "warnings": [],
                "root_page_title": "Historia Argentina",
            },
        ),
    ):
        result = prepare_moodle_course_audit_by_name_payload("Historia Argentina", base_url="https://moodle.local")

    assert result["resolved"] is True
    assert result["resolved_from_query"] == "Historia Argentina"
    assert result["resolution_strategy"] == "exact"
    assert result["audit_job_uid"] == "moodle-job-123"


def test_prepare_moodle_course_audit_by_name_payload_returns_candidates_when_ambiguous():
    from integrations.google_calendar_tools import prepare_moodle_course_audit_by_name_payload
    from unittest.mock import patch

    with patch(
        "integrations.google_calendar_tools.resolve_moodle_course_by_name",
        return_value={
            "query": "Historia",
            "strategy": "ambiguous_contains",
            "matched_course": None,
            "candidates": [
                {"course_name": "Historia Argentina", "course_url": "https://moodle.local/course/view.php?id=123"},
                {"course_name": "Historia Universal", "course_url": "https://moodle.local/course/view.php?id=124"},
            ],
            "course_count": 5,
        },
    ):
        result = prepare_moodle_course_audit_by_name_payload("Historia", base_url="https://moodle.local")

    assert result["resolved"] is False
    assert result["strategy"] == "ambiguous_contains"
    assert len(result["candidates"]) == 2
    assert "No pude resolver" in result["message"]


def test_youtube_embed_is_expanded_to_watch_and_preview_urls():
    from bs4 import BeautifulSoup
    from features.web_scraping.infrastructure.scraping_tools import _extract_moodle_page_videos

    soup = BeautifulSoup(
        '<iframe src="https://www.youtube.com/embed/abc123" title="Clase grabada"></iframe>',
        "html.parser",
    )

    videos = _extract_moodle_page_videos(soup, "https://moodle.local/mod/page/view.php?id=1")

    assert videos[0]["embed_url"] == "https://www.youtube.com/embed/abc123"
    assert videos[0]["watch_url"] == "https://www.youtube.com/watch?v=abc123"
    assert videos[0]["preview_url"] == "https://img.youtube.com/vi/abc123/hqdefault.jpg"


def test_extract_submission_targets_detects_assignment_delivery_links():
    from bs4 import BeautifulSoup
    from features.web_scraping.infrastructure.scraping_tools import _extract_submission_targets

    soup = BeautifulSoup(
        '<a href="/mod/assign/view.php?id=77">Realizar entrega</a>'
        '<a href="/mod/resource/view.php?id=10">PDF</a>',
        "html.parser",
    )

    targets = _extract_submission_targets(soup, "https://moodle.local/mod/assign/view.php?id=12")

    assert len(targets) == 1
    assert targets[0]["url"] == "https://moodle.local/mod/assign/view.php?id=77"


def test_extract_submission_state_reads_status_actions_and_files():
    from bs4 import BeautifulSoup
    from features.web_scraping.infrastructure.scraping_tools import _extract_submission_state

    html = """
    <html><body>
      <table class="submissionstatustable">
        <tr><th>Estado de la entrega</th><td>No entregado</td></tr>
        <tr><th>Estado de calificación</th><td>No calificado</td></tr>
        <tr><th>Fecha de entrega</th><td>16 de abr, 00:00</td></tr>
        <tr><th>Tiempo restante</th><td>Quedan 2 días</td></tr>
      </table>
      <div id="intro">Subí un PDF con la resolución.</div>
      <a href="/mod/assign/view.php?id=77">Realizar entrega</a>
      <div class="fileuploadsubmission">
        <a href="/pluginfile.php/123/resolucion.pdf">resolucion.pdf</a>
      </div>
    </body></html>
    """
    soup = BeautifulSoup(html, "html.parser")

    state = _extract_submission_state(soup, "https://moodle.local/mod/assign/view.php?id=77")

    assert state is not None
    assert state["submission_status"] == "No entregado"
    assert state["grading_status"] == "No calificado"
    assert state["due_date_text"] == "16 de abr, 00:00"
    assert state["available_actions"] == ["Realizar entrega"]
    assert state["submitted_files"][0]["filename"] == "resolucion.pdf"
    assert state["can_submit"] is True
    assert state["is_submitted"] is True
    assert state["is_graded"] is False


def test_build_child_crawl_targets_prioritizes_submission_and_increases_depth():
    from features.web_scraping.infrastructure.scraping_tools import _build_child_crawl_targets

    page_payload = {
        "page_kind": "assignment_detail",
        "url": "https://moodle.local/mod/assign/view.php?id=10",
        "final_url": "https://moodle.local/mod/assign/view.php?id=10",
        "crawl_depth": 0,
        "raw_html": """
        <html><body>
          <a href="/mod/assign/view.php?id=77">Realizar entrega</a>
          <a href="/mod/page/view.php?id=88">Consigna extendida</a>
        </body></html>
        """,
        "links": [
            {
                "label": "Consigna extendida",
                "url": "https://moodle.local/mod/page/view.php?id=88",
                "final_url": "https://moodle.local/mod/page/view.php?id=88",
                "redirect_target": "",
                "redirect_chain": ["https://moodle.local/mod/page/view.php?id=88"],
                "resource_type": "page",
                "is_submission_target": False,
            }
        ],
    }

    targets = _build_child_crawl_targets(
        page_payload,
        root_host="moodle.local",
        max_depth=3,
        max_children=10,
    )

    assert len(targets) == 2
    assert targets[0]["page_kind"] == "submission_page"
    assert targets[0]["crawl_depth"] == 1
    assert targets[0]["url"] == "https://moodle.local/mod/assign/view.php?id=77"
    assert targets[1]["page_kind"] == "linked_resource"


def test_build_file_audit_record_extracts_filename_redirects_and_http_metadata():
    from features.web_scraping.infrastructure.scraping_tools import _build_file_audit_record

    record = _build_file_audit_record(
        label="Descarga",
        url="https://moodle.local/mod/resource/view.php?id=9",
        final_url="https://cdn.local/files/consigna-final.pdf",
        content_type="application/pdf; charset=binary",
        content_length=2048,
        content_disposition='attachment; filename="consigna-final.pdf"',
        status_code=200,
        redirect_chain=[
            "https://moodle.local/mod/resource/view.php?id=9",
            "https://cdn.local/files/consigna-final.pdf",
        ],
    )

    assert record["filename"] == "consigna-final.pdf"
    assert record["mime_hint"] == "pdf"
    assert record["kind"] == "document"
    assert record["redirect_target"] == "https://cdn.local/files/consigna-final.pdf"
    assert record["content_type"] == "application/pdf"
    assert record["content_length"] == 2048
    assert record["is_download"] is True


def test_session_invalid_only_flags_same_site_moodle_login_signals():
    from features.web_scraping.infrastructure.scraping_tools import _session_invalid

    class FakeLocator:
        def __init__(self, text: str):
            self._text = text

        def inner_text(self, timeout: int = 0) -> str:
            return self._text

    class FakePage:
        def __init__(self, url: str, body: str, html: str = ""):
            self.url = url
            self._body = body
            self._html = html or f"<body>{body}</body>"

        def locator(self, selector: str):
            assert selector == "body"
            return FakeLocator(self._body)

        def content(self) -> str:
            return self._html

    invalid, reason = _session_invalid(
        FakePage("https://moodle.local/login/index.php", "Nombre de usuario"),
        expected_root_host="moodle.local",
    )
    assert invalid is True
    assert "redirected_to_login" in reason

    invalid_external, reason_external = _session_invalid(
        FakePage("https://docs.google.com/presentation/d/123/edit", "Nombre de usuario"),
        expected_root_host="moodle.local",
        allow_external=True,
    )
    assert invalid_external is False
    assert reason_external == ""

    invalid_credentials, reason_credentials = _session_invalid(
        FakePage(
            "https://moodle.local/login/index.php",
            "Nombre de usuario o contraseña incorrectos",
            "<div class='loginerrors'>Nombre de usuario o contraseña incorrectos</div>",
        ),
        expected_root_host="moodle.local",
    )
    assert invalid_credentials is True
    assert "credential_error_detected" in reason_credentials
    assert "nombre de usuario o contraseña incorrectos" in reason_credentials


def test_extract_login_error_message_detects_explicit_invalid_credentials():
    from features.web_scraping.infrastructure.scraping_tools import _extract_login_error_message

    assert (
        _extract_login_error_message(
            text="Error: Nombre de usuario o contraseña incorrectos. Probá de nuevo.",
        )
        == "nombre de usuario o contraseña incorrectos. probá de nuevo."
    )
    assert (
        _extract_login_error_message(
            html="<div class='alert'>Invalid username or password</div>",
        )
        == "invalid username or password"
    )


def test_extract_login_form_debug_detects_second_form_sso_and_tokens():
    from features.web_scraping.infrastructure.scraping_tools import _extract_login_form_debug

    html = """
    <html>
      <body>
        <form action="index.php" method="post">
          <input type="hidden" name="logintoken" value="abc123" />
          <input type="text" name="username" />
          <input type="password" name="password" />
          <button type="submit">Acceder</button>
        </form>
        <form action="/auth/saml2/login.php">
          <button type="submit">Ingresar con Microsoft 365</button>
        </form>
        <a href="/auth/saml2/login.php">SSO Microsoft</a>
      </body>
    </html>
    """

    payload = _extract_login_form_debug(
        html=html,
        current_url="https://virtual.uep165.edu.ar/mld-1/login/index.php",
    )

    assert payload["form_count"] == 2
    assert payload["second_form_detected"] is True
    assert payload["login_form_found"] is True
    assert payload["sso_detected"] is True
    assert "logintoken" in payload["token_fields"]
    assert payload["forms"][0]["resolved_action"] == "https://virtual.uep165.edu.ar/mld-1/login/index.php"
    assert payload["forms"][1]["resolved_action"] == "https://virtual.uep165.edu.ar/auth/saml2/login.php"


def test_canonicalize_moodle_crawl_url_dedupes_lang_and_keeps_functional_params():
    from features.web_scraping.infrastructure.scraping_tools import _canonicalize_moodle_crawl_url

    assert (
        _canonicalize_moodle_crawl_url(
            "https://moodle.local/mod/forum/view.php?id=6487&lang=en#section-2",
            root_host="moodle.local",
        )
        == "https://moodle.local/mod/forum/view.php?id=6487"
    )
    assert (
        _canonicalize_moodle_crawl_url(
            "https://moodle.local/mod/quiz/attempt.php?attempt=17&cmid=2&lang=es",
            root_host="moodle.local",
        )
        == "https://moodle.local/mod/quiz/attempt.php?attempt=17&cmid=2"
    )


def test_build_download_backed_audit_page_creates_document_payload():
    from features.web_scraping.infrastructure.scraping_tools import _build_download_backed_audit_page

    payload = _build_download_backed_audit_page(
        target_url="https://moodle.local/mod/resource/view.php?id=8000",
        page_kind="linked_resource",
        source_link_label="Consigna PDF",
        parent_url="https://moodle.local/course/view.php?id=332",
        crawl_depth=1,
        visit_order=3,
        metadata={
            "label": "Consigna PDF",
            "filename": "consigna-final.pdf",
            "final_url": "https://cdn.local/files/consigna-final.pdf",
            "mime_hint": "pdf",
            "kind": "document",
            "content_type": "application/pdf",
            "content_length": 2048,
            "content_disposition": 'attachment; filename=\"consigna-final.pdf\"',
            "status_code": 200,
            "redirect_chain": [
                "https://moodle.local/mod/resource/view.php?id=8000",
                "https://cdn.local/files/consigna-final.pdf",
            ],
        },
    )

    assert payload["resource_type"] == "document"
    assert payload["attachments"][0]["filename"] == "consigna-final.pdf"
    assert payload["attachments"][0]["final_url"] == "https://cdn.local/files/consigna-final.pdf"
    assert payload["raw_html"] == ""


def test_extract_google_slides_external_resource_metadata():
    from bs4 import BeautifulSoup
    from features.web_scraping.infrastructure.scraping_tools import _extract_moodle_page_metadata

    html = """
    <html>
      <head>
        <title>Hardware y SO - Fabricación del CPU - Presentaciones de Google</title>
        <meta property="og:title" content="Hardware y SO - Fabricación del CPU">
        <meta property="og:image" content="https://lh.googleusercontent.com/preview.png">
      </head>
      <body>
        <a href="https://accounts.google.com/ServiceLogin?continue=slides">Acceder</a>
        <a href="/presentation/d/slide-doc-123/htmlpresent">Vista HTML de la presentación</a>
        <div>Solo ver 1 2 3 4 5 6 7 8 9 10</div>
      </body>
    </html>
    """

    page = _extract_moodle_page_metadata(
        "linked_resource",
        "https://moodle.local/mod/url/view.php?id=11348",
        "https://docs.google.com/presentation/d/slide-doc-123/edit?slide=id.g1#slide=id.g1",
        html,
        0,
    )

    assert page["resource_type"] == "google_slides"
    assert page["external_resource"]["provider"] == "google_slides"
    assert page["external_resource"]["resource_id"] == "slide-doc-123"
    assert page["external_resource"]["htmlpresent_url"].endswith("/htmlpresent")
    assert page["external_resource"]["preview_url"] == "https://lh.googleusercontent.com/preview.png"
    assert page["external_resource"]["requires_login"] is True
    assert page["external_resource"]["slide_count"] == 10


def test_extract_google_drive_external_resource_metadata():
    from features.web_scraping.infrastructure.scraping_tools import _extract_moodle_page_metadata

    html = """
    <html>
      <head>
        <title>Proceso de fabricación de un microchip.mp4 - Google Drive</title>
        <meta property="og:title" content="Proceso de fabricación de un microchip.mp4">
      </head>
      <body>
        <a href="https://accounts.google.com/ServiceLogin?continue=drive">Acceder</a>
        <img src="https://drive.google.com/drive-viewer/preview-image=s1600-rw-v1" />
        <div>Transcripción Descargar Abrir Detalles</div>
      </body>
    </html>
    """

    page = _extract_moodle_page_metadata(
        "linked_resource",
        "https://moodle.local/mod/url/view.php?id=12120",
        "https://drive.google.com/file/d/drive-file-123/view",
        html,
        0,
    )

    assert page["resource_type"] == "google_drive"
    assert page["external_resource"]["provider"] == "google_drive"
    assert page["external_resource"]["resource_id"] == "drive-file-123"
    assert page["external_resource"]["resource_type"] == "video"
    assert page["external_resource"]["preview_url"] == "https://drive.google.com/drive-viewer/preview-image=s1600-rw-v1"
    assert page["external_resource"]["download_url"] == "https://drive.google.com/uc?export=download&id=drive-file-123"
    assert page["external_resource"]["requires_login"] is True


def test_extract_assignment_records_from_pages_builds_course_assignments():
    from features.web_scraping.infrastructure.scraping_tools import _extract_assignment_records_from_pages

    pages = [
        {
            "page_kind": "assignment_detail",
            "resource_type": "assignment",
            "title": "TP 1",
            "source_link_label": "",
            "final_url": "https://moodle.local/mod/assign/view.php?id=10",
            "submission_state": {
                "due_date_text": "16 de abr, 00:00",
                "is_graded": False,
                "is_submitted": False,
                "can_submit": True,
                "is_locked": False,
                "submission_status": "No entregado",
            },
        }
    ]

    records = _extract_assignment_records_from_pages(pages, course_name="Historia Argentina")

    assert len(records) == 1
    assert records[0]["name"] == "TP 1"
    assert records[0]["course"] == "Historia Argentina"
    assert records[0]["date"] == "16 de abr, 00:00"
    assert records[0]["status"] == "pendiente"
    assert records[0]["source_stage"] == "course_audit"


def test_extract_visible_moodle_courses_from_html_finds_courses():
    from features.web_scraping.infrastructure.scraping_tools import _extract_visible_moodle_courses_from_html

    html = """
    <html><body>
      <div class="card"><a href="/course/view.php?id=123">Historia Argentina</a></div>
      <div class="card"><a href="/course/view.php?id=456">Matemática I</a></div>
    </body></html>
    """

    courses = _extract_visible_moodle_courses_from_html(html, "https://moodle.local/my/")

    assert len(courses) == 2
    assert courses[0]["course_name"] == "Historia Argentina"
    assert courses[0]["course_url"] == "https://moodle.local/course/view.php?id=123"


def test_resolve_course_match_supports_exact_and_index_queries():
    from features.web_scraping.infrastructure.scraping_tools import _resolve_course_match

    courses = [
        {"course_name": "Historia Argentina", "course_url": "https://moodle.local/course/view.php?id=123"},
        {"course_name": "Matemática I", "course_url": "https://moodle.local/course/view.php?id=456"},
    ]

    match_by_name, _, strategy_name = _resolve_course_match(courses, "Historia Argentina")
    match_by_index, _, strategy_index = _resolve_course_match(courses, "2")

    assert strategy_name == "exact"
    assert match_by_name is not None and match_by_name["course_url"].endswith("id=123")
    assert strategy_index == "index"
    assert match_by_index is not None and match_by_index["course_name"] == "Matemática I"


def test_build_moodle_audit_tree_exposes_hierarchy_and_assets():
    from features.web_scraping.application.moodle_audit_tree import build_moodle_audit_tree
    from features.web_scraping.domain.moodle_audit_models import (
        MoodleAuditAttachment,
        MoodleAuditExternalResource,
        MoodleAuditImage,
        MoodleAuditLink,
        MoodleAuditMeta,
        MoodleAuditPage,
        MoodleAuditSnapshot,
        MoodleAuditSubmissionFile,
        MoodleAuditSubmissionState,
        MoodleAuditVideo,
    )

    snapshot = MoodleAuditSnapshot(
        meta=MoodleAuditMeta(
            job_uid="moodle-job-test-tree",
            base_url="https://moodle.local",
            stats={
                "retained_page_count": 2,
                "external_redirect_count": 1,
                "download_document_count": 1,
                "assignment_like_count": 0,
            },
            resource_type_counts={
                "forum": 1,
                "document": 1,
                "google_slides": 1,
                "google_drive": 1,
            },
        ),
        pages=[
            MoodleAuditPage(
                page_kind="course_home",
                url="https://moodle.local/course/view.php?id=1",
                final_url="https://moodle.local/course/view.php?id=1",
                title="Hardware y Sistemas Operativos",
                description="Inicio de la materia",
                resource_type="page",
            ),
            MoodleAuditPage(
                page_kind="forum_view",
                url="https://moodle.local/mod/forum/view.php?id=20",
                final_url="https://moodle.local/mod/forum/view.php?id=20",
                title="Foro general",
                description="Canal de consultas",
                resource_type="forum",
                parent_url="https://moodle.local/course/view.php?id=1",
                visit_order=2,
                crawl_depth=1,
                links=[
                    MoodleAuditLink(
                        label="TP publicado",
                        url="https://moodle.local/mod/url/view.php?id=50",
                        final_url="https://docs.google.com/presentation/d/abc123/edit",
                        redirect_target="https://docs.google.com/presentation/d/abc123/edit",
                        resource_type="link",
                    )
                ],
                attachments=[
                    MoodleAuditAttachment(
                        label="consigna.pdf",
                        filename="consigna.pdf",
                        url="https://moodle.local/pluginfile.php/1/consigna.pdf",
                        final_url="https://moodle.local/pluginfile.php/1/consigna.pdf",
                        mime_hint="pdf",
                        content_type="application/pdf",
                        is_download=True,
                    )
                ],
                videos=[
                    MoodleAuditVideo(
                        label="Clase grabada",
                        embed_url="https://www.youtube.com/embed/abc",
                        watch_url="https://www.youtube.com/watch?v=abc",
                        provider="youtube",
                        preview_url="https://img.youtube.com/vi/abc/hqdefault.jpg",
                    )
                ],
                images=[
                    MoodleAuditImage(
                        label="Banner",
                        url="https://moodle.local/pluginfile.php/1/banner.png",
                    )
                ],
                external_resource=MoodleAuditExternalResource(
                    provider="google_slides",
                    resource_id="abc123",
                    resource_type="presentation",
                    canonical_url="https://docs.google.com/presentation/d/abc123/edit",
                    htmlpresent_url="https://docs.google.com/presentation/d/abc123/htmlpresent",
                    preview_url="https://lh.googleusercontent.com/preview.png",
                    access_url="https://accounts.google.com/ServiceLogin?continue=slides",
                    requires_login=True,
                    slide_count=12,
                    content_blocks=["TP 1", "Presentación de diapositivas"],
                ),
                submission_state=MoodleAuditSubmissionState(
                    submission_status="No entregado",
                    grading_status="No calificado",
                    can_submit=True,
                    submitted_files=[
                        MoodleAuditSubmissionFile(
                            label="respuesta.pdf",
                            filename="respuesta.pdf",
                            url="https://moodle.local/pluginfile.php/9/respuesta.pdf",
                            final_url="https://moodle.local/pluginfile.php/9/respuesta.pdf",
                            mime_hint="pdf",
                        )
                    ],
                ),
            ),
        ],
        warnings=[],
        assignments=[],
    )

    tree = build_moodle_audit_tree(
        snapshot,
        audit_json_path="/tmp/moodle-job-test-tree__moodle_audit_snapshot.json",
        summary_path="/tmp/moodle-job-test-tree__moodle_audit_summary.md",
    )

    assert tree["courseName"] == "Hardware y Sistemas Operativos"
    assert tree["stats"]["pageCount"] == 2
    assert tree["stats"]["resourceTypeCounts"]["google_slides"] == 1
    root = tree["root"]
    assert root["kind"] == "course"
    assert root["title"] == "Hardware y Sistemas Operativos"

    forum_node = next(child for child in root["children"] if child["kind"] == "forum")
    assert forum_node["title"] == "Foro general"
    assert "can_submit" in forum_node["badges"]

    child_kinds = {child["kind"] for child in forum_node["children"]}
    assert {"document", "video", "image", "google_slides"} <= child_kinds

    external_node = next(child for child in forum_node["children"] if child["kind"] == "google_slides")
    assert external_node["previewUrl"] == "https://lh.googleusercontent.com/preview.png"
    assert "slide_count" not in external_node["metadata"]
    assert any(grandchild["kind"] == "external_redirect" for grandchild in external_node["children"])


def test_build_moodle_audit_tree_prunes_navigation_noise_and_dedupes_duplicate_resources():
    from features.web_scraping.application.moodle_audit_tree import build_moodle_audit_tree
    from features.web_scraping.domain.moodle_audit_models import (
        MoodleAuditAttachment,
        MoodleAuditLink,
        MoodleAuditMeta,
        MoodleAuditPage,
        MoodleAuditSnapshot,
    )

    snapshot = MoodleAuditSnapshot(
        meta=MoodleAuditMeta(
            job_uid="moodle-job-presentable-tree",
            base_url="https://moodle.local",
            stats={"retained_page_count": 2},
            resource_type_counts={"document": 1},
        ),
        pages=[
            MoodleAuditPage(
                page_kind="course_home",
                url="https://moodle.local/course/view.php?id=332",
                final_url="https://moodle.local/course/view.php?id=332",
                title="Hardware y Sistemas Operativos",
                description="Inicio",
                resource_type="page",
                links=[
                    MoodleAuditLink(
                        label="Perfil",
                        url="https://moodle.local/user/profile.php?id=1",
                        final_url="https://moodle.local/user/profile.php?id=1",
                        resource_type="link",
                    ),
                    MoodleAuditLink(
                        label="Programa de contenidos Archivo",
                        url="https://moodle.local/mod/resource/view.php?id=8000",
                        final_url="https://moodle.local/pluginfile.php/23045/programa.pdf",
                        redirect_target="https://moodle.local/pluginfile.php/23045/programa.pdf",
                        resource_type="document",
                    ),
                    MoodleAuditLink(
                        label="CPU - diapositivas de clase URL",
                        url="https://moodle.local/mod/url/view.php?id=11348",
                        final_url="https://docs.google.com/presentation/d/abc123/edit",
                        redirect_target="https://docs.google.com/presentation/d/abc123/edit",
                        resource_type="link",
                    ),
                ],
            ),
            MoodleAuditPage(
                page_kind="linked_resource",
                url="https://moodle.local/mod/resource/view.php?id=8000",
                final_url="https://moodle.local/pluginfile.php/23045/programa.pdf",
                title="Programa de contenidos - 2026.pdf",
                description="Documento principal de la materia",
                resource_type="document",
                parent_url="https://moodle.local/course/view.php?id=332",
                attachments=[
                    MoodleAuditAttachment(
                        label="Programa de contenidos - 2026.pdf",
                        filename="Programa de contenidos - 2026.pdf",
                        url="https://moodle.local/mod/resource/view.php?id=8000",
                        final_url="https://moodle.local/pluginfile.php/23045/programa.pdf",
                        content_type="application/pdf",
                        is_download=True,
                    )
                ],
            ),
        ],
        warnings=[],
        assignments=[],
    )

    tree = build_moodle_audit_tree(snapshot)
    root = tree["root"]
    child_titles = [child["title"] for child in root["children"]]

    assert "Perfil" not in child_titles
    assert "Programa de contenidos Archivo" not in child_titles
    assert "Programa de contenidos - 2026.pdf" in child_titles
    assert "CPU - diapositivas de clase URL" in child_titles

    document_node = next(child for child in root["children"] if child["title"] == "Programa de contenidos - 2026.pdf")
    assert document_node["downloadUrl"] == "https://moodle.local/pluginfile.php/23045/programa.pdf"
    assert any(grandchild["kind"] == "external_redirect" for grandchild in document_node["children"])


def test_build_moodle_audit_tree_prunes_forum_actions_and_google_embed_noise():
    from features.web_scraping.application.moodle_audit_tree import build_moodle_audit_tree
    from features.web_scraping.domain.moodle_audit_models import (
        MoodleAuditLink,
        MoodleAuditMeta,
        MoodleAuditPage,
        MoodleAuditSnapshot,
        MoodleAuditVideo,
        MoodleAuditExternalResource,
    )

    snapshot = MoodleAuditSnapshot(
        meta=MoodleAuditMeta(
            job_uid="moodle-job-noise-tree",
            base_url="https://moodle.local",
            stats={"retained_page_count": 2},
            resource_type_counts={"forum": 1, "google_slides": 1},
        ),
        pages=[
            MoodleAuditPage(
                page_kind="course_home",
                url="https://moodle.local/course/view.php?id=332",
                final_url="https://moodle.local/course/view.php?id=332",
                title="Hardware y Sistemas Operativos",
                resource_type="page",
            ),
            MoodleAuditPage(
                page_kind="linked_resource",
                url="https://moodle.local/mod/forum/view.php?id=6487",
                final_url="https://moodle.local/mod/forum/view.php?id=6487",
                title="Foro de presentación",
                description="Presentate en el foro",
                resource_type="forum",
                parent_url="https://moodle.local/course/view.php?id=332",
                links=[
                    MoodleAuditLink(
                        label="English ‎(en)‎",
                        url="https://moodle.local/mod/forum/view.php?id=6487&lang=en",
                        final_url="https://moodle.local/mod/forum/view.php?id=6487&lang=en",
                        resource_type="forum",
                    ),
                    MoodleAuditLink(
                        label="Suscribir",
                        url="https://moodle.local/mod/forum/subscribe.php?id=373&sesskey=abc",
                        final_url="https://moodle.local/mod/forum/view.php?f=373",
                        redirect_target="https://moodle.local/mod/forum/view.php?f=373",
                        resource_type="forum",
                    ),
                    MoodleAuditLink(
                        label="Presentación",
                        url="https://moodle.local/mod/forum/discuss.php?d=3103",
                        final_url="https://moodle.local/mod/forum/discuss.php?d=3103",
                        resource_type="forum",
                    ),
                    MoodleAuditLink(
                        label="MACIAS Nestor",
                        url="https://moodle.local/user/view.php?id=3256&course=332",
                        final_url="https://moodle.local/user/view.php?id=3256&course=332",
                        resource_type="link",
                    ),
                ],
            ),
            MoodleAuditPage(
                page_kind="linked_resource",
                url="https://moodle.local/mod/url/view.php?id=11348",
                final_url="https://docs.google.com/presentation/d/abc123/edit",
                title="CPU - diapositivas de clase",
                description="Slides de CPU",
                resource_type="google_slides",
                parent_url="https://moodle.local/course/view.php?id=332",
                videos=[
                    MoodleAuditVideo(
                        label="about:blank",
                        embed_url="about:blank",
                        provider="iframe",
                    ),
                    MoodleAuditVideo(
                        label="Tarjeta de información",
                        embed_url="https://contacts.google.com/widget/hovercard/v/2?foo=1",
                        provider="iframe",
                    ),
                ],
                external_resource=MoodleAuditExternalResource(
                    provider="google_slides",
                    resource_id="abc123",
                    resource_type="presentation",
                    canonical_url="https://docs.google.com/presentation/d/abc123/edit",
                    htmlpresent_url="https://docs.google.com/presentation/d/abc123/htmlpresent",
                    preview_url="https://lh.googleusercontent.com/preview.png",
                    access_url="https://accounts.google.com/ServiceLogin?continue=slides",
                    requires_login=True,
                    slide_count=12,
                    content_blocks=["CPU", "Presentación de diapositivas"],
                ),
                links=[
                    MoodleAuditLink(
                        label="Vista HTML de la presentación",
                        url="https://docs.google.com/presentation/d/abc123/htmlpresent",
                        final_url="https://docs.google.com/presentation/d/abc123/htmlpresent",
                        resource_type="google_slides",
                    ),
                    MoodleAuditLink(
                        label="Acceder",
                        url="https://accounts.google.com/ServiceLogin?continue=slides",
                        final_url="https://accounts.google.com/ServiceLogin?continue=slides",
                        resource_type="google_slides",
                    ),
                ],
            ),
        ],
        warnings=[],
        assignments=[],
    )

    tree = build_moodle_audit_tree(snapshot)
    root = tree["root"]

    root_titles = [child["title"] for child in root["children"]]
    assert "Foro de presentación" not in root_titles

    slides_node = next(child for child in root["children"] if child["title"] == "CPU - diapositivas de clase")
    slides_child_titles = [child["title"] for child in slides_node["children"]]
    assert "about:blank" not in slides_child_titles
    assert "Tarjeta de información" not in slides_child_titles
    assert "Vista HTML de la presentación" not in slides_child_titles
    assert "Acceder" not in slides_child_titles
    assert slides_node["previewUrl"] == "https://lh.googleusercontent.com/preview.png"


def test_build_moodle_audit_tree_hides_intro_prompt_and_redirect_hosts():
    from features.web_scraping.application.moodle_audit_tree import build_moodle_audit_tree
    from features.web_scraping.domain.moodle_audit_models import (
        MoodleAuditLink,
        MoodleAuditMeta,
        MoodleAuditPage,
        MoodleAuditSnapshot,
    )

    snapshot = MoodleAuditSnapshot(
        meta=MoodleAuditMeta(
            job_uid="moodle-job-intro-tree",
            base_url="https://moodle.local",
            stats={"retained_page_count": 1},
            resource_type_counts={"forum": 1},
        ),
        pages=[
            MoodleAuditPage(
                page_kind="course_home",
                url="https://moodle.local/course/view.php?id=332",
                final_url="https://moodle.local/course/view.php?id=332",
                title="Hardware y Sistemas Operativos",
                description="Para comenzar, los invito a presentarse brevemente: indiquen su nombre, qué esperan aprender y qué experiencia previa tienen en hardwares y sistemas operativos (aunque sea nula).",
                resource_type="page",
                links=[
                    MoodleAuditLink(
                        label="English (en)",
                        url="https://moodle.local/course/view.php?id=332&lang=en",
                        final_url="https://moodle.local/course/view.php?id=332&lang=en",
                        resource_type="link",
                    ),
                ],
            ),
        ],
        warnings=[],
        assignments=[],
    )

    tree = build_moodle_audit_tree(snapshot)
    root = tree["root"]

    assert root["description"] == ""
    assert all("english" not in child["title"].lower() for child in root["children"])

    redirect_child = {
        "id": "redirect:test",
        "kind": "external_redirect",
        "title": "Redirect detectado",
        "url": "https://moodle.local/from",
        "canonicalUrl": "https://moodle.local/from",
        "redirectUrl": "https://docs.google.com/to",
        "previewUrl": "",
        "downloadUrl": "",
        "mimeType": "",
        "subtitle": "",
        "description": "from -> to",
        "badges": ["redirect"],
        "metadata": {"origin_host": "moodle.local", "target_host": "docs.google.com"},
        "children": [],
    }

    # smoke-check helper outcome through presentable build result shape
    from features.web_scraping.application.moodle_audit_tree import _presentable_metadata

    assert _presentable_metadata(redirect_child) == {}


def test_build_moodle_audit_tree_hides_student_presentations_and_other_course_links():
    from features.web_scraping.application.moodle_audit_tree import build_moodle_audit_tree
    from features.web_scraping.domain.moodle_audit_models import (
        MoodleAuditLink,
        MoodleAuditMeta,
        MoodleAuditPage,
        MoodleAuditSnapshot,
    )

    snapshot = MoodleAuditSnapshot(
        meta=MoodleAuditMeta(
            job_uid="moodle-job-presentations",
            base_url="https://moodle.local",
            stats={"retained_page_count": 2},
            resource_type_counts={"forum": 1},
        ),
        pages=[
            MoodleAuditPage(
                page_kind="course_home",
                url="https://moodle.local/course/view.php?id=332",
                final_url="https://moodle.local/course/view.php?id=332",
                title="Hardware y Sistemas Operativos",
                resource_type="page",
            ),
            MoodleAuditPage(
                page_kind="linked_resource",
                url="https://moodle.local/mod/forum/discuss.php?d=3103",
                final_url="https://moodle.local/mod/forum/discuss.php?d=3103",
                title="Hardware y Sistemas Operativos",
                subtitle="Foro de presentación",
                description="Mi nombre es Néstor, me gusta mucho esta carrera y tengo leve noción de hardware.",
                resource_type="forum",
                parent_url="https://moodle.local/course/view.php?id=332",
                breadcrumbs=["Página Principal", "Mis cursos", "Hardware y Sistemas Operativos", "General", "Foro de presentación", "Presentación"],
                links=[
                    MoodleAuditLink(
                        label="Programación I",
                        url="https://moodle.local/course/view.php?id=328",
                        final_url="https://moodle.local/course/view.php?id=328",
                        resource_type="link",
                    ),
                    MoodleAuditLink(
                        label="MACIAS Nestor",
                        url="https://moodle.local/user/view.php?id=3256&course=332",
                        final_url="https://moodle.local/user/view.php?id=3256&course=332",
                        resource_type="link",
                    ),
                ],
            ),
        ],
        warnings=[],
        assignments=[],
    )

    tree = build_moodle_audit_tree(snapshot)
    root = tree["root"]
    child_titles = [child["title"] for child in root["children"]]

    assert "Hardware y Sistemas Operativos" not in child_titles
    assert "Programación I" not in child_titles
    assert "MACIAS Nestor" not in child_titles


def test_build_moodle_audit_tree_hides_other_course_links_from_same_host_root():
    from features.web_scraping.application.moodle_audit_tree import build_moodle_audit_tree
    from features.web_scraping.domain.moodle_audit_models import (
        MoodleAuditLink,
        MoodleAuditMeta,
        MoodleAuditPage,
        MoodleAuditSnapshot,
    )

    snapshot = MoodleAuditSnapshot(
        meta=MoodleAuditMeta(
            job_uid="moodle-job-other-course-root",
            base_url="https://moodle.local",
            stats={"retained_page_count": 1},
            resource_type_counts={"link": 2},
        ),
        pages=[
            MoodleAuditPage(
                page_kind="course_home",
                url="https://moodle.local/course/view.php?id=332",
                final_url="https://moodle.local/course/view.php?id=332",
                title="Hardware y Sistemas Operativos",
                resource_type="page",
                links=[
                    MoodleAuditLink(
                        label="Módulo 1",
                        url="https://moodle.local/course/view.php?id=332&section=3",
                        final_url="https://moodle.local/course/view.php?id=332&section=3",
                        resource_type="course_section",
                    ),
                    MoodleAuditLink(
                        label="Programación I",
                        url="https://moodle.local/course/view.php?id=328",
                        final_url="https://moodle.local/course/view.php?id=328",
                        resource_type="link",
                    ),
                ],
            ),
        ],
        warnings=[],
        assignments=[],
    )

    tree = build_moodle_audit_tree(snapshot)
    root = tree["root"]
    child_titles = [child["title"] for child in root["children"]]

    assert "Módulo 1" in child_titles
    assert "Programación I" not in child_titles


def test_build_moodle_audit_tree_removes_presentation_and_announcements_forums_and_technical_metadata():
    from features.web_scraping.application.moodle_audit_tree import build_moodle_audit_tree
    from features.web_scraping.domain.moodle_audit_models import (
        MoodleAuditMeta,
        MoodleAuditPage,
        MoodleAuditSnapshot,
    )

    snapshot = MoodleAuditSnapshot(
        meta=MoodleAuditMeta(
            job_uid="moodle-job-forum-prune",
            base_url="https://moodle.local",
            stats={"retained_page_count": 3},
            resource_type_counts={"forum": 2, "document": 1},
        ),
        pages=[
            MoodleAuditPage(
                page_kind="course_home",
                url="https://moodle.local/course/view.php?id=332",
                final_url="https://moodle.local/course/view.php?id=332",
                title="Hardware y Sistemas Operativos",
                resource_type="page",
                breadcrumbs=["Página Principal", "Mis cursos", "Hardware y Sistemas Operativos"],
                visit_order=1,
                crawl_depth=0,
                confidence_score=0.9,
                extracted_items_count=5,
            ),
            MoodleAuditPage(
                page_kind="linked_resource",
                url="https://moodle.local/mod/forum/view.php?id=6487",
                final_url="https://moodle.local/mod/forum/view.php?id=6487",
                title="Hardware y Sistemas Operativos",
                subtitle="Foro de presentación",
                resource_type="forum",
                parent_url="https://moodle.local/course/view.php?id=332",
            ),
            MoodleAuditPage(
                page_kind="linked_resource",
                url="https://moodle.local/mod/forum/view.php?id=6509",
                final_url="https://moodle.local/mod/forum/view.php?id=6509",
                title="Hardware y Sistemas Operativos",
                subtitle="Avisos y debates",
                resource_type="forum",
                parent_url="https://moodle.local/course/view.php?id=332",
            ),
            MoodleAuditPage(
                page_kind="linked_resource",
                url="https://moodle.local/mod/resource/view.php?id=8000",
                final_url="https://moodle.local/pluginfile.php/23045/programa.pdf",
                title="Programa.pdf",
                resource_type="document",
                parent_url="https://moodle.local/course/view.php?id=332",
                visit_order=4,
                crawl_depth=1,
                confidence_score=0.6,
                extracted_items_count=1,
            ),
        ],
        warnings=[],
        assignments=[],
    )

    tree = build_moodle_audit_tree(snapshot)
    root = tree["root"]
    child_titles = [child["title"] for child in root["children"]]

    assert "Programa.pdf" in child_titles
    assert all("foro de presentación" not in (child.get("subtitle", "").lower()) for child in root["children"])
    assert all("avisos y debates" not in (child.get("subtitle", "").lower()) for child in root["children"])
    assert "breadcrumbs" not in root["metadata"]
    assert "visit_order" not in root["metadata"]
    assert "crawl_depth" not in root["metadata"]
    assert "confidence_score" not in root["metadata"]
    assert "extracted_items_count" not in root["metadata"]


def test_build_moodle_audit_tree_hides_low_value_resource_metadata():
    from features.web_scraping.application.moodle_audit_tree import build_moodle_audit_tree
    from features.web_scraping.domain.moodle_audit_models import (
        MoodleAuditAttachment,
        MoodleAuditExternalResource,
        MoodleAuditMeta,
        MoodleAuditPage,
        MoodleAuditSnapshot,
    )

    snapshot = MoodleAuditSnapshot(
        meta=MoodleAuditMeta(
            job_uid="moodle-job-hidden-metadata",
            base_url="https://moodle.local",
            stats={"retained_page_count": 1},
            resource_type_counts={"document": 1, "google_slides": 1},
        ),
        pages=[
            MoodleAuditPage(
                page_kind="course_home",
                url="https://moodle.local/course/view.php?id=332",
                final_url="https://moodle.local/course/view.php?id=332",
                title="Hardware y Sistemas Operativos",
                resource_type="page",
                attachments=[
                    MoodleAuditAttachment(
                        label="Programa.pdf",
                        filename="Programa.pdf",
                        url="https://moodle.local/mod/resource/view.php?id=8000",
                        final_url="https://moodle.local/pluginfile.php/23045/programa.pdf",
                        content_type="application/pdf",
                        content_length=243085,
                        content_disposition='inline; filename="Programa.pdf"',
                        status_code=200,
                        is_download=True,
                    )
                ],
                external_resource=MoodleAuditExternalResource(
                    provider="google_slides",
                    resource_id="abc123",
                    resource_type="presentation",
                    canonical_url="https://docs.google.com/presentation/d/abc123/edit",
                    htmlpresent_url="https://docs.google.com/presentation/d/abc123/htmlpresent",
                    preview_url="https://lh.googleusercontent.com/preview.png",
                    access_url="https://accounts.google.com/ServiceLogin?continue=slides",
                    requires_login=True,
                    slide_count=12,
                ),
            ),
        ],
        warnings=[],
        assignments=[],
    )

    tree = build_moodle_audit_tree(snapshot)
    root = tree["root"]

    attachment_node = next(child for child in root["children"] if child["kind"] == "document")
    assert "content_length" not in attachment_node["metadata"]
    assert "content_disposition" not in attachment_node["metadata"]
    assert "status_code" not in attachment_node["metadata"]
    assert "final_url" not in attachment_node["metadata"]
    slides_node = next(child for child in root["children"] if child["kind"] == "google_slides")
    assert "resource_id" not in slides_node["metadata"]
    assert "slide_count" not in slides_node["metadata"]
    assert "requires_login" not in slides_node["metadata"]
    assert "submission_target" not in slides_node["metadata"]


def test_build_moodle_audit_tree_removes_duplicate_promoted_asset_child_and_redirects():
    from features.web_scraping.application.moodle_audit_tree import build_moodle_audit_tree
    from features.web_scraping.domain.moodle_audit_models import (
        MoodleAuditAttachment,
        MoodleAuditMeta,
        MoodleAuditPage,
        MoodleAuditSnapshot,
    )

    snapshot = MoodleAuditSnapshot(
        meta=MoodleAuditMeta(
            job_uid="moodle-job-dedupe-asset",
            base_url="https://moodle.local",
            stats={"retained_page_count": 2},
            resource_type_counts={"document": 1},
        ),
        pages=[
            MoodleAuditPage(
                page_kind="course_home",
                url="https://moodle.local/course/view.php?id=332",
                final_url="https://moodle.local/course/view.php?id=332",
                title="Hardware y Sistemas Operativos",
                resource_type="page",
            ),
            MoodleAuditPage(
                page_kind="linked_resource",
                url="https://moodle.local/mod/resource/view.php?id=8000",
                final_url="https://moodle.local/pluginfile.php/23045/programa.pdf",
                title="Programa.pdf",
                resource_type="document",
                parent_url="https://moodle.local/course/view.php?id=332",
                attachments=[
                    MoodleAuditAttachment(
                        label="Programa.pdf",
                        filename="Programa.pdf",
                        url="https://moodle.local/mod/resource/view.php?id=8000",
                        final_url="https://moodle.local/pluginfile.php/23045/programa.pdf",
                        content_type="application/pdf",
                        is_download=True,
                    )
                ],
            ),
        ],
        warnings=[],
        assignments=[],
    )

    tree = build_moodle_audit_tree(snapshot)
    root = tree["root"]
    document_node = next(child for child in root["children"] if child["kind"] == "document")
    document_children_kinds = [child["kind"] for child in document_node["children"]]

    assert document_children_kinds.count("document") == 0
    assert document_children_kinds.count("external_redirect") == 1


def test_build_moodle_audit_tree_removes_self_duplicate_links_and_quiz_children():
    from features.web_scraping.application.moodle_audit_tree import build_moodle_audit_tree
    from features.web_scraping.domain.moodle_audit_models import (
        MoodleAuditLink,
        MoodleAuditMeta,
        MoodleAuditPage,
        MoodleAuditSnapshot,
    )

    snapshot = MoodleAuditSnapshot(
        meta=MoodleAuditMeta(
            job_uid="moodle-job-self-duplicates",
            base_url="https://moodle.local",
            stats={"retained_page_count": 2},
            resource_type_counts={"course_section": 1, "quiz": 1},
        ),
        pages=[
            MoodleAuditPage(
                page_kind="course_home",
                url="https://moodle.local/course/view.php?id=332",
                final_url="https://moodle.local/course/view.php?id=332",
                title="Hardware y Sistemas Operativos",
                resource_type="page",
            ),
            MoodleAuditPage(
                page_kind="course_section",
                url="https://moodle.local/course/view.php?id=332&section=4",
                final_url="https://moodle.local/course/view.php?id=332&section=4",
                title="Hardware y Sistemas Operativos",
                resource_type="course_section",
                parent_url="https://moodle.local/course/view.php?id=332",
                links=[
                    MoodleAuditLink(
                        label="Módulo 1 - Evaluación",
                        url="https://moodle.local/course/view.php?id=332&section=4",
                        final_url="https://moodle.local/course/view.php?id=332&section=4",
                        resource_type="course_section",
                    )
                ],
            ),
            MoodleAuditPage(
                page_kind="linked_resource",
                url="https://moodle.local/mod/quiz/view.php?id=12213",
                final_url="https://moodle.local/mod/quiz/view.php?id=12213",
                title="Hardware y Sistemas Operativos",
                subtitle="Cuestionario - Módulo 1 - Parte 1",
                resource_type="quiz",
                parent_url="https://moodle.local/course/view.php?id=332&section=4",
                links=[
                    MoodleAuditLink(
                        label="Cuestionario - Módulo 1 - Parte 1",
                        url="https://moodle.local/mod/quiz/view.php?id=12213",
                        final_url="https://moodle.local/mod/quiz/view.php?id=12213",
                        resource_type="quiz",
                    )
                ],
            ),
        ],
        warnings=[],
        assignments=[],
    )

    tree = build_moodle_audit_tree(snapshot)
    root = tree["root"]
    section_node = next(child for child in root["children"] if child["canonicalUrl"].endswith("section=4"))
    assert all(child["canonicalUrl"] != section_node["canonicalUrl"] for child in section_node["children"])

    quiz_node = next(child for child in section_node["children"] if child["kind"] == "quiz")
    assert all(child["canonicalUrl"] != quiz_node["canonicalUrl"] for child in quiz_node["children"])


def test_build_moodle_audit_tree_collapses_root_section_link_into_richer_section_page():
    from features.web_scraping.application.moodle_audit_tree import build_moodle_audit_tree
    from features.web_scraping.domain.moodle_audit_models import (
        MoodleAuditLink,
        MoodleAuditMeta,
        MoodleAuditPage,
        MoodleAuditSnapshot,
    )

    snapshot = MoodleAuditSnapshot(
        meta=MoodleAuditMeta(
            job_uid="moodle-job-root-section-collapse",
            base_url="https://moodle.local",
            stats={"retained_page_count": 2},
            resource_type_counts={"course_section": 1},
        ),
        pages=[
            MoodleAuditPage(
                page_kind="course_home",
                url="https://moodle.local/course/view.php?id=332",
                final_url="https://moodle.local/course/view.php?id=332",
                title="Hardware y Sistemas Operativos",
                resource_type="page",
                links=[
                    MoodleAuditLink(
                        label="Programa",
                        url="https://moodle.local/course/view.php?id=332&section=2",
                        final_url="https://moodle.local/course/view.php?id=332&section=2",
                        resource_type="course_section",
                    )
                ],
            ),
            MoodleAuditPage(
                page_kind="course_section",
                url="https://moodle.local/course/view.php?id=332&section=2",
                final_url="https://moodle.local/course/view.php?id=332&section=2",
                title="Hardware y Sistemas Operativos",
                source_link_label="Programa",
                description="Programa de la materia",
                resource_type="course_section",
                parent_url="https://moodle.local/course/view.php?id=332",
            ),
        ],
        warnings=[],
        assignments=[],
    )

    tree = build_moodle_audit_tree(snapshot)
    root = tree["root"]
    program_nodes = [child for child in root["children"] if child["canonicalUrl"].endswith("section=2")]

    assert len(program_nodes) == 1
    assert program_nodes[0]["kind"] == "section"
    assert program_nodes[0]["title"] == "Programa"
    assert program_nodes[0]["description"] == "Programa de la materia"


def test_build_moodle_audit_tree_collapses_placeholder_link_into_document_node():
    from features.web_scraping.application.moodle_audit_tree import build_moodle_audit_tree
    from features.web_scraping.domain.moodle_audit_models import (
        MoodleAuditAttachment,
        MoodleAuditLink,
        MoodleAuditMeta,
        MoodleAuditPage,
        MoodleAuditSnapshot,
    )

    snapshot = MoodleAuditSnapshot(
        meta=MoodleAuditMeta(
            job_uid="moodle-job-link-doc-collapse",
            base_url="https://moodle.local",
            stats={"retained_page_count": 2},
            resource_type_counts={"document": 1},
        ),
        pages=[
            MoodleAuditPage(
                page_kind="course_home",
                url="https://moodle.local/course/view.php?id=332",
                final_url="https://moodle.local/course/view.php?id=332",
                title="Hardware y Sistemas Operativos",
                resource_type="page",
            ),
            MoodleAuditPage(
                page_kind="course_section",
                url="https://moodle.local/course/view.php?id=332&section=3",
                final_url="https://moodle.local/course/view.php?id=332&section=3",
                title="Hardware y Sistemas Operativos",
                source_link_label="Módulo 1",
                resource_type="course_section",
                parent_url="https://moodle.local/course/view.php?id=332",
                links=[
                    MoodleAuditLink(
                        label="Contenido del Módulo 1 - Parte 2 Archivo",
                        url="https://moodle.local/mod/resource/view.php?id=9318",
                        final_url="https://moodle.local/mod/resource/view.php?id=9318",
                        resource_type="document",
                    )
                ],
                attachments=[
                    MoodleAuditAttachment(
                        label="Hardware y SO - Contenidos - Parte 2.pdf",
                        filename="Hardware y SO - Contenidos - Parte 2.pdf",
                        url="https://moodle.local/mod/resource/view.php?id=9318",
                        final_url="https://moodle.local/pluginfile.php/24872/mod_resource/content/1/Hardware.pdf",
                        content_type="application/pdf",
                        is_download=True,
                    )
                ],
            ),
        ],
        warnings=[],
        assignments=[],
    )

    tree = build_moodle_audit_tree(snapshot)
    root = tree["root"]
    section_node = next(child for child in root["children"] if child["title"] == "Módulo 1")
    matching_docs = [child for child in section_node["children"] if child["kind"] == "document"]

    assert len(matching_docs) == 1
    assert matching_docs[0]["title"] == "Hardware y SO - Contenidos - Parte 2.pdf"


def test_build_moodle_audit_tree_removes_quiz_language_variants_and_uses_subtitle_as_title():
    from features.web_scraping.application.moodle_audit_tree import build_moodle_audit_tree
    from features.web_scraping.domain.moodle_audit_models import (
        MoodleAuditLink,
        MoodleAuditMeta,
        MoodleAuditPage,
        MoodleAuditSnapshot,
    )

    snapshot = MoodleAuditSnapshot(
        meta=MoodleAuditMeta(
            job_uid="moodle-job-quiz-title",
            base_url="https://moodle.local",
            stats={"retained_page_count": 2},
            resource_type_counts={"quiz": 1},
        ),
        pages=[
            MoodleAuditPage(
                page_kind="course_home",
                url="https://moodle.local/course/view.php?id=332",
                final_url="https://moodle.local/course/view.php?id=332",
                title="Hardware y Sistemas Operativos",
                resource_type="page",
            ),
            MoodleAuditPage(
                page_kind="linked_resource",
                url="https://moodle.local/mod/quiz/view.php?id=12213",
                final_url="https://moodle.local/mod/quiz/view.php?id=12213",
                title="Hardware y Sistemas Operativos",
                subtitle="Cuestionario - Módulo 1 - Parte 1",
                resource_type="quiz",
                parent_url="https://moodle.local/course/view.php?id=332",
                links=[
                    MoodleAuditLink(
                        label="English ‎(en)‎",
                        url="https://moodle.local/mod/quiz/view.php?id=12213&lang=en",
                        final_url="https://moodle.local/mod/quiz/view.php?id=12213&lang=en",
                        resource_type="quiz",
                    ),
                    MoodleAuditLink(
                        label="Español - Internacional ‎(es)‎",
                        url="https://moodle.local/mod/quiz/view.php?id=12213&lang=es",
                        final_url="https://moodle.local/mod/quiz/view.php?id=12213&lang=es",
                        resource_type="quiz",
                    ),
                ],
            ),
        ],
        warnings=[],
        assignments=[],
    )

    tree = build_moodle_audit_tree(snapshot)
    root = tree["root"]
    quiz_node = next(child for child in root["children"] if child["kind"] == "quiz")

    assert quiz_node["title"] == "Cuestionario - Módulo 1 - Parte 1"
    assert quiz_node["children"] == []


def test_build_moodle_audit_tree_hides_quiz_review_child_and_fake_drive_image():
    from features.web_scraping.application.moodle_audit_tree import build_moodle_audit_tree
    from features.web_scraping.domain.moodle_audit_models import (
        MoodleAuditExternalResource,
        MoodleAuditImage,
        MoodleAuditLink,
        MoodleAuditMeta,
        MoodleAuditPage,
        MoodleAuditSnapshot,
    )

    snapshot = MoodleAuditSnapshot(
        meta=MoodleAuditMeta(
            job_uid="moodle-job-quiz-review-drive-image",
            base_url="https://moodle.local",
            stats={"retained_page_count": 2},
            resource_type_counts={"quiz": 1, "google_drive": 1},
        ),
        pages=[
            MoodleAuditPage(
                page_kind="course_home",
                url="https://moodle.local/course/view.php?id=332",
                final_url="https://moodle.local/course/view.php?id=332",
                title="Hardware y Sistemas Operativos",
                resource_type="page",
            ),
            MoodleAuditPage(
                page_kind="linked_resource",
                url="https://moodle.local/mod/quiz/view.php?id=12213",
                final_url="https://moodle.local/mod/quiz/view.php?id=12213",
                title="Hardware y Sistemas Operativos",
                subtitle="Cuestionario - Módulo 1 - Parte 1",
                resource_type="quiz",
                parent_url="https://moodle.local/course/view.php?id=332",
                links=[
                    MoodleAuditLink(
                        label="Revisión",
                        url="https://moodle.local/mod/quiz/review.php?attempt=3313&cmid=12213",
                        final_url="https://moodle.local/mod/quiz/review.php?attempt=3313&cmid=12213",
                        resource_type="quiz",
                    )
                ],
            ),
            MoodleAuditPage(
                page_kind="linked_resource",
                url="https://moodle.local/mod/url/view.php?id=12120",
                final_url="https://drive.google.com/file/d/14CD0RrOoXd7tx3iLOwEhwIkGJtFwcMEH/view",
                title="Proceso de fabricación de un microchip.mp4 - Google Drive",
                resource_type="google_drive",
                parent_url="https://moodle.local/course/view.php?id=332",
                external_resource=MoodleAuditExternalResource(
                    provider="google_drive",
                    resource_id="14CD0RrOoXd7tx3iLOwEhwIkGJtFwcMEH",
                    resource_type="video",
                    canonical_url="https://drive.google.com/file/d/14CD0RrOoXd7tx3iLOwEhwIkGJtFwcMEH/view",
                    preview_url="https://drive.google.com/drive-viewer/preview.png",
                    access_url="https://accounts.google.com/ServiceLogin?continue=drive",
                    download_url="https://drive.google.com/uc?export=download&id=14CD0RrOoXd7tx3iLOwEhwIkGJtFwcMEH",
                ),
                images=[
                    MoodleAuditImage(
                        label="real preview",
                        url="https://drive.google.com/drive-viewer/preview.png",
                    ),
                    MoodleAuditImage(
                        label="https://drive.google.com/file/d/14CD0RrOoXd7tx3iLOwEhwIkGJtFwcMEH/view/",
                        url="https://drive.google.com/file/d/14CD0RrOoXd7tx3iLOwEhwIkGJtFwcMEH/view/",
                    ),
                ],
            ),
        ],
        warnings=[],
        assignments=[],
    )

    tree = build_moodle_audit_tree(snapshot)
    root = tree["root"]
    quiz_node = next(child for child in root["children"] if child["kind"] == "quiz")
    assert quiz_node["children"] == []

    drive_node = next(child for child in root["children"] if child["kind"] == "google_drive")
    drive_child_urls = [child["canonicalUrl"] for child in drive_node["children"] if child["kind"] == "image"]
    assert "https://drive.google.com/file/d/14CD0RrOoXd7tx3iLOwEhwIkGJtFwcMEH/view/" not in drive_child_urls
    assert "https://drive.google.com/drive-viewer/preview.png" not in drive_child_urls


def test_build_moodle_audit_tree_normalizes_raw_image_titles_to_preview():
    from features.web_scraping.application.moodle_audit_tree import build_moodle_audit_tree
    from features.web_scraping.domain.moodle_audit_models import (
        MoodleAuditImage,
        MoodleAuditMeta,
        MoodleAuditPage,
        MoodleAuditSnapshot,
    )

    snapshot = MoodleAuditSnapshot(
        meta=MoodleAuditMeta(
            job_uid="moodle-job-image-title",
            base_url="https://moodle.local",
            stats={"retained_page_count": 1},
            resource_type_counts={"image": 1},
        ),
        pages=[
            MoodleAuditPage(
                page_kind="course_home",
                url="https://moodle.local/course/view.php?id=332",
                final_url="https://moodle.local/course/view.php?id=332",
                title="Hardware y Sistemas Operativos",
                resource_type="page",
                images=[
                    MoodleAuditImage(
                        label="https://drive.google.com/drive-viewer/preview.png",
                        url="https://drive.google.com/drive-viewer/preview.png",
                    )
                ],
            )
        ],
        warnings=[],
        assignments=[],
    )

    tree = build_moodle_audit_tree(snapshot)
    root = tree["root"]
    image_node = next(child for child in root["children"] if child["kind"] == "image")
    assert image_node["title"] == "Preview"


def test_collect_followable_child_links_keeps_same_course_sections_and_folder_resources_only():
    from features.web_scraping.infrastructure.scraping_tools import _collect_followable_child_links

    page_payload = {
        "url": "https://moodle.local/course/view.php?id=332",
        "final_url": "https://moodle.local/course/view.php?id=332",
        "links": [
            {
                "label": "General",
                "url": "https://moodle.local/course/view.php?id=332&section=0",
                "resource_type": "course_section",
            },
            {
                "label": "Programa",
                "url": "https://moodle.local/mod/folder/view.php?id=8010",
                "resource_type": "folder",
            },
            {
                "label": "Programación I",
                "url": "https://moodle.local/course/view.php?id=328",
                "resource_type": "link",
            },
        ],
    }

    followable = _collect_followable_child_links(page_payload, root_host="moodle.local", limit=10)
    urls = [item["url"] for item in followable]

    assert "https://moodle.local/course/view.php?id=332&section=0" in urls
    assert "https://moodle.local/mod/folder/view.php?id=8010" in urls
    assert "https://moodle.local/course/view.php?id=328" not in urls


def test_infer_child_page_kind_marks_course_sections_explicitly():
    from features.web_scraping.infrastructure.scraping_tools import _infer_child_page_kind

    assert (
        _infer_child_page_kind(
            "course_home",
            "https://moodle.local/course/view.php?id=332&section=3",
            "Módulo 1",
        )
        == "course_section"
    )
