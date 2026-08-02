from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_linkedin_browser_launch_config_defaults_to_bundled_chromium(monkeypatch):
    from features.web_scraping.infrastructure.authenticated_browser import (
        AuthenticatedBrowserLaunchConfig,
    )

    monkeypatch.delenv("LINKEDIN_BROWSER", raising=False)
    monkeypatch.delenv("LINKEDIN_BROWSER_EXECUTABLE_PATH", raising=False)

    config = AuthenticatedBrowserLaunchConfig.from_env()

    assert config.browser == "chromium"
    assert config.uses_installed_browser is False
    assert config.launch_kwargs(headless=False) == {"headless": False}


@pytest.mark.parametrize(
    ("requested", "expected_channel"),
    [("chrome", "chrome"), ("edge", "msedge"), ("msedge", "msedge")],
)
def test_linkedin_browser_launch_config_uses_installed_channels(
    requested,
    expected_channel,
):
    from features.web_scraping.infrastructure.authenticated_browser import (
        AuthenticatedBrowserLaunchConfig,
    )

    config = AuthenticatedBrowserLaunchConfig.from_env(browser=requested)

    assert config.uses_installed_browser is True
    assert config.launch_kwargs(headless=True) == {
        "headless": True,
        "channel": expected_channel,
    }


def test_linkedin_browser_launch_config_requires_explicit_brave_path(tmp_path):
    from features.web_scraping.infrastructure.authenticated_browser import (
        AuthenticatedBrowserLaunchConfig,
    )

    with pytest.raises(ValueError, match="requiere"):
        AuthenticatedBrowserLaunchConfig.from_env(browser="brave")

    executable = tmp_path / "brave"
    executable.write_text("", encoding="utf-8")
    executable.chmod(0o700)
    config = AuthenticatedBrowserLaunchConfig.from_env(
        browser="brave",
        executable_path=executable,
    )

    assert config.launch_kwargs(headless=False) == {
        "headless": False,
        "executable_path": str(executable),
    }


def test_linkedin_browser_launch_config_does_not_silently_fallback(tmp_path):
    from features.web_scraping.infrastructure.authenticated_browser import (
        AuthenticatedBrowserLaunchConfig,
    )

    with pytest.raises(ValueError, match="no apunta"):
        AuthenticatedBrowserLaunchConfig.from_env(
            browser="executable",
            executable_path=tmp_path / "missing-browser",
        )


def test_linkedin_browser_config_precedence_env_then_metadata_then_default(
    tmp_path,
    monkeypatch,
):
    from features.web_scraping.infrastructure.authenticated_browser import (
        AuthenticatedBrowserLaunchConfig,
    )

    brave = tmp_path / "brave"
    brave.write_text("", encoding="utf-8")
    brave.chmod(0o700)
    monkeypatch.delenv("LINKEDIN_BROWSER", raising=False)
    monkeypatch.delenv("LINKEDIN_BROWSER_EXECUTABLE_PATH", raising=False)

    persisted = AuthenticatedBrowserLaunchConfig.from_env(
        persisted_browser="brave",
        persisted_executable_path=brave,
    )
    assert persisted.browser == "brave"
    assert persisted.executable_path == brave
    assert persisted.source == "session_metadata"

    monkeypatch.setenv("LINKEDIN_BROWSER", "chrome")
    from_env = AuthenticatedBrowserLaunchConfig.from_env(
        persisted_browser="brave",
        persisted_executable_path=brave,
    )
    assert from_env.browser == "chrome"
    assert from_env.executable_path is None
    assert from_env.source == "env"

    explicit = AuthenticatedBrowserLaunchConfig.from_env(
        browser="edge",
        persisted_browser="brave",
        persisted_executable_path=brave,
    )
    assert explicit.browser == "msedge"
    assert explicit.source == "cli"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, False),
        ("false", False),
        ("0", False),
        ("true", True),
        ("1", True),
    ],
)
def test_linkedin_headless_config_parses_strict_values(monkeypatch, raw, expected):
    from features.web_scraping.infrastructure.authenticated_browser import (
        configured_linkedin_headless,
    )

    if raw is None:
        monkeypatch.delenv("LINKEDIN_HEADLESS", raising=False)
    else:
        monkeypatch.setenv("LINKEDIN_HEADLESS", raw)

    assert configured_linkedin_headless() is expected

    monkeypatch.setenv("LINKEDIN_HEADLESS", "sometimes")
    with pytest.raises(ValueError, match="LINKEDIN_HEADLESS"):
        configured_linkedin_headless()


def test_linkedin_persistent_context_uses_dedicated_profile_and_closes_lock(
    tmp_path,
):
    from features.web_scraping.infrastructure.authenticated_browser import (
        AuthenticatedBrowserLaunchConfig,
        BrowserProfileLock,
        BrowserProfileInUseError,
        open_persistent_authenticated_context,
    )

    profile_path = tmp_path / "private" / "browser-profile"
    profile_path.mkdir(parents=True)
    profile_path.chmod(0o700)
    page = SimpleNamespace(url="about:blank")
    context = MagicMock()
    context.pages = [page]
    chromium = MagicMock()
    chromium.launch_persistent_context.return_value = context
    playwright = SimpleNamespace(chromium=chromium)

    session = open_persistent_authenticated_context(
        playwright=playwright,
        profile_path=profile_path,
        headless=False,
        launch_config=AuthenticatedBrowserLaunchConfig(browser="chromium"),
    )
    competing_lock = BrowserProfileLock(profile_path)
    with pytest.raises(BrowserProfileInUseError, match="ya está en uso"):
        competing_lock.acquire()

    session.close()

    chromium.launch_persistent_context.assert_called_once_with(
        str(profile_path),
        headless=False,
        viewport={"width": 1366, "height": 900},
        locale="es-AR",
    )
    context.close.assert_called_once()
    competing_lock.acquire()
    competing_lock.release()


def test_linkedin_persistent_session_does_not_assume_separate_browser():
    from features.web_scraping.infrastructure.authenticated_browser import (
        AuthenticatedBrowserSession,
    )

    context = MagicMock()
    lock = MagicMock()
    session = AuthenticatedBrowserSession(
        context=context,
        page=SimpleNamespace(),
        profile_lock=lock,
    )

    session.close()

    context.close.assert_called_once()
    lock.release.assert_called_once()


def test_linkedin_persistent_session_replaces_only_page_in_same_context():
    from features.web_scraping.infrastructure.authenticated_browser import (
        AuthenticatedBrowserSession,
    )

    previous_page = MagicMock()
    replacement_page = MagicMock()
    context = MagicMock()
    context.new_page.return_value = replacement_page
    session = AuthenticatedBrowserSession(
        context=context,
        page=previous_page,
    )

    returned = session.replace_page()

    assert returned is replacement_page
    assert session.page is replacement_page
    context.new_page.assert_called_once()
    previous_page.close.assert_called_once()
    context.close.assert_not_called()


def test_linkedin_job_page_scope_closes_every_page_created_by_job():
    from features.web_scraping.infrastructure.authenticated_browser import (
        AuthenticatedBrowserSession,
    )

    preexisting_page = MagicMock()
    first_job_page = MagicMock()
    second_job_page = MagicMock()
    context = MagicMock()
    context.pages = [preexisting_page]
    context.new_page.side_effect = [first_job_page, second_job_page]
    browser = MagicMock()
    session = AuthenticatedBrowserSession(
        context=context,
        page=preexisting_page,
        browser=browser,
    )

    scope = session.begin_job_pages()
    session.replace_page()
    session.replace_page()
    session.close_job_pages(scope)

    first_job_page.close.assert_called_once()
    second_job_page.close.assert_called_once()


def test_linkedin_job_page_scope_preserves_preexisting_pages():
    from features.web_scraping.infrastructure.authenticated_browser import (
        AuthenticatedBrowserSession,
    )

    preexisting_page = MagicMock()
    job_page = MagicMock()
    context = MagicMock()
    context.pages = [preexisting_page]
    context.new_page.return_value = job_page
    session = AuthenticatedBrowserSession(
        context=context,
        page=preexisting_page,
    )

    scope = session.begin_job_pages()
    session.replace_page()
    session.close_job_pages(scope)

    preexisting_page.close.assert_not_called()
    job_page.close.assert_called_once()


def test_linkedin_job_page_scope_cleanup_runs_after_exception():
    from features.web_scraping.infrastructure.authenticated_browser import (
        AuthenticatedBrowserSession,
    )

    preexisting_page = MagicMock()
    job_page = MagicMock()
    context = MagicMock()
    context.pages = [preexisting_page]
    context.new_page.return_value = job_page
    session = AuthenticatedBrowserSession(
        context=context,
        page=preexisting_page,
    )

    scope = session.begin_job_pages()
    with pytest.raises(RuntimeError, match="job failed"):
        try:
            session.replace_page()
            raise RuntimeError("job failed")
        finally:
            session.close_job_pages(scope)

    job_page.close.assert_called_once()
    preexisting_page.close.assert_not_called()


def test_linkedin_job_page_scope_cleanup_runs_after_cancellation():
    from features.web_scraping.infrastructure.authenticated_browser import (
        AuthenticatedBrowserSession,
    )

    preexisting_page = MagicMock()
    job_page = MagicMock()
    context = MagicMock()
    context.pages = [preexisting_page]
    context.new_page.return_value = job_page
    session = AuthenticatedBrowserSession(
        context=context,
        page=preexisting_page,
    )

    scope = session.begin_job_pages()
    with pytest.raises(KeyboardInterrupt):
        try:
            session.new_job_page()
            raise KeyboardInterrupt()
        finally:
            session.close_job_pages(scope)

    job_page.close.assert_called_once()
    preexisting_page.close.assert_not_called()
    context.close.assert_not_called()


def test_linkedin_job_page_cleanup_keeps_context_and_browser_open():
    from features.web_scraping.infrastructure.authenticated_browser import (
        AuthenticatedBrowserSession,
    )

    preexisting_page = MagicMock()
    job_page = MagicMock()
    context = MagicMock()
    context.pages = [preexisting_page]
    context.new_page.return_value = job_page
    browser = MagicMock()
    browser.is_connected.return_value = True
    session = AuthenticatedBrowserSession(
        context=context,
        page=preexisting_page,
        browser=browser,
    )

    scope = session.begin_job_pages()
    session.replace_page()
    session.close_job_pages(scope)

    context.close.assert_not_called()
    browser.close.assert_not_called()
    assert session.context_is_alive() is True


def test_linkedin_reusable_context_survives_job_page_cleanup(tmp_path):
    from features.web_scraping.infrastructure.authenticated_browser import (
        AuthenticatedBrowserLaunchConfig,
        open_persistent_authenticated_context,
    )

    profile_path = tmp_path / "private" / "browser-profile"
    profile_path.mkdir(parents=True)
    preexisting_page = MagicMock()
    preexisting_page.url = "about:blank"
    job_page = MagicMock()
    job_page.url = "about:blank"
    context = MagicMock()
    context.pages = [preexisting_page]
    context.new_page.return_value = job_page
    chromium = MagicMock()
    chromium.launch_persistent_context.return_value = context
    playwright = SimpleNamespace(chromium=chromium)

    session = open_persistent_authenticated_context(
        playwright=playwright,
        profile_path=profile_path,
        headless=False,
        launch_config=AuthenticatedBrowserLaunchConfig(browser="chromium"),
        reuse=True,
    )
    scope = session.begin_job_pages()
    session.new_job_page()
    session.close_job_pages(scope)

    job_page.close.assert_called_once()
    preexisting_page.close.assert_not_called()
    context.close.assert_not_called()

    reused = open_persistent_authenticated_context(
        playwright=playwright,
        profile_path=profile_path,
        headless=False,
        launch_config=AuthenticatedBrowserLaunchConfig(browser="chromium"),
        reuse=True,
    )
    assert reused is session
    chromium.launch_persistent_context.assert_called_once()
    session.close()


def test_linkedin_persistent_session_keeps_previous_page_if_replacement_unusable():
    from features.web_scraping.infrastructure.authenticated_browser import (
        AuthenticatedBrowserSession,
    )

    previous_page = MagicMock()
    previous_page.is_closed.return_value = False
    replacement_page = MagicMock()
    replacement_page.is_closed.return_value = True
    context = MagicMock()
    context.pages = [previous_page]
    context.new_page.return_value = replacement_page
    session = AuthenticatedBrowserSession(
        context=context,
        page=previous_page,
    )

    with pytest.raises(RuntimeError, match="replacement page closed"):
        session.replace_page()

    assert session.page is previous_page
    previous_page.close.assert_not_called()
    replacement_page.close.assert_called_once()
    context.close.assert_not_called()


def test_linkedin_persistent_session_releases_lock_when_close_fails():
    from features.web_scraping.infrastructure.authenticated_browser import (
        AuthenticatedBrowserSession,
    )

    context = MagicMock()
    context.close.side_effect = RuntimeError("close failed")
    lock = MagicMock()
    session = AuthenticatedBrowserSession(
        context=context,
        page=SimpleNamespace(),
        profile_lock=lock,
    )

    with pytest.raises(RuntimeError, match="close failed"):
        session.close()

    lock.release.assert_called_once()


def test_linkedin_profile_path_rejects_personal_browser_profiles(
    tmp_path,
    monkeypatch,
):
    from features.web_scraping.infrastructure.linkedin_session_store import (
        LinkedInSessionStore,
    )

    store = LinkedInSessionStore(
        tmp_path / "project" / "data" / "private" / "linkedin" / "default" / "storage_state.json"
    )
    personal_profile = (
        tmp_path
        / "Library"
        / "Application Support"
        / "BraveSoftware"
        / "Brave-Browser"
        / "Default"
    )
    personal_profile.mkdir(parents=True)
    monkeypatch.setenv("LINKEDIN_PROFILE_DIR", str(personal_profile))

    with pytest.raises(ValueError, match="perfil personal"):
        store.resolve_profile_path(create=False)


def test_linkedin_profile_named_default_is_allowed_only_in_managed_private_area(
    tmp_path,
):
    from features.web_scraping.infrastructure.linkedin_session_store import (
        LinkedInSessionStore,
    )

    store = LinkedInSessionStore(
        tmp_path
        / "data"
        / "private"
        / "linkedin"
        / "bootstrap"
        / "storage_state.json"
    )
    managed_profile = store.path.parent / "Default"

    assert store.resolve_profile_path(
        profile_dir=managed_profile,
        create=True,
    ) == managed_profile.resolve()


def test_linkedin_profile_precedence_env_then_metadata_then_default(
    tmp_path,
    monkeypatch,
):
    from features.web_scraping.infrastructure.linkedin_session_store import (
        LinkedInSessionStore,
    )

    store = LinkedInSessionStore(
        tmp_path / "private" / "linkedin" / "default" / "storage_state.json"
    )
    monkeypatch.delenv("LINKEDIN_PROFILE_DIR", raising=False)
    default_profile = store.resolve_profile_path(create=True)
    assert default_profile == store.default_profile_path.resolve()

    metadata_profile = store.path.parent / "metadata-profile"
    metadata_profile.mkdir()
    metadata_profile.chmod(0o700)
    assert store.resolve_profile_path(
        persisted_profile_path=metadata_profile
    ) == metadata_profile.resolve()

    env_profile = store.path.parent / "env-profile"
    env_profile.mkdir()
    env_profile.chmod(0o700)
    monkeypatch.setenv("LINKEDIN_PROFILE_DIR", str(env_profile))
    assert store.resolve_profile_path(
        persisted_profile_path=metadata_profile
    ) == env_profile.resolve()

    cli_profile = store.path.parent / "cli-profile"
    cli_profile.mkdir()
    cli_profile.chmod(0o700)
    assert store.resolve_profile_path(
        profile_dir=cli_profile,
        persisted_profile_path=metadata_profile,
    ) == cli_profile.resolve()


def test_linkedin_bootstrap_detects_google_automation_block_only_on_google():
    from scripts.bootstrap_linkedin_session import is_google_oauth_automation_block

    message = "No puedes acceder. Es posible que no sean seguros este navegador o la app."
    assert is_google_oauth_automation_block(
        "https://accounts.google.com/signin/oauth/error",
        message,
    )
    assert not is_google_oauth_automation_block(
        "https://www.linkedin.com/login",
        message,
    )
    assert not is_google_oauth_automation_block(
        "https://accounts.google.com/signin/oauth/error",
        "Elegí una cuenta",
    )


def test_linkedin_bootstrap_detects_google_block_in_popup_context():
    from scripts.bootstrap_linkedin_session import (
        detect_google_oauth_automation_block_in_context,
    )

    class FakeLocator:
        def __init__(self, text):
            self.text = text

        def inner_text(self, *, timeout):
            assert timeout == 3000
            return self.text

    class FakePage:
        def __init__(self, url, text):
            self.url = url
            self.text = text

        def locator(self, selector):
            assert selector == "body"
            return FakeLocator(self.text)

    context = SimpleNamespace(
        pages=[
            FakePage("https://www.linkedin.com/login", "Iniciar sesión"),
            FakePage(
                "https://accounts.google.com/signin/oauth/error",
                "This browser or app may not be secure.",
            ),
        ]
    )

    assert detect_google_oauth_automation_block_in_context(context)


def test_linkedin_intent_requires_jobs_and_target_topic():
    from features.web_scraping.application.linkedin_intent import (
        detect_linkedin_jobs_intent,
    )

    assert detect_linkedin_jobs_intent(
        "Buscá vacantes LinkedIn de Machine Learning publicadas hoy"
    )
    assert detect_linkedin_jobs_intent(
        "Quiero empleos de Data Science de las últimas 24 horas"
    )
    assert detect_linkedin_jobs_intent(
        "Buscá vacantes LinkedIn para AI Agent Engineer publicadas hoy"
    )
    assert detect_linkedin_jobs_intent(
        "Buscá trabajos de AI Architect de las últimas 24 horas"
    )
    assert not detect_linkedin_jobs_intent("Mostrame publicaciones de LinkedIn")
    assert not detect_linkedin_jobs_intent("Buscá empleos de contador")


def test_linkedin_location_extraction_is_multi_value_and_alias_driven():
    from features.web_scraping.application.linkedin_intent import (
        extract_linkedin_locations,
    )

    assert extract_linkedin_locations(
        "Buscá vacantes en Corea del Sur y Japón"
    ) == ("South Korea", "Japan")
    assert extract_linkedin_locations(
        "Find LinkedIn AI jobs in Japan and South Korea"
    ) == ("South Korea",)
    assert extract_linkedin_locations("Buscá vacantes AI en Argentina") == ()


def test_linkedin_url_policy_is_jobs_only_and_canonicalizes_tracking():
    from features.web_scraping.infrastructure.linkedin_url_policy import (
        canonicalize_linkedin_job_url,
        is_linkedin_auth_checkpoint,
        validate_linkedin_jobs_url,
    )

    accepted = validate_linkedin_jobs_url(
        "https://www.linkedin.com/jobs/search/?keywords=AI&f_TPR=r86400"
        "&location=Japan&geoId=101355337"
    )
    assert accepted.startswith("https://www.linkedin.com/jobs/search/")
    assert canonicalize_linkedin_job_url(
        "https://linkedin.com/jobs/view/ai-engineer-123?trackingId=sensitive&trk=foo"
    ) == "https://www.linkedin.com/jobs/view/ai-engineer-123"
    assert is_linkedin_auth_checkpoint("https://www.linkedin.com/checkpoint/challenge/") is True
    assert is_linkedin_auth_checkpoint("https://www.linkedin.com/login") is True

    with pytest.raises(ValueError, match="host_not_allowed"):
        validate_linkedin_jobs_url("https://evil.example/jobs/view/123")
    with pytest.raises(ValueError, match="scheme_not_allowed"):
        validate_linkedin_jobs_url("http://www.linkedin.com/jobs/view/123")
    with pytest.raises(ValueError, match="path_not_allowed"):
        validate_linkedin_jobs_url("https://www.linkedin.com/in/example")
    with pytest.raises(ValueError, match="query_parameter_not_allowed"):
        validate_linkedin_jobs_url(
            "https://www.linkedin.com/jobs/search/?keywords=AI&token=secret"
        )
    assert is_linkedin_auth_checkpoint("https://evil.example/login") is False


def test_linkedin_session_store_writes_atomically_with_private_permissions(tmp_path):
    from features.web_scraping.infrastructure.authenticated_browser import (
        AuthenticatedBrowserLaunchConfig,
    )
    from features.web_scraping.infrastructure.linkedin_session_store import LinkedInSessionStore

    state_path = tmp_path / "private" / "storage_state.json"
    store = LinkedInSessionStore(state_path)
    executable = tmp_path / "private" / "brave"
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.write_text("", encoding="utf-8")
    executable.chmod(0o700)
    launch_config = AuthenticatedBrowserLaunchConfig.from_env(
        browser="brave",
        executable_path=executable,
    )

    class FakeContext:
        def storage_state(self, *, path: str) -> None:
            Path(path).write_text(
                json.dumps({"cookies": [], "origins": []}),
                encoding="utf-8",
            )

    profile_path = store.resolve_profile_path(create=True)
    store.save_from_context(
        FakeContext(),
        launch_config=launch_config,
        profile_path=profile_path,
    )
    store.validate()
    metadata = store.load_browser_metadata()

    assert state_path.exists()
    assert state_path.stat().st_mode & 0o777 == 0o600
    assert state_path.parent.stat().st_mode & 0o777 == 0o700
    assert metadata == {
        "browser": "brave",
        "executable_path": str(executable),
        "profile_mode": "persistent",
        "profile_path": str(profile_path),
    }
    assert store.browser_metadata_path.stat().st_mode & 0o777 == 0o600
    assert not list(state_path.parent.glob("*.tmp"))

    with pytest.raises(ValueError, match="confirmación explícita"):
        store.invalidate("LinkedInAuthRequiredError")
    assert state_path.exists()

    store.invalidate("corrupt_state")
    assert not state_path.exists()
    assert not store.browser_metadata_path.exists()
    assert not list(state_path.parent.glob("*.invalid-*.json"))
    reason_files = list(state_path.parent.glob("*.invalid-*.reason"))
    assert len(reason_files) == 1
    assert reason_files[0].read_text(encoding="utf-8") == "corrupt_state"
    assert reason_files[0].stat().st_mode & 0o777 == 0o600


def test_linkedin_session_store_rejects_permissive_state_file(tmp_path):
    from features.web_scraping.infrastructure.linkedin_session_store import LinkedInSessionStore

    state_path = tmp_path / "storage_state.json"
    state_path.write_text(
        json.dumps({"cookies": [], "origins": []}),
        encoding="utf-8",
    )
    state_path.chmod(0o644)

    with pytest.raises(ValueError, match="permisos inseguros"):
        LinkedInSessionStore(state_path).validate()


@pytest.mark.parametrize(
    ("text", "expected_within"),
    [
        ("a few seconds ago", True),
        ("hace unos segundos", True),
        ("15 minutes ago", True),
        ("hace 4 horas", True),
        ("23 hours ago", True),
        ("hace 25 horas", False),
        ("1 day ago", True),
        ("hace 2 días", False),
        ("fecha desconocida", False),
    ],
)
def test_relative_time_parser_filters_last_24_hours(text, expected_within):
    from features.web_scraping.infrastructure.linkedin_scraper import (
        parse_linkedin_relative_time,
    )

    published, confidence, within = parse_linkedin_relative_time(
        text,
        now=datetime(2026, 7, 28, 18, 0, tzinfo=timezone.utc),
    )
    assert within is expected_within
    if expected_within:
        assert published is not None
        assert confidence in {"medium", "high"}


def test_linkedin_html_parser_and_dedupe_use_local_fixture():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        dedupe_linkedin_vacancies,
        parse_linkedin_jobs_html,
    )

    html = Path("tests/fixtures/linkedin_jobs_search.html").read_text(encoding="utf-8")
    records = parse_linkedin_jobs_html(
        html,
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI&f_TPR=r86400",
        now=datetime(2026, 7, 28, 14, 0, tzinfo=timezone.utc),
    )

    assert len(records) == 2
    assert records[0].linkedin_job_id == "111"
    assert records[0].is_within_24_hours is True
    assert records[0].workplace_type == "remote"
    assert "trackingId" not in records[0].canonical_url
    assert records[1].is_within_24_hours is False
    assert len(dedupe_linkedin_vacancies([records[0], records[0]])) == 1

    with pytest.raises(ValueError, match="host_not_allowed"):
        parse_linkedin_jobs_html(
            html,
            source_url="https://evil.example/jobs/search/",
            now=datetime(2026, 7, 28, 14, 0, tzinfo=timezone.utc),
        )


def test_linkedin_parser_handles_authenticated_dom_relative_hrefs_and_wrappers():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _parse_linkedin_jobs_html_with_diagnostics,
    )

    html = """
    <main>
      <ul>
        <li data-occludable-job-id="111">
          <div class="job-card-container" data-job-id="111">
            <a class="job-card-list__title--link"
               href="/jobs/view/ai-engineer-111/?trackingId=secret">
              <span class="job-card-list__title">AI Engineer AI Engineer</span>
            </a>
            <div class="job-card-container__primary-description">Example AI</div>
            <span class="job-card-container__metadata-item">Tokyo · Remote</span>
            <time datetime="2026-07-28T12:00:00Z">2 hours ago</time>
          </div>
        </li>
        <li class="scaffold-layout__list-item">
          <div data-job-id="222">
            <a href="/jobs/view/ml-engineer-222/">
              <span class="job-card-list__title">Machine Learning Engineer</span>
            </a>
            <time datetime="2026-07-28T11:00:00Z">3 hours ago</time>
          </div>
        </li>
      </ul>
    </main>
    """

    records, diagnostics = _parse_linkedin_jobs_html_with_diagnostics(
        html,
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI&f_TPR=r86400",
        now=datetime(2026, 7, 28, 14, 0, tzinfo=timezone.utc),
    )

    assert [record.linkedin_job_id for record in records] == ["111", "222"]
    assert records[0].title == "AI Engineer"
    assert records[0].canonical_url == "https://www.linkedin.com/jobs/view/ai-engineer-111"
    assert diagnostics.selector_counts[".job-card-container"] == 1
    assert diagnostics.selector_counts["li[data-occludable-job-id]"] == 1
    assert diagnostics.selector_counts[".scaffold-layout__list-item"] == 1
    assert diagnostics.selector_counts["[data-job-id]"] == 2
    assert diagnostics.href_count == 2
    assert diagnostics.parseable_candidate_count == 2
    assert diagnostics.discard_reasons["duplicate_wrapper"] >= 2


def test_linkedin_search_hydration_waits_progressively_for_late_cards():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _wait_for_search_results_hydration,
    )

    state = {"polls": 0}

    class FakeLocator:
        def __init__(self, selector: str) -> None:
            self.selector = selector

        @property
        def first(self):
            return self

        def count(self) -> int:
            return int(
                self.selector == ".job-card-container"
                and state["polls"] >= 2
            )

        def is_visible(self, timeout: int) -> bool:
            return bool(self.count())

    class FakePage:
        url = "https://www.linkedin.com/jobs/search/?keywords=AI"

        def __init__(self) -> None:
            self.waits = []

        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator(selector)

        def wait_for_timeout(self, milliseconds: int) -> None:
            self.waits.append(milliseconds)
            state["polls"] += 1

    page = FakePage()
    result = _wait_for_search_results_hydration(page, max_wait_ms=1000)

    assert result == "results"
    assert page.waits == [100, 150]
    assert 1200 not in page.waits


def test_linkedin_search_hydration_distinguishes_empty_and_indeterminate_timeout():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _wait_for_search_results_hydration,
    )

    class FakeLocator:
        def __init__(self, active: bool) -> None:
            self.active = active

        @property
        def first(self):
            return self

        def count(self) -> int:
            return int(self.active)

        def is_visible(self, timeout: int) -> bool:
            return self.active

    class FakePage:
        url = "https://www.linkedin.com/jobs/search/?keywords=AI"

        def __init__(self, empty: bool) -> None:
            self.empty = empty
            self.waits = []

        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator(
                self.empty and selector == ".jobs-search-no-results-banner"
            )

        def wait_for_timeout(self, milliseconds: int) -> None:
            self.waits.append(milliseconds)

    empty_page = FakePage(empty=True)
    assert _wait_for_search_results_hydration(
        empty_page,
        max_wait_ms=450,
    ) == "empty"
    assert empty_page.waits == []

    indeterminate_page = FakePage(empty=False)
    assert _wait_for_search_results_hydration(
        indeterminate_page,
        max_wait_ms=450,
    ) == "timeout"
    assert indeterminate_page.waits == [100, 150, 200]


def test_linkedin_detail_hydration_waits_for_late_parseable_date():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _wait_for_detail_hydration,
    )

    state = {"polls": 0}

    class FakeLocator:
        def __init__(self, selector: str) -> None:
            self.selector = selector

        @property
        def first(self):
            return self

        def count(self) -> int:
            if self.selector.endswith("job-title h1"):
                return 1
            return int(self.selector == "time" and state["polls"] >= 2)

        def is_visible(self, timeout: int) -> bool:
            return bool(self.count())

        def inner_text(self, timeout: int) -> str:
            if self.selector == "time":
                return "Hace 2 horas"
            return "AI Engineer"

        def get_attribute(self, name: str):
            return None

    class FakePage:
        url = "https://www.linkedin.com/jobs/view/111"

        def __init__(self) -> None:
            self.waits = []

        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator(selector)

        def wait_for_timeout(self, milliseconds: int) -> None:
            self.waits.append(milliseconds)
            state["polls"] += 1

    page = FakePage()
    result = _wait_for_detail_hydration(
        page,
        require_date=True,
        max_wait_ms=1000,
    )

    assert result == "ready"
    assert page.waits == [100, 150]


def test_linkedin_panel_click_enriches_late_detail_without_top_level_goto():
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _enrich_job_detail_via_panel,
        _safe_job_card_link,
    )

    state = {"clicked": False, "polls": 0, "clicks": 0}

    class FakeLocator:
        def __init__(self, page, selector: str) -> None:
            self.page = page
            self.selector = selector

        @property
        def first(self):
            return self

        def count(self) -> int:
            if self.selector == ".job-card-container a[href*='/jobs/view/']":
                return 1
            if not state["clicked"]:
                return 0
            if self.selector in {
                ".jobs-search__job-details--container a[href*='/jobs/view/']",
                ".job-details-jobs-unified-top-card__job-title h1",
                ".jobs-description-content__text",
            }:
                return 1
            return int(self.selector == "time" and state["polls"] >= 2)

        def nth(self, _index: int):
            return self

        def is_visible(self, timeout: int) -> bool:
            return bool(self.count())

        def get_attribute(self, name: str):
            if name == "href" and "a[href*='/jobs/view/']" in self.selector:
                return "/jobs/view/111/?trk=safe"
            return None

        def inner_text(self, timeout: int) -> str:
            if self.selector == "time":
                return "2 hours ago"
            if self.selector == ".jobs-description-content__text":
                return "AI systems role with English required."
            return "AI Engineer"

        def click(self, timeout: int) -> None:
            state["clicked"] = True
            state["clicks"] += 1
            self.page.url = (
                "https://www.linkedin.com/jobs/search/?"
                "keywords=AI&currentJobId=111"
            )

    class FakePage:
        url = "https://www.linkedin.com/jobs/search/?keywords=AI"

        def __init__(self) -> None:
            self.waits = []

        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator(self, selector)

        def wait_for_timeout(self, milliseconds: int) -> None:
            self.waits.append(milliseconds)
            state["polls"] += 1

    page = FakePage()
    candidate = LinkedInVacancyRecord(
        linkedin_job_id="111",
        title="AI Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/111",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        matched_terms=["ai"],
    )
    card_link = _safe_job_card_link(page, candidate)

    enriched = _enrich_job_detail_via_panel(
        page,
        candidate,
        card_link=card_link,
        include_description=True,
        now=datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc),
    )

    assert state["clicks"] == 1
    assert page.waits == [100, 150]
    assert enriched.posted_at_text == "2 hours ago"
    assert enriched.published_at is not None
    assert enriched.description_excerpt.startswith("AI systems role")
    assert enriched.description_full_text.startswith("AI systems role")


def test_linkedin_detail_panel_wait_detects_stale_previous_job():
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _wait_for_detail_panel_hydration,
    )

    class FakeLocator:
        def __init__(self, selector: str) -> None:
            self.selector = selector

        @property
        def first(self):
            return self

        def count(self) -> int:
            return int(
                self.selector
                in {
                    ".jobs-search__job-details--container a[href*='/jobs/view/']",
                    ".job-details-jobs-unified-top-card__job-title h1",
                    ".jobs-description-content__text",
                    "time",
                }
            )

        def nth(self, _index: int):
            return self

        def is_visible(self, timeout: int) -> bool:
            return bool(self.count())

        def get_attribute(self, name: str):
            if name == "href" and "a[href*='/jobs/view/']" in self.selector:
                return "/jobs/view/999/"
            return None

        def inner_text(self, timeout: int) -> str:
            if self.selector == "time":
                return "1 hour ago"
            if self.selector == ".jobs-description-content__text":
                return "Previous role description"
            return "Previous Job"

    class FakePage:
        url = (
            "https://www.linkedin.com/jobs/search/?"
            "keywords=AI&currentJobId=999"
        )

        def __init__(self) -> None:
            self.waits = []

        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator(selector)

        def wait_for_timeout(self, milliseconds: int) -> None:
            self.waits.append(milliseconds)

    candidate = LinkedInVacancyRecord(
        linkedin_job_id="111",
        title="AI Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/111",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        matched_terms=["ai"],
    )

    assert _wait_for_detail_panel_hydration(
        FakePage(),
        candidate,
        include_description=True,
        max_wait_ms=250,
    ) == "stale"

    class EmptyPage(FakePage):
        url = "https://www.linkedin.com/jobs/search/?keywords=AI"

        def locator(self, selector: str):
            locator = super().locator(selector)
            locator.count = lambda: 0
            return locator

    assert _wait_for_detail_panel_hydration(
        EmptyPage(),
        candidate,
        include_description=True,
        max_wait_ms=250,
    ) == "timeout"


def test_linkedin_panel_selector_and_click_failures_are_distinct():
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_scraper import (
        LinkedInDetailPanelError,
        _enrich_job_detail_via_panel,
        _safe_job_card_link,
    )

    class EmptyLocator:
        @property
        def first(self):
            return self

        def count(self) -> int:
            return 0

    class EmptyPage:
        def locator(self, _selector: str) -> EmptyLocator:
            return EmptyLocator()

    class FailingLink:
        def click(self, timeout: int) -> None:
            raise ValueError("not clickable")

    candidate = LinkedInVacancyRecord(
        linkedin_job_id="111",
        title="AI Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/111",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        matched_terms=["ai"],
    )

    assert _safe_job_card_link(EmptyPage(), candidate) is None
    with pytest.raises(LinkedInDetailPanelError) as error:
        _enrich_job_detail_via_panel(
            EmptyPage(),
            candidate,
            card_link=FailingLink(),
            include_description=False,
        )
    assert error.value.reason == "card_click_failed"
    assert error.value.safe_label == "ValueError:runtime"


def test_linkedin_detail_click_cadence_is_deterministic():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _respect_detail_click_cadence,
    )

    class FakePage:
        def __init__(self) -> None:
            self.waits = []

        def wait_for_timeout(self, milliseconds: int) -> None:
            self.waits.append(milliseconds)

    page = FakePage()
    _respect_detail_click_cadence(
        page,
        last_detail_click_at=10.0,
        interval_ms=1200,
        now_fn=lambda: 10.2,
    )
    assert page.waits == [1000]


@pytest.mark.parametrize(
    ("selector", "posted_text", "expected_published", "expected_within"),
    [
        ("time", "2 hours ago", True, True),
        (".jobs-unified-top-card__posted-date", "2 days ago", True, False),
        (
            ".job-details-jobs-unified-top-card__primary-description-container",
            "Example AI · Tokyo",
            False,
            False,
        ),
    ],
)
def test_linkedin_detail_enrichment_verifies_date_without_inventing_it(
    selector,
    posted_text,
    expected_published,
    expected_within,
):
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _enrich_job_detail,
    )

    class FakeLocator:
        def __init__(self, active: bool) -> None:
            self._active = active

        @property
        def first(self):
            return self

        def count(self) -> int:
            return int(self._active)

        def inner_text(self, timeout: int) -> str:
            return posted_text

        def get_attribute(self, name: str):
            return None

    class FakePage:
        url = "https://www.linkedin.com/jobs/view/111"

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url

        def locator(self, requested: str) -> FakeLocator:
            return FakeLocator(requested == selector)

        def wait_for_timeout(self, _milliseconds: int) -> None:
            return None

    candidate = LinkedInVacancyRecord(
        linkedin_job_id="111",
        title="AI Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/111",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        matched_terms=["ai"],
    )
    with patch(
        "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
    ):
        enriched = _enrich_job_detail(
            FakePage(),
            candidate,
            include_description=False,
            now=datetime(2026, 7, 28, 14, 0, tzinfo=timezone.utc),
        )

    assert (enriched.published_at is not None) is expected_published
    assert enriched.is_within_24_hours is expected_within
    if not expected_published:
        assert enriched.posted_at_text == ""
        assert enriched.freshness_confidence == "low"


def test_linkedin_detail_enrichment_extracts_metadata_and_conservative_signals():
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _enrich_job_detail,
    )

    values = {
        ".job-details-jobs-unified-top-card__job-title h1": (
            "AI Engineer AI Engineer with verification"
        ),
        ".job-details-jobs-unified-top-card__company-name a": "Example AI",
        (
            ".job-details-jobs-unified-top-card__primary-description-container "
            ".tvm__text--low-emphasis"
        ): "Tokyo, Japan",
        ".job-details-jobs-unified-top-card__workplace-type": "Remote",
        ".jobs-description-content__text": (
            "Visa sponsorship is available. Relocation assistance is provided. "
            "International candidates are welcome. Business-level English and "
            "Japanese required, JLPT N2. 5+ years of experience building ML systems. "
            "Contact recruiter@example.com or +81 90 1234 5678."
        ),
        "time": "Hace 5 horas En las últimas 24 horas",
    }

    class FakeLocator:
        def __init__(self, value: str) -> None:
            self.value = value

        @property
        def first(self):
            return self

        def count(self) -> int:
            return int(bool(self.value))

        def inner_text(self, timeout: int) -> str:
            return self.value

        def get_attribute(self, name: str):
            return None

    class FakePage:
        url = "https://www.linkedin.com/jobs/view/111"

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url

        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator(values.get(selector, ""))

    candidate = LinkedInVacancyRecord(
        linkedin_job_id="111",
        title="AI Engineer AI Engineer with verification",
        canonical_url="https://www.linkedin.com/jobs/view/111",
        source_url=(
            "https://www.linkedin.com/jobs/search/?keywords=AI&location=Japan"
        ),
        matched_terms=["ai"],
    )
    with patch(
        "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
    ):
        enriched = _enrich_job_detail(
            FakePage(),
            candidate,
            include_description=True,
            now=datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc),
        )

    assert enriched.title == "AI Engineer"
    assert enriched.company_name == "Example AI"
    assert enriched.location == "Tokyo, Japan"
    assert enriched.workplace_type == "remote"
    assert enriched.posted_at_text == "Hace 5 horas"
    assert "Japanese (JLPT N2)" in enriched.language_requirements
    assert "English (business)" in enriched.language_requirements
    assert any("5+ years of experience" in item for item in enriched.experience_requirements)
    assert enriched.foreigner_acceptance == "yes"
    assert enriched.visa_status == "sponsorship"
    assert enriched.relocation_support == "yes"
    assert "recruiter@example.com" not in enriched.description_excerpt
    assert "[redacted-email]" in enriched.description_excerpt
    assert "+81 90 1234 5678" not in enriched.description_excerpt
    assert "[redacted-phone]" in enriched.description_excerpt
    assert "recruiter@example.com" in enriched.description_full_text
    assert "+81 90 1234 5678" in enriched.description_full_text


@pytest.mark.parametrize(
    ("text", "foreigner", "visa", "relocation"),
    [
        (
            "Foreign applicants are not accepted. No visa sponsorship. "
            "No relocation support.",
            "no",
            "no_sponsorship",
            "no",
        ),
        (
            "Visa sponsorship and relocation may be discussed. "
            "International candidates should review local requirements.",
            "ambiguous",
            "ambiguous",
            "ambiguous",
        ),
        (
            "We build machine learning systems for retail customers.",
            "unknown",
            "unknown",
            "unknown",
        ),
    ],
)
def test_linkedin_metadata_inference_is_conservative(
    text,
    foreigner,
    visa,
    relocation,
):
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _infer_foreigner_acceptance,
        _infer_relocation_support,
        _infer_visa_status,
    )

    assert _infer_foreigner_acceptance(text) == foreigner
    assert _infer_visa_status(text) == visa
    assert _infer_relocation_support(text) == relocation


def test_linkedin_multilingual_language_and_experience_inference():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _infer_experience_requirements,
        _infer_language_requirements,
    )

    text = (
        "Business-level English is required.\n"
        "日本語はビジネスレベル必須、JLPT N1以上。\n"
        "한국어 원어민 수준 우대, TOPIK 5급.\n"
        "Minimum of 5 years of experience in ML.\n"
        "データ分析経験3年以上。\n"
        "관련 분야 경력 4년 이상."
    )

    languages = _infer_language_requirements(text)
    experience = _infer_experience_requirements(text)

    assert "English (business)" in languages
    assert "Japanese (JLPT N1)" in languages
    assert "Korean (TOPIK 5)" in languages
    assert any("5 years of experience" in item for item in experience)
    assert any("3年以上" in item for item in experience)
    assert any("4년 이상" in item for item in experience)


@pytest.mark.parametrize(
    ("text", "foreigner", "visa", "relocation"),
    [
        (
            "外国人応募者歓迎。ビザサポートあり。転居支援あり。",
            "yes",
            "sponsorship",
            "yes",
        ),
        (
            "외국인 지원자 불가. 비자 지원 없음. 이주 지원 불가.",
            "no",
            "no_sponsorship",
            "no",
        ),
        (
            "海外応募者について相談可能。ビザと転居については面談で相談。",
            "ambiguous",
            "ambiguous",
            "ambiguous",
        ),
    ],
)
def test_linkedin_japanese_korean_mobility_inference(
    text,
    foreigner,
    visa,
    relocation,
):
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _infer_foreigner_acceptance,
        _infer_relocation_support,
        _infer_visa_status,
    )

    assert _infer_foreigner_acceptance(text) == foreigner
    assert _infer_visa_status(text) == visa
    assert _infer_relocation_support(text) == relocation


def test_linkedin_structured_sections_and_explicit_skills_are_bounded_deduped():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _extract_structured_sections,
        _infer_hard_skills,
        _infer_soft_skills,
    )

    text = """
Requirements
- 5+ years of experience with Python and SQL
- Strong communication skills and teamwork
- Strong communication skills and teamwork
- Experience with PyTorch, AWS, Docker and Kubernetes
- Familiarity with LLM, RAG and vector databases
- Problem-solving and stakeholder management
- Ownership and adaptability
- This seventh requirement must be capped
Responsibilities
- Build data pipelines with Spark and Databricks
- Develop NLP and Computer Vision models
- Collaborate with cross-functional teams
- Lead MLOps delivery
- Maintain Git workflows
- Present results in Power BI
- This seventh responsibility must be capped
"""
    expectations, responsibilities = _extract_structured_sections(text)
    hard_skills = _infer_hard_skills(text)
    soft_skills = _infer_soft_skills(text)

    assert len(expectations) == 6
    assert len(responsibilities) == 6
    assert len({item.casefold() for item in expectations}) == len(expectations)
    assert all(len(item) <= 180 for item in expectations + responsibilities)
    for skill in (
        "Python",
        "SQL",
        "PyTorch",
        "AWS",
        "Docker",
        "Kubernetes",
        "LLM",
        "RAG",
        "Vector DB",
        "Spark",
        "Databricks",
        "NLP",
        "Computer Vision",
        "MLOps",
        "Git",
        "Power BI",
    ):
        assert skill in hard_skills
    for skill in (
        "Communication",
        "Collaboration",
        "Problem solving",
        "Stakeholder management",
        "Ownership",
        "Adaptability",
        "Cross-functional collaboration",
    ):
        assert skill in soft_skills


def test_linkedin_japanese_korean_structured_section_headings():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _extract_structured_sections,
    )

    expectations, responsibilities = _extract_structured_sections(
        """
応募資格
- 日本語N2以上
- Pythonでの開発経験
仕事内容
- 機械学習モデルを開発する
자격요건
- 한국어 비즈니스 수준 필수
담당업무
- 데이터 파이프라인 구축
"""
    )

    assert expectations == [
        "日本語N2以上",
        "Pythonでの開発経験",
        "한국어 비즈니스 수준 필수",
    ]
    assert responsibilities == [
        "機械学習モデルを開発する",
        "데이터 파이프라인 구축",
    ]


def test_linkedin_full_description_is_audited_and_drives_inference(
    tmp_path,
    monkeypatch,
):
    from features.web_scraping.application import linkedin_audit
    from features.web_scraping.application.linkedin_audit import (
        persist_linkedin_audit_snapshot,
    )
    from features.web_scraping.application.linkedin_service import (
        _render_user_summary,
    )
    from features.web_scraping.domain.linkedin_models import (
        LinkedInJobsResult,
        LinkedInVacancyRecord,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _extract_job_detail_from_current_page,
    )

    marker = "FULL_DESCRIPTION_PRIVATE_MARKER"
    full_description = (
        "A" * 1200
        + f"\n{marker} Python Kubernetes TOPIK 4급 한국어 필수."
    )
    values = {
        ".job-details-jobs-unified-top-card__job-title h1": "AI Engineer",
        ".jobs-description-content__text": full_description,
    }

    class FakeLocator:
        def __init__(self, value: str) -> None:
            self.value = value

        @property
        def first(self):
            return self

        def count(self) -> int:
            return int(bool(self.value))

        def inner_text(self, timeout: int) -> str:
            return self.value

        def get_attribute(self, name: str):
            return None

    class FakePage:
        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator(values.get(selector, ""))

    candidate = LinkedInVacancyRecord(
        linkedin_job_id="111",
        title="AI Engineer",
        posted_at_text="2 hours ago",
        published_at=datetime(2026, 7, 29, 10, 0, tzinfo=timezone.utc),
        freshness_confidence="medium",
        is_within_24_hours=True,
        canonical_url="https://www.linkedin.com/jobs/view/111",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        matched_terms=["ai"],
    )

    enriched = _extract_job_detail_from_current_page(
        FakePage(),
        candidate,
        include_description=True,
    )
    serialized = enriched.model_dump_json()

    assert len(enriched.description_excerpt) == 1000
    assert marker not in enriched.description_excerpt
    assert marker in enriched.description_full_text
    assert marker in serialized
    assert "Python" in enriched.hard_skills
    assert "Kubernetes" in enriched.hard_skills
    assert "Korean (TOPIK 4)" in enriched.language_requirements

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        linkedin_audit,
        "get_request_runtime_config",
        lambda: SimpleNamespace(
            session_id="transient-description",
            request_id="req-1",
        ),
    )
    paths = persist_linkedin_audit_snapshot(
        original_query="AI jobs",
        queries=[],
        timings=[],
        vacancies=[enriched],
        rejected=[],
        warnings=[],
    )
    rendered = _render_user_summary(
        LinkedInJobsResult(status="ok", records=[enriched])
    )
    persisted_json = paths.json_path.read_text(encoding="utf-8")
    audit_summary = paths.summary_path.read_text(encoding="utf-8")

    assert marker in persisted_json
    assert "description_full_text" in persisted_json
    assert "Python Kubernetes TOPIK" in persisted_json
    assert marker in rendered
    assert "Python Kubernetes TOPIK" in rendered
    assert marker not in audit_summary


def test_linkedin_title_and_posted_date_cleanup_only_remove_verified_ui_noise():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _clean_posted_at_text,
        _normalize_repeated_title,
    )

    assert _normalize_repeated_title(
        "AI Engineer/AI エンジニア AI Engineer/AI エンジニア with verification"
    ) == "AI Engineer/AI エンジニア"
    assert _normalize_repeated_title("Verification Engineer") == "Verification Engineer"
    assert _clean_posted_at_text(
        "Hace 27 minutos En las últimas 24 horas"
    ) == "Hace 27 minutos"


def test_linkedin_vacancy_metadata_defaults_are_backward_compatible_and_forbid_extra():
    from pydantic import ValidationError

    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord

    legacy_payload = {
        "title": "AI Engineer",
        "canonical_url": "https://www.linkedin.com/jobs/view/111",
        "source_url": "https://www.linkedin.com/jobs/search/?keywords=AI",
    }
    record = LinkedInVacancyRecord.model_validate(legacy_payload)

    assert record.language_requirements == []
    assert record.experience_requirements == []
    assert record.hard_skills == []
    assert record.soft_skills == []
    assert record.candidate_expectations == []
    assert record.responsibilities == []
    assert record.foreigner_acceptance == "unknown"
    assert record.visa_status == "unknown"
    assert record.relocation_support == "unknown"

    with pytest.raises(ValidationError):
        LinkedInVacancyRecord.model_validate(
            {**legacy_payload, "unsupported_metadata": "value"}
        )
    with pytest.raises(ValidationError):
        LinkedInVacancyRecord.model_validate(
            {
                **legacy_payload,
                "candidate_expectations": [f"item {index}" for index in range(7)],
            }
        )
    with pytest.raises(ValidationError):
        LinkedInVacancyRecord.model_validate(
            {
                **legacy_payload,
                "responsibilities": ["x" * 181],
            }
        )


def test_linkedin_schema_1_1_audit_payload_loads_with_new_metadata_defaults():
    from features.web_scraping.domain.linkedin_models import LinkedInAuditSnapshot

    snapshot = LinkedInAuditSnapshot.model_validate(
        {
            "meta": {
                "schema_version": "1.1.0",
                "job_uid": "linkedin-job-legacy",
                "original_query": "AI jobs",
            },
            "vacancies": [
                {
                    "title": "AI Engineer",
                    "canonical_url": "https://www.linkedin.com/jobs/view/111",
                    "source_url": "https://www.linkedin.com/jobs/search/?keywords=AI",
                }
            ],
        }
    )

    assert snapshot.meta.schema_version == "1.1.0"
    assert snapshot.vacancies[0].hard_skills == []
    assert snapshot.vacancies[0].soft_skills == []
    assert snapshot.vacancies[0].candidate_expectations == []
    assert snapshot.vacancies[0].responsibilities == []


def test_linkedin_service_classifies_invalid_max_results_without_scraping(
    tmp_path,
    monkeypatch,
):
    from features.web_scraping.application import linkedin_audit
    from features.web_scraping.application.linkedin_service import (
        run_linkedin_jobs_vertical,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        linkedin_audit,
        "get_request_runtime_config",
        lambda: SimpleNamespace(session_id="sess-validation", request_id="req-validation"),
    )
    with patch(
        "features.web_scraping.application.linkedin_service.scrape_linkedin_jobs"
    ) as scrape:
        result = run_linkedin_jobs_vertical("vacantes AI de hoy", max_results=51)

    scrape.assert_not_called()
    assert result.status == "validation_error"
    assert "entre 1 y 50" in result.user_summary
    assert result.warnings == ["validation_error:ValidationError"]


def test_linkedin_tool_returns_readable_validation_error(tmp_path, monkeypatch):
    from features.web_scraping.application import linkedin_audit
    from integrations.linkedin_tools import scrape_linkedin_jobs_authenticated

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        linkedin_audit,
        "get_request_runtime_config",
        lambda: SimpleNamespace(session_id="sess-tool", request_id="req-tool"),
    )

    payload = json.loads(
        scrape_linkedin_jobs_authenticated.invoke(
            {"query": "", "location": "", "max_results": 0}
        )
    )

    assert payload["status"] == "validation_error"
    assert "entre 1 y 50" in payload["user_summary"]
    assert "storage_state" not in json.dumps(payload)


def test_linkedin_query_builder_enforces_date_and_sort_filters():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        build_linkedin_search_queries,
    )

    queries = build_linkedin_search_queries("Argentina")

    assert queries
    assert all("f_TPR=r86400" in url for _, url in queries)
    assert all("sortBy=DD" in url for _, url in queries)
    assert all("location=Argentina" in url for _, url in queries)


def test_linkedin_query_builder_consolidates_topics_per_location():
    from urllib.parse import parse_qs, urlparse

    from features.web_scraping.infrastructure.linkedin_scraper import (
        CONSOLIDATED_QUERY_PLANS,
        build_linkedin_search_queries,
    )

    queries = build_linkedin_search_queries(["South Korea", "Japan"])

    assert len(CONSOLIDATED_QUERY_PLANS) == 2
    assert len(queries) == 4
    assert sum("location=Japan" in url for _, url in queries) == 2
    assert sum("location=South+Korea" in url for _, url in queries) == 2
    assert all("f_TPR=r86400" in url and "sortBy=DD" in url for _, url in queries)
    assert [label for label, _url in queries] == [
        "AI/ML/Data/GenAI @ South Korea",
        "AI/ML/Data/GenAI @ Japan",
        "AI Agents/Product/Architecture @ South Korea",
        "AI Agents/Product/Architecture @ Japan",
    ]
    primary_keywords = [
        parse_qs(urlparse(url).query)["keywords"][0]
        for _, url in queries[:2]
    ]
    assert primary_keywords[0] == primary_keywords[1]
    keywords = " ".join(
        parse_qs(urlparse(url).query)["keywords"][0]
        for _, url in queries
    )
    for topic in (
        "AI Engineer",
        "Artificial Intelligence Engineer",
        "Machine Learning Engineer",
        "ML Engineer",
        "Data Scientist",
        "Data Analyst",
        "Deep Learning Engineer",
        "DL Engineer",
        "MLOps Engineer",
        "Generative AI Engineer",
        "Generative AI",
        "LLM Engineer",
        "LLM Scientist",
        "Speech LLM Engineer",
        "AI Agent Engineer",
        "AI Agent Developer",
        "AI Agent",
        "AI Product",
        "Applied AI Engineer",
        "AI Specialist",
        "AI Mentor",
        "AI Architect",
        "AI Solutions Architect",
        "Solution Architect AI",
        "Developer Technology Engineer AI",
        "Developer Technology Engineer - AI",
        "AI Automation Engineer",
        "RAG LLM System",
        "RAG & LLM System",
    ):
        assert topic in keywords



def test_linkedin_semantic_dedupe_collapses_same_post_with_different_ids():
    from datetime import datetime, timezone

    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _dedupe_linkedin_vacancies_semantically,
    )

    body = "\n".join(
        [
            "Acerca del empleo",
            "Position: AI Mentor",
            "We are looking for passionate AI Mentors to guide high school students as they explore artificial intelligence and build independent AI projects.",
            "Key Responsibilities",
            "Mentor and support students in learning AI and Python programming",
            "Guide students in developing independent AI projects",
            "Requirements",
            "Knowledge of AI concepts including Machine Learning, NLP, Computer Vision, or LLMs",
            "Strong Python programming fundamentals",
            "Experience with scikit-learn, TensorFlow, or other ML tools",
            "Native or fluent Japanese language skills are required.",
        ]
    )
    generic = LinkedInVacancyRecord(
        linkedin_job_id="4390928986",
        title="AI Mentor",
        company_name="Crimson Education",
        location="Japón",
        canonical_url="https://www.linkedin.com/jobs/view/4390928986",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI+Mentor&location=Japan",
        description_full_text=body,
        published_at=datetime(2026, 8, 2, 12, 50, tzinfo=timezone.utc),
    )
    specific = LinkedInVacancyRecord(
        linkedin_job_id="4390929954",
        title="AI Mentor",
        company_name="Crimson Education",
        location="Tokio, Tokio, Japón",
        canonical_url="https://www.linkedin.com/jobs/view/4390929954",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI+Mentor&location=Japan",
        description_full_text=body,
        published_at=datetime(2026, 8, 2, 14, 24, tzinfo=timezone.utc),
    )

    deduped, warnings = _dedupe_linkedin_vacancies_semantically([generic, specific])

    assert [record.linkedin_job_id for record in deduped] == ["4390929954"]
    assert warnings == ["semantic_duplicate_dropped:4390928986:kept:4390929954"]


def test_linkedin_relevance_accepts_seniority_prefixes_for_data_analyst():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        parse_linkedin_jobs_html,
    )

    html = """
    <div class="job-card-container">
      <a class="job-card-list__title--link" href="https://www.linkedin.com/jobs/view/123">
        <span class="job-card-list__title">Senior Data Analyst</span>
      </a>
      <span class="job-card-container__primary-description">Example Corp</span>
      <span class="job-card-container__metadata-item">Tokyo, Japan</span>
      <time datetime="2026-08-02T10:00:00Z">1 hour ago</time>
    </div>
    """

    records = parse_linkedin_jobs_html(
        html,
        source_url="https://www.linkedin.com/jobs/search/?keywords=Data+Analyst",
    )

    assert records
    assert records[0].title == "Senior Data Analyst"
    assert "data analyst" in records[0].matched_terms


def test_linkedin_query_builder_resolves_canonical_geo_ids_and_aliases():
    from urllib.parse import parse_qs, urlparse

    from features.web_scraping.infrastructure.linkedin_scraper import (
        build_linkedin_search_queries,
        resolve_linkedin_location,
    )

    assert resolve_linkedin_location("Corea del Sur").canonical_label == "South Korea"
    assert resolve_linkedin_location("대한민국").geo_id == "105149562"
    assert resolve_linkedin_location("Japón").canonical_label == "Japan"
    assert resolve_linkedin_location("日本").geo_id == "101355337"

    queries = build_linkedin_search_queries(
        ["Corea del Sur", "대한민국", "Japón", "日本"]
    )

    assert [label for label, _url in queries] == [
        "AI/ML/Data/GenAI @ South Korea",
        "AI/ML/Data/GenAI @ Japan",
        "AI Agents/Product/Architecture @ South Korea",
        "AI Agents/Product/Architecture @ Japan",
    ]
    assert len(queries) == 4
    params = [parse_qs(urlparse(url).query) for _label, url in queries]
    assert [query["location"] for query in params] == [
        ["South Korea"],
        ["Japan"],
        ["South Korea"],
        ["Japan"],
    ]
    assert [query["geoId"] for query in params] == [
        ["105149562"],
        ["101355337"],
        ["105149562"],
        ["101355337"],
    ]
    assert all(query["f_TPR"] == ["r86400"] for query in params)
    assert all(query["sortBy"] == ["DD"] for query in params)


def test_linkedin_query_builder_keeps_unknown_location_without_geo_id():
    from urllib.parse import parse_qs, urlparse

    from features.web_scraping.infrastructure.linkedin_scraper import (
        build_linkedin_search_queries,
        resolve_linkedin_location,
    )

    resolved = resolve_linkedin_location("Argentina")
    assert resolved.canonical_label == "Argentina"
    assert resolved.geo_id == ""

    queries = build_linkedin_search_queries(["Argentina", "argentina"])

    assert len(queries) == 2
    assert [label for label, _url in queries] == [
        "AI/ML/Data/GenAI @ Argentina",
        "AI Agents/Product/Architecture @ Argentina",
    ]
    for _label, url in queries:
        query = parse_qs(urlparse(url).query)
        assert query["location"] == ["Argentina"]
        assert "geoId" not in query


def test_linkedin_service_prompt_locations_override_environment(
    tmp_path,
    monkeypatch,
):
    from features.web_scraping.application import linkedin_audit
    from features.web_scraping.application.linkedin_service import (
        run_linkedin_jobs_vertical,
    )
    from features.web_scraping.domain.linkedin_models import (
        LinkedInQueryTiming,
        LinkedInRejectedRecord,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LINKEDIN_LOCATION", "Argentina")
    monkeypatch.setattr(
        linkedin_audit,
        "get_request_runtime_config",
        lambda: SimpleNamespace(session_id="sess-locations", request_id="req-locations"),
    )
    captured = {}

    def fake_scrape(request):
        captured["request"] = request
        now = datetime.now(timezone.utc)
        timing = LinkedInQueryTiming(
            query="AI Engineer @ Japan",
            started_at=now,
            completed_at=now,
            elapsed_ms=0,
            discovered_count=1,
            diagnostics={"parseable_candidate_count": 1},
        )
        rejected = [
            LinkedInRejectedRecord(
                source_url="https://www.linkedin.com/jobs/view/111",
                title="Accountant",
                reason="low_topic_relevance",
            )
        ]
        return [], rejected, [timing], [], ["https://www.linkedin.com/jobs/search/"]

    with patch(
        "features.web_scraping.application.linkedin_service.scrape_linkedin_jobs",
        side_effect=fake_scrape,
    ):
        result = run_linkedin_jobs_vertical(
            "Buscá vacantes LinkedIn de AI en Japón y Corea del Sur"
        )

    assert captured["request"].locations == ["Japan", "South Korea"]
    assert "Argentina" not in captured["request"].locations
    assert result.status == "ok"
    assert "No encontré vacantes verificables" in result.user_summary


def test_linkedin_service_uses_environment_only_without_prompt_location(
    tmp_path,
    monkeypatch,
):
    from features.web_scraping.application import linkedin_audit
    from features.web_scraping.application.linkedin_service import (
        run_linkedin_jobs_vertical,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LINKEDIN_LOCATION", "Argentina")
    monkeypatch.setattr(
        linkedin_audit,
        "get_request_runtime_config",
        lambda: SimpleNamespace(session_id="sess-env", request_id="req-env"),
    )
    captured = {}

    def fake_scrape(request):
        captured["request"] = request
        return [], [], [], [], []

    with patch(
        "features.web_scraping.application.linkedin_service.scrape_linkedin_jobs",
        side_effect=fake_scrape,
    ):
        result = run_linkedin_jobs_vertical("Buscá vacantes LinkedIn de AI de hoy")

    assert captured["request"].locations == ["Argentina"]
    assert result.status == "extraction_incomplete"


def test_linkedin_service_reports_incomplete_extraction_instead_of_false_zero(
    tmp_path,
    monkeypatch,
):
    from features.web_scraping.application import linkedin_audit
    from features.web_scraping.application.linkedin_service import (
        run_linkedin_jobs_vertical,
    )
    from features.web_scraping.domain.linkedin_models import LinkedInQueryTiming

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        linkedin_audit,
        "get_request_runtime_config",
        lambda: SimpleNamespace(session_id="sess-empty", request_id="req-empty"),
    )
    now = datetime.now(timezone.utc)
    timing = LinkedInQueryTiming(
        query="AI Engineer @ Japan",
        started_at=now,
        completed_at=now,
        elapsed_ms=4,
        diagnostics={
            "selector_counts": {".job-card-container": 7},
            "href_count": 9,
            "candidate_count": 16,
            "parseable_candidate_count": 0,
            "discard_reasons": {"invalid_url": 9, "missing_href": 7},
        },
    )

    with patch(
        "features.web_scraping.application.linkedin_service.scrape_linkedin_jobs",
        return_value=([], [], [timing], [], ["https://www.linkedin.com/jobs/search/"]),
    ):
        result = run_linkedin_jobs_vertical(
            "Buscá vacantes LinkedIn de AI en Japón"
        )

    assert result.status == "extraction_incomplete"
    assert "No puedo afirmar que haya cero vacantes" in result.user_summary
    assert "No encontré vacantes" not in result.user_summary
    assert result.warnings == ["extraction_incomplete:no_parseable_candidates"]


@pytest.mark.parametrize(
    ("probe_category", "summary_fragment"),
    [
        ("query_rate_limited", "limitó temporalmente"),
        ("query_access_rejected", "rechazó el acceso"),
        ("query_upstream_failure", "falla temporal"),
        ("query_navigation_failure", "completar la navegación"),
    ],
)
def test_linkedin_service_prioritizes_query_failure_over_no_parseable(
    tmp_path,
    monkeypatch,
    probe_category,
    summary_fragment,
):
    from features.web_scraping.application import linkedin_audit
    from features.web_scraping.application.linkedin_service import (
        run_linkedin_jobs_vertical,
    )
    from features.web_scraping.domain.linkedin_models import LinkedInQueryTiming

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        linkedin_audit,
        "get_request_runtime_config",
        lambda: SimpleNamespace(
            session_id="sess-probe",
            request_id="req-probe",
        ),
    )
    now = datetime.now(timezone.utc)
    timing = LinkedInQueryTiming(
        query="AI Engineer @ Japan",
        started_at=now,
        completed_at=now,
        elapsed_ms=5,
        error=f"probe:{probe_category}",
    )
    warnings = [
        f"query_probe_result:Japan:status_0:{probe_category}:no_cards"
    ]

    with patch(
        "features.web_scraping.application.linkedin_service.scrape_linkedin_jobs",
        return_value=(
            [],
            [],
            [timing],
            warnings,
            ["https://www.linkedin.com/jobs/search/"],
        ),
    ):
        result = run_linkedin_jobs_vertical(
            "Buscá vacantes LinkedIn de AI en Japón"
        )

    assert result.status == "extraction_incomplete"
    assert summary_fragment in result.user_summary
    assert "candidatos parseables" not in result.user_summary
    assert (
        f"extraction_incomplete:{probe_category}" in result.warnings
    )


@pytest.mark.parametrize(
    ("scrape_warning", "expected_status", "summary_fragment"),
    [
        (
            "query_empty_results_explicit:AI Engineer @ Japan",
            "ok",
            "No encontré vacantes verificables",
        ),
        (
            "query_hydration_timeout:no_terminal_signal:AI Engineer @ Japan",
            "extraction_incomplete",
            "no mostró cards ni un estado explícito",
        ),
    ],
)
def test_linkedin_service_distinguishes_explicit_empty_from_hydration_timeout(
    tmp_path,
    monkeypatch,
    scrape_warning,
    expected_status,
    summary_fragment,
):
    from features.web_scraping.application import linkedin_audit
    from features.web_scraping.application.linkedin_service import (
        run_linkedin_jobs_vertical,
    )
    from features.web_scraping.domain.linkedin_models import LinkedInQueryTiming

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        linkedin_audit,
        "get_request_runtime_config",
        lambda: SimpleNamespace(
            session_id="sess-hydration",
            request_id="req-hydration",
        ),
    )
    now = datetime.now(timezone.utc)
    timing = LinkedInQueryTiming(
        query="AI Engineer @ Japan",
        started_at=now,
        completed_at=now,
        elapsed_ms=450,
    )
    with patch(
        "features.web_scraping.application.linkedin_service.scrape_linkedin_jobs",
        return_value=(
            [],
            [],
            [timing],
            [scrape_warning],
            ["https://www.linkedin.com/jobs/search/"],
        ),
    ):
        result = run_linkedin_jobs_vertical(
            "Buscá vacantes LinkedIn de AI en Japón"
        )

    assert result.status == expected_status
    assert summary_fragment in result.user_summary
    if expected_status == "extraction_incomplete":
        assert "extraction_incomplete:query_hydration_timeout" in result.warnings


@pytest.mark.parametrize(
    ("rejection_reason", "expected_status", "summary_fragment"),
    [
        (
            "unverified_posted_date",
            "extraction_incomplete",
            "devolvió candidatos",
        ),
        (
            "detail_budget_exhausted",
            "extraction_incomplete",
            "presupuesto seguro",
        ),
        (
            "detail_network_failure",
            "extraction_incomplete",
            "falla de red",
        ),
        (
            "outside_24_hours",
            "ok",
            "No encontré vacantes verificables",
        ),
    ],
)
def test_linkedin_service_distinguishes_unverified_dates_from_true_zero(
    tmp_path,
    monkeypatch,
    rejection_reason,
    expected_status,
    summary_fragment,
):
    from features.web_scraping.application import linkedin_audit
    from features.web_scraping.application.linkedin_service import (
        run_linkedin_jobs_vertical,
    )
    from features.web_scraping.domain.linkedin_models import (
        LinkedInQueryTiming,
        LinkedInRejectedRecord,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        linkedin_audit,
        "get_request_runtime_config",
        lambda: SimpleNamespace(session_id="sess-date", request_id="req-date"),
    )
    now = datetime.now(timezone.utc)
    timing = LinkedInQueryTiming(
        query="AI Engineer @ Japan",
        started_at=now,
        completed_at=now,
        elapsed_ms=1,
        discovered_count=1,
        diagnostics={"parseable_candidate_count": 1},
    )
    rejected = [
        LinkedInRejectedRecord(
            source_url="https://www.linkedin.com/jobs/view/111",
            title="AI Engineer",
            reason=rejection_reason,
        )
    ]

    with patch(
        "features.web_scraping.application.linkedin_service.scrape_linkedin_jobs",
        return_value=([], rejected, [timing], [], ["https://www.linkedin.com/jobs/search/"]),
    ):
        result = run_linkedin_jobs_vertical("Buscá vacantes LinkedIn de AI en Japón")

    assert result.status == expected_status
    assert summary_fragment in result.user_summary
    if rejection_reason == "detail_network_failure":
        assert "NO se agotó" in result.user_summary
        assert result.warnings == [
            "extraction_incomplete:detail_network_failure"
        ]


def _linkedin_summary_record(
    job_id: str,
    location: str,
    country: str,
    *,
    long_metadata: bool = False,
):
    from features.web_scraping.domain.linkedin_models import (
        LinkedInVacancyRecord,
    )

    metadata_suffix = (
        " for production machine learning platforms and international teams"
        if long_metadata
        else ""
    )
    return LinkedInVacancyRecord(
        linkedin_job_id=job_id,
        title=f"Senior AI Machine Learning Engineer {job_id}{metadata_suffix}",
        company_name=f"Company {job_id} International Research",
        location=location,
        workplace_type="hybrid",
        posted_at_text="Hace 2 horas",
        published_at=datetime(2026, 7, 29, 10, 0, tzinfo=timezone.utc),
        freshness_confidence="medium",
        is_within_24_hours=True,
        canonical_url=f"https://www.linkedin.com/jobs/view/{job_id}",
        source_url=(
            "https://www.linkedin.com/jobs/search/?keywords=AI"
            f"&location={country}"
        ),
        description_excerpt="Visible job description.",
        description_full_text=(
            f"Full job body for {job_id}.\n"
            "This role expects ownership of AI systems, production delivery, "
            "stakeholder communication, model evaluation, and platform collaboration."
        ),
        language_requirements=[
            "English business proficiency",
            "Japanese professional proficiency",
        ],
        experience_requirements=[
            "5+ years of experience building production machine learning systems"
        ],
        hard_skills=[
            "Python",
            "PyTorch",
            "TensorFlow",
            "Distributed systems",
        ],
        soft_skills=["Communication", "Collaboration"],
        foreigner_acceptance="yes",
        visa_status="sponsorship",
        relocation_support="yes",
        matched_terms=["ai"],
    )


def test_linkedin_user_summary_groups_countries_and_keeps_requested_metadata():
    from features.web_scraping.application.linkedin_service import (
        _render_user_summary,
    )
    from features.web_scraping.domain.linkedin_models import LinkedInJobsResult

    result = LinkedInJobsResult(
        status="ok",
        records=[
            _linkedin_summary_record("111", "Seoul, South Korea", "South+Korea"),
            _linkedin_summary_record("222", "Tokyo, Japan", "Japan"),
        ],
    )

    summary = _render_user_summary(result)

    assert "### 🇰🇷 Corea del Sur" in summary
    assert "### 🇯🇵 Japón" in summary
    assert "Company 111" in summary
    assert "Company 222" in summary
    assert "Communication" in summary
    for label in (
        "Empresa:",
        "Ubicación:",
        "Modalidad:",
        "Fecha:",
        "Skills:",
        "Idioma:",
        "Experiencia:",
        "Expectativas:",
        "Responsabilidades:",
        "Extranjeros / Visa:",
    ):
        assert summary.count(label) == 2
    assert "##### Body completo" in summary
    assert "Full job body for 111" in summary
    assert "Full job body for 222" in summary
    assert "Empresa no informada" not in summary


@pytest.mark.parametrize(
    ("korea_count", "japan_count"),
    [(5, 1), (1, 5)],
)
def test_linkedin_user_summary_guarantees_asymmetric_country_coverage(
    korea_count,
    japan_count,
):
    from features.web_scraping.application.linkedin_service import (
        _render_user_summary,
    )
    from features.web_scraping.domain.linkedin_models import LinkedInJobsResult

    korea_records = [
        _linkedin_summary_record(
            f"1{index:02d}",
            "Seoul, South Korea",
            "South+Korea",
        )
        for index in range(korea_count)
    ]
    japan_records = [
        _linkedin_summary_record(
            f"2{index:02d}",
            "Tokyo, Japan",
            "Japan",
        )
        for index in range(japan_count)
    ]

    summary = _render_user_summary(
        LinkedInJobsResult(
            status="ok",
            records=[*korea_records, *japan_records],
        )
    )

    assert korea_records[0].canonical_url in summary
    assert japan_records[0].canonical_url in summary
    assert "##### Body completo" in summary


def test_linkedin_user_summary_covers_three_groups_with_long_metadata():
    from features.web_scraping.application.linkedin_service import (
        _render_user_summary,
    )
    from features.web_scraping.domain.linkedin_models import LinkedInJobsResult

    records = [
        _linkedin_summary_record(
            "301",
            "Seoul, South Korea",
            "South+Korea",
            long_metadata=True,
        ),
        _linkedin_summary_record(
            "302",
            "Tokyo, Japan",
            "Japan",
            long_metadata=True,
        ),
        _linkedin_summary_record(
            "303",
            "Singapore",
            "Singapore",
            long_metadata=True,
        ),
    ]

    summary = _render_user_summary(
        LinkedInJobsResult(status="ok", records=records)
    )

    for heading in ("### 🇰🇷 Corea del Sur", "### 🇯🇵 Japón", "### 📍 Otras ubicaciones"):
        assert heading in summary
    for record in records:
        assert record.canonical_url in summary
    assert "##### Body completo" in summary


def test_linkedin_user_summary_has_no_orphan_country_headings():
    from features.web_scraping.application.linkedin_service import (
        _render_user_summary,
    )
    from features.web_scraping.domain.linkedin_models import LinkedInJobsResult

    records = [
        *[
            _linkedin_summary_record(
                f"4{index:02d}",
                "Seoul, South Korea",
                "South+Korea",
            )
            for index in range(5)
        ],
        _linkedin_summary_record("499", "Tokyo, Japan", "Japan"),
    ]
    summary_lines = _render_user_summary(
        LinkedInJobsResult(status="ok", records=records)
    ).splitlines()

    for index, line in enumerate(summary_lines):
        if line.startswith("### ") and "vacantes verificadas" not in line:
            following = next(
                (candidate for candidate in summary_lines[index + 1 :] if candidate.strip()),
                "",
            )
            assert following.startswith("#### 📌 ")


def test_linkedin_user_summary_reports_exact_omitted_count_and_is_deterministic():
    from features.web_scraping.application.linkedin_service import (
        _render_user_summary,
    )
    from features.web_scraping.domain.linkedin_models import LinkedInJobsResult

    records = [
        *[
            _linkedin_summary_record(
                f"5{index:02d}",
                "Seoul, South Korea",
                "South+Korea",
                long_metadata=True,
            )
            for index in range(5)
        ],
        *[
            _linkedin_summary_record(
                f"6{index:02d}",
                "Tokyo, Japan",
                "Japan",
                long_metadata=True,
            )
            for index in range(5)
        ],
    ]
    result = LinkedInJobsResult(status="ok", records=records)

    first = _render_user_summary(result)
    second = _render_user_summary(result)
    rendered_urls = sum(record.canonical_url in first for record in records)
    omitted = len(records) - rendered_urls

    assert first == second
    assert rendered_urls == len(records)
    assert omitted == 0


def test_linkedin_scraper_dedupes_before_applying_result_limit():
    from features.web_scraping.domain.linkedin_models import (
        LinkedInJobsRequest,
        LinkedInParseDiagnostics,
        LinkedInVacancyRecord,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        scrape_linkedin_jobs,
    )

    def record(job_id: str) -> LinkedInVacancyRecord:
        return LinkedInVacancyRecord(
                linkedin_job_id=job_id,
                title=f"AI Engineer {job_id}",
                posted_at_text="2 hours ago",
                published_at=datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc),
                freshness_confidence="medium",
                is_within_24_hours=True,
            canonical_url=f"https://www.linkedin.com/jobs/view/{job_id}",
            source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
            matched_terms=["ai"],
        )

    first = record("111")
    second = record("222")
    unverified = record("333").model_copy(
        update={
            "posted_at_text": "fecha desconocida",
            "published_at": None,
            "freshness_confidence": "low",
            "is_within_24_hours": False,
        }
    )

    class FakeStore:
        path = Path("/private/fake-storage-state.json")

        def load_browser_metadata(self):
            return {
                "browser": "chromium",
                "executable_path": "",
                "profile_mode": "persistent",
                "profile_path": "/private/browser-profile",
            }

        def resolve_profile_path(self, **_kwargs):
            return Path("/private/browser-profile")

        def record_runtime_failure(self, *_args, **_kwargs) -> None:
            raise AssertionError("unexpected runtime failure")

        def invalidate(self, reason: str = "") -> None:
            raise AssertionError(f"unexpected invalidation: {reason}")

    class FakePage:
        url = "https://www.linkedin.com/jobs/"

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url

        def wait_for_timeout(self, _milliseconds: int) -> None:
            return None

        def content(self) -> str:
            return "<html></html>"

    class FakeSession:
        page = FakePage()

        def close(self) -> None:
            return None

    with (
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.open_persistent_authenticated_context",
            return_value=FakeSession(),
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.build_linkedin_search_queries",
            return_value=[
                ("AI Engineer", "https://www.linkedin.com/jobs/search/?keywords=AI"),
                ("ML Engineer", "https://www.linkedin.com/jobs/search/?keywords=ML"),
            ],
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._parse_linkedin_jobs_html_with_diagnostics",
            side_effect=[
                (
                    [first, first, unverified],
                    LinkedInParseDiagnostics(parseable_candidate_count=3),
                ),
                (
                    [second],
                    LinkedInParseDiagnostics(parseable_candidate_count=1),
                ),
            ],
        ),
    ):
        records, rejected, _timings, _warnings, _queries = scrape_linkedin_jobs(
            LinkedInJobsRequest(
                query="vacantes AI de hoy",
                max_results=2,
                include_description=False,
            ),
            session_store=FakeStore(),  # type: ignore[arg-type]
        )

    assert [item.linkedin_job_id for item in records] == ["111", "222"]
    assert [item.reason for item in rejected] == [
        "duplicate",
        "card_click_failed",
    ]


def test_linkedin_scraper_enriches_inline_once_and_never_gotos_job_detail(
    monkeypatch,
):
    from features.web_scraping.domain.linkedin_models import (
        LinkedInJobsRequest,
        LinkedInParseDiagnostics,
        LinkedInVacancyRecord,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        scrape_linkedin_jobs,
    )

    monkeypatch.setenv("LINKEDIN_DIRECT_DETAIL_FALLBACK", "false")
    candidate = LinkedInVacancyRecord(
        linkedin_job_id="111",
        title="AI Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/111",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        matched_terms=["ai"],
    )
    navigations = []

    class FakePage:
        url = "https://www.linkedin.com/jobs/"

        def goto(self, url: str, **_kwargs) -> None:
            navigations.append(url)
            self.url = url

        def wait_for_timeout(self, _milliseconds: int) -> None:
            return None

        def content(self) -> str:
            return "<html></html>"

    class FakeSession:
        page = FakePage()

        def close(self) -> None:
            return None

    class FakeStore:
        def load_browser_metadata(self):
            return {
                "browser": "chromium",
                "executable_path": "",
                "profile_path": "/private/browser-profile",
            }

        def resolve_profile_path(self, **_kwargs):
            return Path("/private/browser-profile")

        def record_runtime_failure(self, *_args, **_kwargs) -> None:
            raise AssertionError("unexpected runtime failure")

    recent = candidate.model_copy(
        update={
            "posted_at_text": "2 hours ago",
            "published_at": datetime(2026, 7, 29, 10, 0, tzinfo=timezone.utc),
            "freshness_confidence": "medium",
            "is_within_24_hours": True,
        }
    )
    with (
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.open_persistent_authenticated_context",
            return_value=FakeSession(),
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._wait_for_search_results_hydration",
            return_value="results",
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.build_linkedin_search_queries",
            return_value=[
                ("Primary @ Japan", "https://www.linkedin.com/jobs/search/?keywords=AI"),
                ("Fallback @ Japan", "https://www.linkedin.com/jobs/search/?keywords=ML"),
            ],
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._parse_linkedin_jobs_html_with_diagnostics",
            return_value=(
                [candidate],
                LinkedInParseDiagnostics(parseable_candidate_count=1),
            ),
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._safe_job_card_link",
            return_value=MagicMock(),
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._enrich_job_detail_via_panel",
            return_value=recent,
        ) as panel_detail,
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._enrich_job_detail"
        ) as direct_detail,
    ):
        records, rejected, _timings, _warnings, _queries = scrape_linkedin_jobs(
            LinkedInJobsRequest(
                query="AI jobs Japan",
                locations=["Japan"],
                max_results=5,
                include_description=False,
            ),
            session_store=FakeStore(),  # type: ignore[arg-type]
        )

    assert [record.linkedin_job_id for record in records] == ["111"]
    assert [item.reason for item in rejected] == ["duplicate"]
    assert panel_detail.call_count == 1
    direct_detail.assert_not_called()
    assert not any("/jobs/view/" in url for url in navigations)


def test_linkedin_discovery_enriches_by_source_before_final_round_robin(
    monkeypatch,
):
    from features.web_scraping.domain.linkedin_models import (
        LinkedInJobsRequest,
        LinkedInParseDiagnostics,
        LinkedInVacancyRecord,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        scrape_linkedin_jobs,
    )

    monkeypatch.setenv("LINKEDIN_DETAIL_BUDGET", "4")
    monkeypatch.setenv("LINKEDIN_MAX_QUERIES_PER_LOCATION", "2")
    monkeypatch.setenv("LINKEDIN_DIRECT_DETAIL_FALLBACK", "false")
    query_plan = [
        ("Primary @ South Korea", "https://www.linkedin.com/jobs/search/?keywords=kr-primary"),
        ("Primary @ Japan", "https://www.linkedin.com/jobs/search/?keywords=jp-primary"),
        ("Complementary @ South Korea", "https://www.linkedin.com/jobs/search/?keywords=kr-extra"),
        ("Complementary @ Japan", "https://www.linkedin.com/jobs/search/?keywords=jp-extra"),
    ]
    events: list[str] = []
    published_at = datetime(2026, 7, 29, 10, 0, tzinfo=timezone.utc)

    class FakePage:
        url = "https://www.linkedin.com/jobs/"

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url
            if "/jobs/search/" in url:
                events.append(f"goto:{url}")

        def wait_for_timeout(self, _milliseconds: int) -> None:
            return None

        def content(self) -> str:
            return "<html></html>"

        def is_closed(self) -> bool:
            return False

    class FakeSession:
        page = FakePage()

        def page_is_alive(self) -> bool:
            return True

        def close(self) -> None:
            return None

    class FakeStore:
        def load_browser_metadata(self):
            return {
                "browser": "chromium",
                "executable_path": "",
                "profile_path": "/private/browser-profile",
            }

        def resolve_profile_path(self, **_kwargs):
            return Path("/private/browser-profile")

        def record_runtime_failure(self, *_args, **_kwargs) -> None:
            raise AssertionError("unexpected runtime failure")

    def parse_results(_html, *, source_url, now):
        del _html, now
        if "primary" not in source_url:
            return [], LinkedInParseDiagnostics()
        if "kr-primary" in source_url:
            records = [
                LinkedInVacancyRecord(
                    linkedin_job_id="101",
                    title="AI Engineer South Korea 101",
                    canonical_url="https://www.linkedin.com/jobs/view/101",
                    source_url=source_url,
                    posted_at_text="Hace 1 hora",
                    published_at=published_at,
                    freshness_confidence="medium",
                    is_within_24_hours=True,
                    matched_terms=["ai"],
                ),
                LinkedInVacancyRecord(
                    linkedin_job_id="102",
                    title="AI Engineer South Korea 102",
                    canonical_url="https://www.linkedin.com/jobs/view/102",
                    source_url=source_url,
                    posted_at_text="Hace 1 hora",
                    published_at=published_at,
                    freshness_confidence="medium",
                    is_within_24_hours=True,
                    matched_terms=["ai"],
                ),
            ]
        else:
            records = [
                LinkedInVacancyRecord(
                    linkedin_job_id="201",
                    title="AI Engineer Japan 201",
                    canonical_url="https://www.linkedin.com/jobs/view/201",
                    source_url=source_url,
                    posted_at_text="Hace 1 hora",
                    published_at=published_at,
                    freshness_confidence="medium",
                    is_within_24_hours=True,
                    matched_terms=["ai"],
                )
            ]
        return (
            records,
            LinkedInParseDiagnostics(parseable_candidate_count=len(records)),
        )

    def enrich(_page, record, **_kwargs):
        events.append(f"detail:{record.linkedin_job_id}")
        return record.model_copy(
            update={
                "company_name": f"Company {record.linkedin_job_id}",
                "location": (
                    "Seoul, South Korea"
                    if record.linkedin_job_id == "101"
                    else "Tokyo, Japan"
                ),
                "workplace_type": "hybrid",
                "hard_skills": ["Python"],
                "description_full_text": "Build AI systems with Python for Tokyo products.",
            }
        )

    with (
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.open_persistent_authenticated_context",
            return_value=FakeSession(),
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._wait_for_search_results_hydration",
            return_value="results",
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.build_linkedin_search_queries",
            return_value=query_plan,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._parse_linkedin_jobs_html_with_diagnostics",
            side_effect=parse_results,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._safe_job_card_link",
            return_value=MagicMock(),
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._enrich_job_detail_via_panel",
            side_effect=enrich,
        ) as panel_detail,
    ):
        records, _rejected, timings, _warnings, queries = scrape_linkedin_jobs(
            LinkedInJobsRequest(
                query="AI jobs South Korea and Japan",
                locations=["South Korea", "Japan"],
                max_results=4,
                include_description=True,
            ),
            session_store=FakeStore(),  # type: ignore[arg-type]
        )

    first_detail = next(
        index for index, event in enumerate(events) if event.startswith("detail:")
    )
    assert events[:4] == [f"goto:{url}" for _, url in query_plan]
    assert first_detail >= 4
    assert [
        event for event in events[first_detail:] if event.startswith("detail:")
    ][:3] == ["detail:101", "detail:102", "detail:201"]
    assert queries == [url for _, url in query_plan]
    assert [timing.query for timing in timings] == [
        label for label, _ in query_plan
    ]
    assert [record.company_name for record in records] == [
        "Company 101",
        "Company 201",
        "Company 102",
    ]
    assert panel_detail.call_count == 3



def test_linkedin_stale_detail_panel_retries_same_candidate_once(monkeypatch):
    from features.web_scraping.domain.linkedin_models import (
        LinkedInJobsRequest,
        LinkedInParseDiagnostics,
        LinkedInVacancyRecord,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        LinkedInDetailPanelError,
        scrape_linkedin_jobs,
    )

    monkeypatch.setenv("LINKEDIN_DETAIL_BUDGET", "3")
    monkeypatch.setenv("LINKEDIN_MAX_QUERIES_PER_LOCATION", "1")
    monkeypatch.setenv("LINKEDIN_DIRECT_DETAIL_FALLBACK", "false")
    source_url = "https://www.linkedin.com/jobs/search/?keywords=kr-primary"
    events: list[str] = []

    class FakePage:
        url = "https://www.linkedin.com/jobs/"

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url
            if "/jobs/search/" in url:
                events.append(f"goto:{url}")

        def wait_for_timeout(self, _milliseconds: int) -> None:
            return None

        def content(self) -> str:
            return "<html></html>"

        def is_closed(self) -> bool:
            return False

    class FakeSession:
        page = FakePage()

        def page_is_alive(self) -> bool:
            return True

        def close(self) -> None:
            return None

    class FakeStore:
        def load_browser_metadata(self):
            return {
                "browser": "chromium",
                "executable_path": "",
                "profile_path": "/private/browser-profile",
            }

        def resolve_profile_path(self, **_kwargs):
            return Path("/private/browser-profile")

        def record_runtime_failure(self, *_args, **_kwargs) -> None:
            raise AssertionError("unexpected runtime failure")

    candidate = LinkedInVacancyRecord(
        linkedin_job_id="101",
        title="AI Engineer Korea",
        canonical_url="https://www.linkedin.com/jobs/view/101",
        source_url=source_url,
        posted_at_text="Hace 1 hora",
        published_at=datetime(2026, 7, 29, 10, 0, tzinfo=timezone.utc),
        freshness_confidence="medium",
        is_within_24_hours=True,
        matched_terms=["ai"],
    )

    def enrich(_page, record, **_kwargs):
        events.append(f"detail:{record.linkedin_job_id}")
        if events.count("detail:101") == 1:
            raise LinkedInDetailPanelError("stale_detail_panel")
        return record.model_copy(
            update={
                "company_name": "Company 101",
                "location": "Seoul, South Korea",
                "workplace_type": "hybrid",
                "hard_skills": ["Python"],
                "description_full_text": "Build AI systems with Python in Korea.",
            }
        )

    with (
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.open_persistent_authenticated_context",
            return_value=FakeSession(),
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._wait_for_search_results_hydration",
            return_value="results",
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.build_linkedin_search_queries",
            return_value=[("Primary @ South Korea", source_url)],
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._parse_linkedin_jobs_html_with_diagnostics",
            return_value=(
                [candidate],
                LinkedInParseDiagnostics(parseable_candidate_count=1),
            ),
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._safe_job_card_link",
            return_value=MagicMock(),
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._enrich_job_detail_via_panel",
            side_effect=enrich,
        ) as panel_detail,
    ):
        records, rejected, _timings, warnings, _queries = scrape_linkedin_jobs(
            LinkedInJobsRequest(
                query="AI jobs South Korea",
                locations=["South Korea"],
                max_results=3,
                include_description=True,
            ),
            session_store=FakeStore(),  # type: ignore[arg-type]
        )

    assert [record.linkedin_job_id for record in records] == ["101"]
    assert rejected == []
    assert panel_detail.call_count == 2
    assert events.count("detail:101") == 2
    assert any(warning == "detail_panel_stale_retry:101" for warning in warnings)


def test_linkedin_panel_failure_isolated_and_bodyless_record_is_rejected(
    monkeypatch,
):
    from features.web_scraping.domain.linkedin_models import (
        LinkedInJobsRequest,
        LinkedInParseDiagnostics,
        LinkedInVacancyRecord,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        LinkedInDetailPanelError,
        scrape_linkedin_jobs,
    )

    monkeypatch.setenv("LINKEDIN_DETAIL_BUDGET", "4")
    monkeypatch.setenv("LINKEDIN_DIRECT_DETAIL_FALLBACK", "false")
    published_at = datetime(2026, 7, 29, 10, 0, tzinfo=timezone.utc)
    query_plan = [
        ("Primary @ South Korea", "https://www.linkedin.com/jobs/search/?keywords=kr"),
        ("Primary @ Japan", "https://www.linkedin.com/jobs/search/?keywords=jp"),
    ]

    class FakePage:
        url = "https://www.linkedin.com/jobs/"

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url

        def wait_for_timeout(self, _milliseconds: int) -> None:
            return None

        def content(self) -> str:
            return "<html></html>"

        def is_closed(self) -> bool:
            return False

    class FakeSession:
        page = FakePage()

        def page_is_alive(self) -> bool:
            return True

        def close(self) -> None:
            return None

    class FakeStore:
        def load_browser_metadata(self):
            return {
                "browser": "chromium",
                "executable_path": "",
                "profile_path": "/private/browser-profile",
            }

        def resolve_profile_path(self, **_kwargs):
            return Path("/private/browser-profile")

        def record_runtime_failure(self, *_args, **_kwargs) -> None:
            raise AssertionError("unexpected runtime failure")

    def parse_results(_html, *, source_url, now):
        del _html, now
        country = "South Korea" if "keywords=kr" in source_url else "Japan"
        job_id = "101" if country == "South Korea" else "201"
        return (
            [
                LinkedInVacancyRecord(
                    linkedin_job_id=job_id,
                    title=f"AI Engineer {country}",
                    canonical_url=f"https://www.linkedin.com/jobs/view/{job_id}",
                    source_url=source_url,
                    posted_at_text="Hace 1 hora",
                    published_at=published_at,
                    freshness_confidence="medium",
                    is_within_24_hours=True,
                    matched_terms=["ai"],
                )
            ],
            LinkedInParseDiagnostics(parseable_candidate_count=1),
        )

    def enrich(_page, record, **_kwargs):
        if record.linkedin_job_id == "101":
            raise LinkedInDetailPanelError(
                "detail_network_failure",
                safe_label="chromium_err_connection_reset",
            )
        return record.model_copy(
            update={
                "company_name": "Tokyo AI",
                "location": "Tokyo, Japan",
                "workplace_type": "remote",
                "hard_skills": ["Python"],
                "description_full_text": "Build AI systems with Python for Tokyo products.",
            }
        )

    with (
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.open_persistent_authenticated_context",
            return_value=FakeSession(),
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._wait_for_search_results_hydration",
            return_value="results",
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.build_linkedin_search_queries",
            return_value=query_plan,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._parse_linkedin_jobs_html_with_diagnostics",
            side_effect=parse_results,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._safe_job_card_link",
            return_value=MagicMock(),
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._enrich_job_detail_via_panel",
            side_effect=enrich,
        ),
    ):
        records, rejected, _timings, warnings, _queries = scrape_linkedin_jobs(
            LinkedInJobsRequest(
                query="AI jobs South Korea and Japan",
                locations=["South Korea", "Japan"],
                max_results=4,
                include_description=True,
            ),
            session_store=FakeStore(),  # type: ignore[arg-type]
        )

    assert [record.linkedin_job_id for record in records] == ["201"]
    assert records[0].company_name == "Tokyo AI"
    assert records[0].description_full_text
    assert [item.reason for item in rejected] == ["detail_network_failure"]
    assert (
        "metadata_enrichment_incomplete:101:detail_network_failure"
        in warnings
    )


def test_linkedin_inline_panel_budget_and_click_cadence_are_enforced(
    monkeypatch,
):
    from features.web_scraping.domain.linkedin_models import (
        LinkedInJobsRequest,
        LinkedInParseDiagnostics,
        LinkedInVacancyRecord,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        scrape_linkedin_jobs,
    )

    monkeypatch.setenv("LINKEDIN_DETAIL_BUDGET", "2")
    monkeypatch.setenv("LINKEDIN_DETAIL_CLICK_INTERVAL_MS", "1200")
    monkeypatch.setenv("LINKEDIN_DIRECT_DETAIL_FALLBACK", "false")

    candidates = [
        LinkedInVacancyRecord(
            linkedin_job_id=job_id,
            title=f"AI Engineer {job_id}",
            canonical_url=f"https://www.linkedin.com/jobs/view/{job_id}",
            source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
            matched_terms=["ai"],
        )
        for job_id in ("111", "222", "333")
    ]

    class FakePage:
        url = "https://www.linkedin.com/jobs/"

        def __init__(self) -> None:
            self.waits = []

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url

        def wait_for_timeout(self, milliseconds: int) -> None:
            self.waits.append(milliseconds)

        def content(self) -> str:
            return "<html></html>"

    class FakeSession:
        page = FakePage()

        def close(self) -> None:
            return None

    class FakeStore:
        def load_browser_metadata(self):
            return {
                "browser": "chromium",
                "executable_path": "",
                "profile_path": "/private/browser-profile",
            }

        def resolve_profile_path(self, **_kwargs):
            return Path("/private/browser-profile")

        def record_runtime_failure(self, *_args, **_kwargs) -> None:
            raise AssertionError("unexpected runtime failure")

    now = datetime(2026, 7, 29, 10, 0, tzinfo=timezone.utc)

    def enrich(_page, record, **_kwargs):
        return record.model_copy(
            update={
                "posted_at_text": "2 hours ago",
                "published_at": now,
                "freshness_confidence": "medium",
                "is_within_24_hours": True,
            }
        )

    session = FakeSession()
    with (
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.open_persistent_authenticated_context",
            return_value=session,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._wait_for_search_results_hydration",
            return_value="results",
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.build_linkedin_search_queries",
            return_value=[
                ("Primary @ Japan", "https://www.linkedin.com/jobs/search/?keywords=AI")
            ],
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._parse_linkedin_jobs_html_with_diagnostics",
            return_value=(
                candidates,
                LinkedInParseDiagnostics(parseable_candidate_count=3),
            ),
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._safe_job_card_link",
            return_value=MagicMock(),
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._enrich_job_detail_via_panel",
            side_effect=enrich,
        ) as panel_detail,
    ):
        records, rejected, _timings, _warnings, _queries = scrape_linkedin_jobs(
            LinkedInJobsRequest(
                query="AI jobs Japan",
                locations=["Japan"],
                max_results=5,
                include_description=False,
            ),
            session_store=FakeStore(),  # type: ignore[arg-type]
        )

    assert [record.linkedin_job_id for record in records] == ["111", "222"]
    assert [item.reason for item in rejected] == ["detail_budget_exhausted"]
    assert panel_detail.call_count == 2
    assert any(wait >= 1100 for wait in session.page.waits)



def test_linkedin_scraper_detail_budget_stops_queries_and_classifies_dates(
    monkeypatch,
):
    from features.web_scraping.domain.linkedin_models import (
        LinkedInJobsRequest,
        LinkedInParseDiagnostics,
        LinkedInVacancyRecord,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        scrape_linkedin_jobs,
    )

    monkeypatch.setenv("LINKEDIN_DETAIL_BUDGET", "3")
    monkeypatch.setenv("LINKEDIN_DIRECT_DETAIL_FALLBACK", "true")
    monkeypatch.setenv("LINKEDIN_MAX_QUERIES_PER_LOCATION", "2")

    def candidate(job_id: str) -> LinkedInVacancyRecord:
        return LinkedInVacancyRecord(
            linkedin_job_id=job_id,
            title=f"AI Engineer {job_id}",
            canonical_url=f"https://www.linkedin.com/jobs/view/{job_id}",
            source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
            matched_terms=["ai"],
        )

    candidates = [candidate("111"), candidate("222"), candidate("333")]

    class FakeStore:
        def load_browser_metadata(self):
            return {
                "browser": "chromium",
                "executable_path": "",
                "profile_path": "/private/browser-profile",
            }

        def resolve_profile_path(self, **_kwargs):
            return Path("/private/browser-profile")

        def record_runtime_failure(self, *_args, **_kwargs) -> None:
            raise AssertionError("unexpected runtime failure")

    class FakePage:
        url = "https://www.linkedin.com/jobs/"

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url

        def wait_for_timeout(self, _milliseconds: int) -> None:
            return None

        def content(self) -> str:
            return "<html></html>"

    class FakeSession:
        page = FakePage()

        def close(self) -> None:
            return None

    now = datetime(2026, 7, 28, 14, 0, tzinfo=timezone.utc)

    def enrich(_page, record, **_kwargs):
        if record.linkedin_job_id == "111":
            return record.model_copy(
                update={
                    "posted_at_text": "2 hours ago",
                    "published_at": now,
                    "freshness_confidence": "medium",
                    "is_within_24_hours": True,
                }
            )
        if record.linkedin_job_id == "222":
            return record.model_copy(
                update={
                    "posted_at_text": "2 days ago",
                    "published_at": now.replace(day=26),
                    "freshness_confidence": "medium",
                    "is_within_24_hours": False,
                }
            )
        return record

    with (
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.open_persistent_authenticated_context",
            return_value=FakeSession(),
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.build_linkedin_search_queries",
            return_value=[
                ("AI Engineer @ Japan", "https://www.linkedin.com/jobs/search/?keywords=AI"),
                ("ML Engineer @ Japan", "https://www.linkedin.com/jobs/search/?keywords=ML"),
            ],
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._parse_linkedin_jobs_html_with_diagnostics",
            return_value=(
                candidates,
                LinkedInParseDiagnostics(parseable_candidate_count=3),
            ),
        ) as parse_html,
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._enrich_job_detail",
            side_effect=enrich,
        ) as enrich_detail,
    ):
        records, rejected, timings, warnings, queries = scrape_linkedin_jobs(
            LinkedInJobsRequest(
                query="vacantes AI en Japón",
                locations=["Japan"],
                max_results=5,
                include_description=False,
            ),
            session_store=FakeStore(),  # type: ignore[arg-type]
        )

    assert [record.linkedin_job_id for record in records] == ["111"]
    assert [item.reason for item in rejected] == [
        "duplicate",
        "duplicate",
        "duplicate",
        "outside_24_hours",
        "unverified_posted_date",
    ]
    assert len(queries) == 2
    assert len(timings) == 2
    assert parse_html.call_count == 2
    assert enrich_detail.call_count == 3
    assert not any(
        warning.startswith("location_early_stop:")
        for warning in warnings
    )


def test_linkedin_scraper_recovers_page_and_opens_network_circuits(
    monkeypatch,
):
    from features.web_scraping.domain.linkedin_models import (
        LinkedInJobsRequest,
        LinkedInParseDiagnostics,
        LinkedInVacancyRecord,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        scrape_linkedin_jobs,
    )

    monkeypatch.setenv("LINKEDIN_DETAIL_BUDGET", "12")
    monkeypatch.setenv("LINKEDIN_DIRECT_DETAIL_FALLBACK", "true")
    monkeypatch.setenv("LINKEDIN_MAX_QUERIES_PER_LOCATION", "2")

    class PlaywrightError(Exception):
        pass

    def candidate(job_id: str, relevant: bool) -> LinkedInVacancyRecord:
        return LinkedInVacancyRecord(
            linkedin_job_id=job_id,
            title=("AI Engineer" if relevant else "Sales Manager"),
            canonical_url=f"https://www.linkedin.com/jobs/view/{job_id}",
            source_url=(
                "https://www.linkedin.com/jobs/search/?keywords=AI"
                "&location=South+Korea"
            ),
            matched_terms=["ai"] if relevant else [],
        )

    candidates = [
        candidate("111", True),
        candidate("222", True),
        candidate("333", True),
        candidate("444", False),
        candidate("555", False),
        candidate("666", False),
        candidate("777", False),
    ]
    navigation_state = {"first_query_completed": False}

    class FakePage:
        url = "https://www.linkedin.com/jobs/"

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url
            if "/jobs/search/" not in url:
                return
            if not navigation_state["first_query_completed"]:
                navigation_state["first_query_completed"] = True
                return
            raise PlaywrightError(
                "Page.goto: net::ERR_CONNECTION_RESET at "
                "https://www.linkedin.com/jobs/search/?token=secret"
            )

        def wait_for_timeout(self, _milliseconds: int) -> None:
            return None

        def content(self) -> str:
            return "<html></html>"

        def close(self) -> None:
            return None

    class FakeSession:
        def __init__(self) -> None:
            self.page = FakePage()
            self.replace_count = 0

        def replace_page(self):
            self.replace_count += 1
            self.page = FakePage()
            return self.page

        def close(self) -> None:
            return None

    class FakeStore:
        def load_browser_metadata(self):
            return {
                "browser": "chromium",
                "executable_path": "",
                "profile_path": "/private/browser-profile",
            }

        def resolve_profile_path(self, **_kwargs):
            return Path("/private/browser-profile")

        def record_runtime_failure(self, *_args, **_kwargs) -> None:
            raise AssertionError("unexpected runtime failure")

    session = FakeSession()
    query_plan = [
        (
            f"Role {index} @ {'South Korea' if index <= 2 else 'Japan'}",
            f"https://www.linkedin.com/jobs/search/?keywords=role{index}",
        )
        for index in range(1, 5)
    ]
    with (
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.open_persistent_authenticated_context",
            return_value=session,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.build_linkedin_search_queries",
            return_value=query_plan,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._parse_linkedin_jobs_html_with_diagnostics",
            return_value=(
                candidates,
                LinkedInParseDiagnostics(parseable_candidate_count=7),
            ),
        ) as parse_html,
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._enrich_job_detail",
            side_effect=PlaywrightError(
                "Page.goto: net::ERR_NAME_NOT_RESOLVED at "
                "https://www.linkedin.com/jobs/view/secret"
            ),
        ) as enrich_detail,
    ):
        records, rejected, timings, warnings, queries = scrape_linkedin_jobs(
            LinkedInJobsRequest(
                query="vacantes AI en Corea del Sur y Japón",
                locations=["South Korea", "Japan"],
                max_results=5,
                include_description=True,
            ),
            session_store=FakeStore(),  # type: ignore[arg-type]
        )

    assert records == []
    assert len(queries) == 4
    assert len(timings) == 4
    assert parse_html.call_count == 1
    enrich_detail.assert_not_called()
    assert session.replace_count == 0
    assert [item.reason for item in rejected].count("low_topic_relevance") == 4
    assert [item.reason for item in rejected].count("detail_network_failure") == 3
    assert "detail_budget_exhausted" not in {
        item.reason for item in rejected
    }
    assert "unverified_posted_date" not in {
        item.reason for item in rejected
    }
    assert any(
        warning.startswith("query_location_stopped:South Korea:")
        for warning in warnings
    )
    assert any(
        warning.startswith("query_location_stopped:Japan:")
        for warning in warnings
    )
    assert any(
        warning.startswith("detail_location_circuit_open:")
        for warning in warnings
    )
    serialized_warnings = json.dumps(warnings)
    assert "token=secret" not in serialized_warnings
    assert "/jobs/" not in serialized_warnings


def test_linkedin_http_failure_opens_immediate_circuit_and_preserves_candidates(
    monkeypatch,
):
    from features.web_scraping.domain.linkedin_models import (
        LinkedInJobsRequest,
        LinkedInParseDiagnostics,
        LinkedInVacancyRecord,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        scrape_linkedin_jobs,
    )

    monkeypatch.setenv("LINKEDIN_MAX_QUERIES_PER_LOCATION", "2")
    monkeypatch.setenv("LINKEDIN_QUERY_INTERVAL_MS", "2500")
    monkeypatch.setenv("LINKEDIN_DIRECT_DETAIL_FALLBACK", "true")

    recent = LinkedInVacancyRecord(
        linkedin_job_id="111",
        title="AI Engineer",
        posted_at_text="Hace 2 horas",
        published_at=datetime(2026, 7, 29, 10, 0, tzinfo=timezone.utc),
        freshness_confidence="medium",
        is_within_24_hours=True,
        canonical_url="https://www.linkedin.com/jobs/view/111",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        matched_terms=["ai"],
    )
    unresolved = [
        LinkedInVacancyRecord(
            linkedin_job_id=job_id,
            title="Machine Learning Engineer",
            canonical_url=f"https://www.linkedin.com/jobs/view/{job_id}",
            source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
            matched_terms=["machine learning"],
        )
        for job_id in ("222", "333")
    ]

    class PlaywrightError(Exception):
        pass

    class FakePage:
        url = "https://www.linkedin.com/jobs/"

        def __init__(self, session) -> None:
            self.session = session
            self.search_count = 0
            self.waits = []

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url
            if "/jobs/search/" not in url:
                return
            self.search_count += 1
            if self.search_count == 2:
                self.session.alive = False
                raise PlaywrightError(
                    "Page.goto: net::ERR_HTTP_RESPONSE_CODE_FAILURE at "
                    "https://www.linkedin.com/jobs/search/?token=secret"
                )

        def wait_for_timeout(self, milliseconds: int) -> None:
            self.waits.append(milliseconds)

        def content(self) -> str:
            return "<html></html>"

    class FakeSession:
        def __init__(self) -> None:
            self.alive = True
            self.page = FakePage(self)
            self.replace_count = 0

        def page_is_alive(self) -> bool:
            return self.alive

        def replace_page(self):
            self.replace_count += 1
            raise AssertionError("HTTP circuit must not replace Page")

        def close(self) -> None:
            return None

    class FakeStore:
        def load_browser_metadata(self):
            return {
                "browser": "chromium",
                "executable_path": "",
                "profile_path": "/private/browser-profile",
            }

        def resolve_profile_path(self, **_kwargs):
            return Path("/private/browser-profile")

        def record_runtime_failure(self, *_args, **_kwargs) -> None:
            raise AssertionError("unexpected runtime failure")

    session = FakeSession()
    with (
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.open_persistent_authenticated_context",
            return_value=session,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._wait_for_search_results_hydration",
            return_value="results",
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.build_linkedin_search_queries",
            return_value=[
                (
                    "AI/ML/Data/GenAI @ South Korea",
                    "https://www.linkedin.com/jobs/search/?keywords=primary",
                ),
                (
                    "AI Agents/Product/Architecture @ South Korea",
                    "https://www.linkedin.com/jobs/search/?keywords=fallback",
                ),
            ],
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._parse_linkedin_jobs_html_with_diagnostics",
            return_value=(
                [recent, *unresolved],
                LinkedInParseDiagnostics(parseable_candidate_count=3),
            ),
        ) as parse_html,
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._enrich_job_detail"
        ) as enrich_detail,
    ):
        records, rejected, timings, warnings, queries = scrape_linkedin_jobs(
            LinkedInJobsRequest(
                query="vacantes AI en Corea del Sur",
                locations=["South Korea"],
                max_results=5,
                include_description=False,
            ),
            session_store=FakeStore(),  # type: ignore[arg-type]
        )

    assert [record.linkedin_job_id for record in records] == ["111"]
    assert [item.reason for item in rejected] == [
        "detail_network_failure",
        "detail_network_failure",
    ]
    assert len(queries) == 2
    assert len(timings) == 2
    assert parse_html.call_count == 1
    enrich_detail.assert_not_called()
    assert session.replace_count == 2
    assert (
        "query_location_circuit_open:South Korea:"
        "http_response_code_failure"
    ) in warnings
    assert any(
        warning.startswith("detail_runtime_session_unavailable:")
        for warning in warnings
    )
    assert not any(
        warning.startswith("query_budget_exhausted")
        for warning in warnings
    )
    assert any(
        warning.startswith("page_recovery_failed:query:South Korea:")
        for warning in warnings
    )
    assert any(wait >= 2000 for wait in session.page.waits)
    assert "token=secret" not in json.dumps(warnings)


def test_linkedin_open_degraded_page_is_replaced_and_same_query_retried_once(
    monkeypatch,
):
    from features.web_scraping.domain.linkedin_models import (
        LinkedInJobsRequest,
        LinkedInParseDiagnostics,
        LinkedInVacancyRecord,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        scrape_linkedin_jobs,
    )

    monkeypatch.setenv("LINKEDIN_MAX_QUERIES_PER_LOCATION", "1")
    monkeypatch.setenv("LINKEDIN_QUERY_INTERVAL_MS", "2000")
    navigation_urls: list[str] = []

    class PlaywrightError(Exception):
        pass

    class FakePage:
        url = "https://www.linkedin.com/jobs/"

        def __init__(self, *, degraded: bool) -> None:
            self.degraded = degraded

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url
            if "/jobs/search/" not in url:
                return
            navigation_urls.append(url)
            if self.degraded:
                raise PlaywrightError(
                    "Page.goto: net::ERR_HTTP_RESPONSE_CODE_FAILURE at "
                    "https://www.linkedin.com/jobs/search/?token=secret"
                )

        def wait_for_timeout(self, _milliseconds: int) -> None:
            return None

        def content(self) -> str:
            return "<html></html>"

    class FakeSession:
        def __init__(self) -> None:
            self.page = FakePage(degraded=True)
            self.replace_count = 0

        def page_is_alive(self) -> bool:
            return True

        def replace_page(self):
            self.replace_count += 1
            self.page = FakePage(degraded=False)
            return self.page

        def close(self) -> None:
            return None

    class FakeStore:
        def load_browser_metadata(self):
            return {
                "browser": "chromium",
                "executable_path": "",
                "profile_path": "/private/browser-profile",
            }

        def resolve_profile_path(self, **_kwargs):
            return Path("/private/browser-profile")

        def record_runtime_failure(self, *_args, **_kwargs) -> None:
            raise AssertionError("unexpected runtime failure")

    source_url = "https://www.linkedin.com/jobs/search/?keywords=primary"
    candidate = LinkedInVacancyRecord(
        linkedin_job_id="101",
        title="AI Engineer",
        company_name="Example",
        location="South Korea",
        workplace_type="hybrid",
        posted_at_text="Hace 1 hora",
        published_at=datetime(2026, 7, 29, 10, 0, tzinfo=timezone.utc),
        freshness_confidence="medium",
        is_within_24_hours=True,
        canonical_url="https://www.linkedin.com/jobs/view/101",
        source_url=source_url,
        matched_terms=["ai"],
        hard_skills=["Python"],
        visa_status="no_sponsorship",
    )
    session = FakeSession()
    with (
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.open_persistent_authenticated_context",
            return_value=session,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._wait_for_search_results_hydration",
            return_value="results",
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.build_linkedin_search_queries",
            return_value=[("Primary @ South Korea", source_url)],
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._parse_linkedin_jobs_html_with_diagnostics",
            return_value=(
                [candidate],
                LinkedInParseDiagnostics(parseable_candidate_count=1),
            ),
        ),
    ):
        records, rejected, timings, warnings, queries = scrape_linkedin_jobs(
            LinkedInJobsRequest(
                query="AI jobs",
                locations=["South Korea"],
                max_results=5,
                include_description=False,
            ),
            session_store=FakeStore(),  # type: ignore[arg-type]
        )

    assert [record.linkedin_job_id for record in records] == ["101"]
    assert rejected == []
    assert queries == [source_url]
    assert navigation_urls == [source_url, source_url]
    assert len(timings) == 1
    assert timings[0].error == ""
    assert session.replace_count == 1
    assert (
        "query_navigation_retry:South Korea:http_response_code_failure"
        in warnings
    )
    assert not any(
        warning.startswith("query_location_circuit_open:South Korea:")
        for warning in warnings
    )
    assert "token=secret" not in json.dumps(warnings)


def test_linkedin_query_retry_does_not_exceed_global_hard_cap(monkeypatch):
    from features.web_scraping.domain.linkedin_models import LinkedInJobsRequest
    from features.web_scraping.infrastructure import linkedin_scraper

    monkeypatch.setenv("LINKEDIN_MAX_QUERIES_PER_LOCATION", "1")
    monkeypatch.setenv("LINKEDIN_QUERY_INTERVAL_MS", "2000")
    monkeypatch.setattr(linkedin_scraper, "_HARD_MAX_TOTAL_QUERY_ATTEMPTS", 1)
    navigation_count = 0

    class PlaywrightError(Exception):
        pass

    class FakePage:
        url = "https://www.linkedin.com/jobs/"

        def goto(self, url: str, **_kwargs) -> None:
            nonlocal navigation_count
            self.url = url
            if "/jobs/search/" not in url:
                return
            navigation_count += 1
            raise PlaywrightError(
                "Page.goto: net::ERR_HTTP_RESPONSE_CODE_FAILURE at "
                "https://www.linkedin.com/jobs/search/?token=secret"
            )

        def wait_for_timeout(self, _milliseconds: int) -> None:
            return None

    class FakeSession:
        def __init__(self) -> None:
            self.page = FakePage()
            self.replace_count = 0

        def page_is_alive(self) -> bool:
            return True

        def replace_page(self):
            self.replace_count += 1
            self.page = FakePage()
            return self.page

        def close(self) -> None:
            return None

    class FakeStore:
        def load_browser_metadata(self):
            return {
                "browser": "chromium",
                "executable_path": "",
                "profile_path": "/private/browser-profile",
            }

        def resolve_profile_path(self, **_kwargs):
            return Path("/private/browser-profile")

        def record_runtime_failure(self, *_args, **_kwargs) -> None:
            raise AssertionError("unexpected runtime failure")

    session = FakeSession()
    with (
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.open_persistent_authenticated_context",
            return_value=session,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.build_linkedin_search_queries",
            return_value=[
                (
                    "Primary @ South Korea",
                    "https://www.linkedin.com/jobs/search/?keywords=primary",
                )
            ],
        ),
    ):
        _records, _rejected, _timings, warnings, _queries = (
            linkedin_scraper.scrape_linkedin_jobs(
                LinkedInJobsRequest(
                    query="AI jobs",
                    locations=["South Korea"],
                    max_results=5,
                    include_description=False,
                ),
                session_store=FakeStore(),  # type: ignore[arg-type]
            )
        )

    assert navigation_count == 1
    assert session.replace_count == 0
    assert not any(
        warning.startswith("query_navigation_retry:")
        for warning in warnings
    )
    assert (
        "query_location_circuit_open:South Korea:"
        "http_response_code_failure"
    ) in warnings
    assert "token=secret" not in json.dumps(warnings)


def test_linkedin_failed_goto_probe_recovers_candidates_once(monkeypatch):
    from features.web_scraping.domain.linkedin_models import (
        LinkedInJobsRequest,
        LinkedInParseDiagnostics,
        LinkedInVacancyRecord,
    )
    from features.web_scraping.infrastructure import linkedin_scraper

    monkeypatch.setenv("LINKEDIN_DETAIL_BUDGET", "0")
    monkeypatch.setenv("LINKEDIN_MAX_QUERIES_PER_LOCATION", "1")
    monkeypatch.setenv("LINKEDIN_QUERY_INTERVAL_MS", "2000")
    navigation_count = 0
    probe_count = 0
    source_url = "https://www.linkedin.com/jobs/search/?keywords=AI"
    body = """
    <li data-occludable-job-id="777">
      <a class="job-card-list__title" href="/jobs/view/777">AI Engineer</a>
      <time datetime="2026-07-29T12:00:00Z">Hace 1 hora</time>
    </li>
    """

    class PlaywrightError(Exception):
        pass

    class FakeResponse:
        status = 200
        url = source_url

        def __init__(self):
            self.disposed = False

        def text(self):
            return body

        def dispose(self):
            self.disposed = True

    response = FakeResponse()
    recovered_record = LinkedInVacancyRecord(
        linkedin_job_id="777",
        title="AI Engineer",
        posted_at_text="Hace 1 hora",
        published_at=datetime.now(timezone.utc),
        freshness_confidence="medium",
        is_within_24_hours=True,
        canonical_url="https://www.linkedin.com/jobs/view/777",
        source_url=source_url,
        matched_terms=["ai"],
    )

    class FakeRequest:
        def get(self, url, **kwargs):
            nonlocal probe_count
            probe_count += 1
            assert url == source_url
            assert kwargs["fail_on_status_code"] is False
            return response

    class FakePage:
        url = "https://www.linkedin.com/jobs/"

        def goto(self, url, **_kwargs):
            nonlocal navigation_count
            self.url = url
            if "/jobs/search/" in url:
                navigation_count += 1
                raise PlaywrightError(
                    "Page.goto: net::ERR_HTTP_RESPONSE_CODE_FAILURE at "
                    "https://www.linkedin.com/jobs/search/?token=secret"
                )

        def wait_for_timeout(self, _milliseconds):
            return None

        def is_closed(self):
            return False

    class FakeSession:
        def __init__(self):
            self.page = FakePage()
            self.context = SimpleNamespace(request=FakeRequest())
            self.replace_count = 0

        def page_is_alive(self):
            return True

        def replace_page(self):
            self.replace_count += 1
            self.page = FakePage()
            return self.page

        def close(self):
            return None

    class FakeStore:
        def load_browser_metadata(self):
            return {
                "browser": "chromium",
                "executable_path": "",
                "profile_path": "/private/browser-profile",
            }

        def resolve_profile_path(self, **_kwargs):
            return Path("/private/browser-profile")

        def record_runtime_failure(self, *_args, **_kwargs):
            raise AssertionError("unexpected runtime failure")

    session = FakeSession()
    with (
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.open_persistent_authenticated_context",
            return_value=session,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.build_linkedin_search_queries",
            return_value=[("Primary @ Japan", source_url)],
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._parse_linkedin_jobs_html_with_diagnostics",
            return_value=(
                [recovered_record],
                LinkedInParseDiagnostics(
                    href_count=1,
                    candidate_count=1,
                    parseable_candidate_count=1,
                ),
            ),
        ),
    ):
        records, _rejected, timings, warnings, _queries = (
            linkedin_scraper.scrape_linkedin_jobs(
                LinkedInJobsRequest(
                    query="AI jobs",
                    locations=["Japan"],
                    max_results=5,
                    include_description=False,
                ),
                session_store=FakeStore(),  # type: ignore[arg-type]
            )
        )

    assert navigation_count == 2
    assert session.replace_count == 1
    assert probe_count == 1
    assert response.disposed is True
    assert [record.linkedin_job_id for record in records] == ["777"]
    assert timings[0].diagnostics.parseable_candidate_count == 1
    assert "query_probe_result:Japan:status_200:ok" in warnings
    assert "token=secret" not in json.dumps(warnings)
    assert body not in json.dumps(warnings)


def test_linkedin_probe_runs_once_and_consumes_global_hard_cap(monkeypatch):
    from features.web_scraping.domain.linkedin_models import LinkedInJobsRequest
    from features.web_scraping.infrastructure import linkedin_scraper

    monkeypatch.setenv("LINKEDIN_DETAIL_BUDGET", "0")
    monkeypatch.setenv("LINKEDIN_MAX_QUERIES_PER_LOCATION", "1")
    monkeypatch.setenv("LINKEDIN_QUERY_INTERVAL_MS", "2000")
    monkeypatch.setattr(linkedin_scraper, "_HARD_MAX_TOTAL_QUERY_ATTEMPTS", 3)
    navigation_count = 0
    probe_count = 0

    class PlaywrightError(Exception):
        pass

    class FakeResponse:
        status = 429
        url = "https://www.linkedin.com/jobs/search/?keywords=AI"

        def __init__(self):
            self.disposed = False

        def text(self):
            raise AssertionError("429 body must not be read")

        def dispose(self):
            self.disposed = True

    response = FakeResponse()

    class FakeRequest:
        def get(self, *_args, **_kwargs):
            nonlocal probe_count
            probe_count += 1
            return response

    class FakePage:
        url = "https://www.linkedin.com/jobs/"

        def goto(self, url, **_kwargs):
            nonlocal navigation_count
            self.url = url
            if "/jobs/search/" in url:
                navigation_count += 1
                raise PlaywrightError(
                    "Page.goto: net::ERR_HTTP_RESPONSE_CODE_FAILURE"
                )

        def wait_for_timeout(self, _milliseconds):
            return None

        def is_closed(self):
            return False

    class FakeSession:
        def __init__(self):
            self.page = FakePage()
            self.context = SimpleNamespace(request=FakeRequest())

        def page_is_alive(self):
            return True

        def replace_page(self):
            self.page = FakePage()
            return self.page

        def close(self):
            return None

    class FakeStore:
        def load_browser_metadata(self):
            return {
                "browser": "chromium",
                "executable_path": "",
                "profile_path": "/private/browser-profile",
            }

        def resolve_profile_path(self, **_kwargs):
            return Path("/private/browser-profile")

        def record_runtime_failure(self, *_args, **_kwargs):
            raise AssertionError("unexpected runtime failure")

    with (
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.open_persistent_authenticated_context",
            return_value=FakeSession(),
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.build_linkedin_search_queries",
            return_value=[
                (
                    "Primary @ Japan",
                    "https://www.linkedin.com/jobs/search/?keywords=AI",
                ),
                (
                    "Primary @ South Korea",
                    "https://www.linkedin.com/jobs/search/?keywords=ML",
                ),
            ],
        ),
    ):
        _records, _rejected, _timings, warnings, queries = (
            linkedin_scraper.scrape_linkedin_jobs(
                LinkedInJobsRequest(
                    query="AI jobs",
                    locations=["Japan", "South Korea"],
                    max_results=5,
                    include_description=False,
                ),
                session_store=FakeStore(),  # type: ignore[arg-type]
            )
        )

    assert navigation_count == 2
    assert probe_count == 1
    assert response.disposed is True
    assert len(queries) == 1
    assert "query_global_budget_exhausted:3" in warnings
    assert (
        "query_probe_result:Japan:status_429:query_rate_limited"
        in warnings
    )


@pytest.mark.parametrize(
    ("status_code", "expected_category"),
    [
        (429, "query_rate_limited"),
        (403, "query_access_rejected"),
        (999, "query_access_rejected"),
        (503, "query_upstream_failure"),
        (418, "query_navigation_failure"),
    ],
)
def test_linkedin_authenticated_probe_classifies_status_and_disposes(
    status_code,
    expected_category,
):
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _probe_linkedin_search_with_authenticated_request,
    )

    class FakeResponse:
        url = "https://www.linkedin.com/jobs/search/?keywords=AI"

        def __init__(self):
            self.status = status_code
            self.disposed = False

        def text(self):
            raise AssertionError("non-2xx body must not be read")

        def dispose(self):
            self.disposed = True

    response = FakeResponse()
    session = SimpleNamespace(
        context=SimpleNamespace(
            request=SimpleNamespace(get=lambda *_args, **_kwargs: response)
        )
    )

    result = _probe_linkedin_search_with_authenticated_request(
        session,
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
    )

    assert result.status_code == status_code
    assert result.category == expected_category
    assert response.disposed is True


def test_linkedin_authenticated_probe_rejects_final_url_and_login_redirect():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        LinkedInAuthRequiredError,
        _probe_linkedin_search_with_authenticated_request,
    )

    class FakeResponse:
        status = 200

        def __init__(self, url):
            self.url = url
            self.disposed = False

        def text(self):
            return "<html>token=secret</html>"

        def dispose(self):
            self.disposed = True

    external = FakeResponse("https://example.com/jobs/search/")
    external_session = SimpleNamespace(
        context=SimpleNamespace(
            request=SimpleNamespace(get=lambda *_args, **_kwargs: external)
        )
    )
    result = _probe_linkedin_search_with_authenticated_request(
        external_session,
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
    )
    assert result.category == "query_navigation_failure"
    assert result.detail == "final_url_rejected"
    assert external.disposed is True

    login = FakeResponse("https://www.linkedin.com/login")
    login_session = SimpleNamespace(
        context=SimpleNamespace(
            request=SimpleNamespace(get=lambda *_args, **_kwargs: login)
        )
    )
    with pytest.raises(LinkedInAuthRequiredError):
        _probe_linkedin_search_with_authenticated_request(
            login_session,
            source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        )
    assert login.disposed is True


def test_linkedin_authenticated_probe_disposes_when_body_read_fails():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _probe_linkedin_search_with_authenticated_request,
    )

    class FakeResponse:
        status = 200
        url = "https://www.linkedin.com/jobs/search/?keywords=AI"

        def __init__(self):
            self.disposed = False

        def text(self):
            raise RuntimeError("body token=secret")

        def dispose(self):
            self.disposed = True

    response = FakeResponse()
    session = SimpleNamespace(
        context=SimpleNamespace(
            request=SimpleNamespace(get=lambda *_args, **_kwargs: response)
        )
    )
    result = _probe_linkedin_search_with_authenticated_request(
        session,
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
    )

    assert result.category == "query_navigation_failure"
    assert result.detail == "body_parse_failed"
    assert response.disposed is True
    assert "secret" not in repr(result)


def test_linkedin_authenticated_probe_2xx_without_cards_is_navigation_incomplete():
    from features.web_scraping.domain.linkedin_models import (
        LinkedInParseDiagnostics,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _probe_linkedin_search_with_authenticated_request,
    )

    class FakeResponse:
        status = 200
        url = "https://www.linkedin.com/jobs/search/?keywords=AI"

        def __init__(self):
            self.disposed = False

        def text(self):
            return "<html><body>No cards rendered</body></html>"

        def dispose(self):
            self.disposed = True

    response = FakeResponse()
    session = SimpleNamespace(
        context=SimpleNamespace(
            request=SimpleNamespace(get=lambda *_args, **_kwargs: response)
        )
    )
    with patch(
        "features.web_scraping.infrastructure.linkedin_scraper._parse_linkedin_jobs_html_with_diagnostics",
        return_value=([], LinkedInParseDiagnostics()),
    ):
        result = _probe_linkedin_search_with_authenticated_request(
            session,
            source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        )

    assert result.category == "query_navigation_failure"
    assert result.detail == "no_cards"
    assert response.disposed is True


@pytest.mark.parametrize("failed_primary_location", ["South Korea", "Japan"])
def test_linkedin_query_circuit_isolated_by_country_and_round_robin_continues(
    monkeypatch,
    failed_primary_location,
):
    from features.web_scraping.domain.linkedin_models import (
        LinkedInJobsRequest,
        LinkedInParseDiagnostics,
        LinkedInVacancyRecord,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        scrape_linkedin_jobs,
    )

    monkeypatch.setenv("LINKEDIN_DETAIL_BUDGET", "0")
    monkeypatch.setenv("LINKEDIN_MAX_QUERIES_PER_LOCATION", "2")
    monkeypatch.setenv("LINKEDIN_QUERY_INTERVAL_MS", "2000")
    monkeypatch.setenv("LINKEDIN_DIRECT_DETAIL_FALLBACK", "false")

    query_plan = [
        (
            "AI/ML/Data/GenAI @ South Korea",
            "https://www.linkedin.com/jobs/search/?keywords=kr-primary",
        ),
        (
            "AI/ML/Data/GenAI @ Japan",
            "https://www.linkedin.com/jobs/search/?keywords=jp-primary",
        ),
        (
            "AI Agents/Product/Architecture @ South Korea",
            "https://www.linkedin.com/jobs/search/?keywords=kr-complementary",
        ),
        (
            "AI Agents/Product/Architecture @ Japan",
            "https://www.linkedin.com/jobs/search/?keywords=jp-complementary",
        ),
    ]
    failed_token = (
        "kr-primary"
        if failed_primary_location == "South Korea"
        else "jp-primary"
    )
    navigation_order: list[str] = []

    class PlaywrightError(Exception):
        pass

    class FakePage:
        url = "https://www.linkedin.com/jobs/"

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url
            if "/jobs/search/" not in url:
                return
            navigation_order.append(url)
            if failed_token in url:
                raise PlaywrightError(
                    "Page.goto: net::ERR_HTTP_RESPONSE_CODE_FAILURE at "
                    "https://www.linkedin.com/jobs/search/?token=secret"
                )

        def wait_for_timeout(self, _milliseconds: int) -> None:
            return None

        def content(self) -> str:
            return "<html></html>"

        def is_closed(self) -> bool:
            return False

    class FakeSession:
        def __init__(self) -> None:
            self.page = FakePage()
            self.replace_count = 0

        def page_is_alive(self) -> bool:
            return True

        def replace_page(self):
            self.replace_count += 1
            self.page = FakePage()
            return self.page

        def close(self) -> None:
            return None

    class FakeStore:
        def load_browser_metadata(self):
            return {
                "browser": "chromium",
                "executable_path": "",
                "profile_path": "/private/browser-profile",
            }

        def resolve_profile_path(self, **_kwargs):
            return Path("/private/browser-profile")

        def record_runtime_failure(self, *_args, **_kwargs) -> None:
            raise AssertionError("unexpected runtime failure")

    def parse_results(_html, *, source_url, now):
        del _html, now
        token = source_url.split("keywords=", 1)[-1]
        job_id = {
            "kr-primary": "101",
            "jp-primary": "201",
            "kr-complementary": "102",
            "jp-complementary": "202",
        }[token]
        location = "South Korea" if token.startswith("kr-") else "Japan"
        return (
            [
                LinkedInVacancyRecord(
                    linkedin_job_id=job_id,
                    title=f"AI Engineer {location} {token}",
                    location=location,
                    posted_at_text="Hace 1 hora",
                    published_at=datetime(
                        2026,
                        7,
                        29,
                        10,
                        0,
                        tzinfo=timezone.utc,
                    ),
                    freshness_confidence="medium",
                    is_within_24_hours=True,
                    canonical_url=(
                        f"https://www.linkedin.com/jobs/view/{job_id}"
                    ),
                    source_url=source_url,
                    matched_terms=["ai"],
                )
            ],
            LinkedInParseDiagnostics(parseable_candidate_count=1),
        )

    session = FakeSession()
    with (
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.open_persistent_authenticated_context",
            return_value=session,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._wait_for_search_results_hydration",
            return_value="results",
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.build_linkedin_search_queries",
            return_value=query_plan,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._parse_linkedin_jobs_html_with_diagnostics",
            side_effect=parse_results,
        ),
    ):
        records, rejected, timings, warnings, queries = scrape_linkedin_jobs(
            LinkedInJobsRequest(
                query="vacantes AI en Corea del Sur y Japón",
                locations=["South Korea", "Japan"],
                max_results=10,
                include_description=False,
            ),
            session_store=FakeStore(),  # type: ignore[arg-type]
        )

    expected_plan = [
        item
        for item in query_plan
        if not (
            failed_primary_location in item[0]
            and "AI Agents/Product/Architecture" in item[0]
        )
    ]
    assert queries == [url for _, url in expected_plan]
    failed_url = next(url for _, url in query_plan if failed_token in url)
    assert navigation_order.count(failed_url) == 2
    assert len(navigation_order) == len(expected_plan) + 1
    assert [timing.query for timing in timings] == [
        label for label, _ in expected_plan
    ]
    assert len(records) == 2
    assert rejected == []
    other_location = (
        "Japan" if failed_primary_location == "South Korea" else "South Korea"
    )
    assert {
        record.location for record in records
    } == {other_location}
    assert any(
        warning.startswith(
            f"query_location_circuit_open:{failed_primary_location}:"
        )
        for warning in warnings
    )
    assert not any(
        warning.startswith(
            f"query_location_circuit_open:{other_location}:"
        )
        for warning in warnings
    )
    assert not any(
        warning.startswith("query_global_budget_exhausted:")
        for warning in warnings
    )
    assert session.replace_count == 1
    assert (
        f"query_navigation_retry:{failed_primary_location}:"
        "http_response_code_failure"
    ) in warnings
    assert "token=secret" not in json.dumps(warnings)


def test_linkedin_query_global_budget_hard_caps_multi_location_plan(
    monkeypatch,
):
    from features.web_scraping.domain.linkedin_models import (
        LinkedInJobsRequest,
        LinkedInParseDiagnostics,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        scrape_linkedin_jobs,
    )

    monkeypatch.setenv("LINKEDIN_MAX_QUERIES_PER_LOCATION", "2")
    monkeypatch.setenv("LINKEDIN_QUERY_INTERVAL_MS", "2000")
    monkeypatch.setenv("LINKEDIN_DIRECT_DETAIL_FALLBACK", "false")

    locations = ["South Korea", "Japan", "Singapore", "India", "Australia"]
    query_plan = [
        (
            f"Plan {plan_index} @ {location}",
            (
                "https://www.linkedin.com/jobs/search/"
                f"?keywords={plan_index}-{location}"
            ),
        )
        for plan_index in (1, 2)
        for location in locations
    ]

    class FakePage:
        url = "https://www.linkedin.com/jobs/"

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url

        def wait_for_timeout(self, _milliseconds: int) -> None:
            return None

        def content(self) -> str:
            return "<html></html>"

        def is_closed(self) -> bool:
            return False

    class FakeSession:
        def __init__(self) -> None:
            self.page = FakePage()

        def page_is_alive(self) -> bool:
            return True

        def close(self) -> None:
            return None

    class FakeStore:
        def load_browser_metadata(self):
            return {
                "browser": "chromium",
                "executable_path": "",
                "profile_path": "/private/browser-profile",
            }

        def resolve_profile_path(self, **_kwargs):
            return Path("/private/browser-profile")

        def record_runtime_failure(self, *_args, **_kwargs) -> None:
            raise AssertionError("unexpected runtime failure")

    with (
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.open_persistent_authenticated_context",
            return_value=FakeSession(),
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._wait_for_search_results_hydration",
            return_value="results",
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.build_linkedin_search_queries",
            return_value=query_plan,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._parse_linkedin_jobs_html_with_diagnostics",
            return_value=(
                [],
                LinkedInParseDiagnostics(),
            ),
        ),
    ):
        records, rejected, timings, warnings, queries = scrape_linkedin_jobs(
            LinkedInJobsRequest(
                query="vacantes AI multirregión",
                locations=locations,
                max_results=50,
                include_description=False,
            ),
            session_store=FakeStore(),  # type: ignore[arg-type]
        )

    assert records == []
    assert rejected == []
    assert len(queries) == 8
    assert len(timings) == 8
    assert warnings[-1] == "query_global_budget_exhausted:8"


def test_linkedin_error_labels_are_sanitized_and_budgets_are_hard_bounded(
    monkeypatch,
):
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _respect_query_cadence,
        _safe_error_label,
        configured_linkedin_detail_click_interval_ms,
        configured_linkedin_detail_budget,
        configured_linkedin_direct_detail_fallback,
        configured_linkedin_max_queries_per_location,
        configured_linkedin_query_interval_ms,
    )

    label = _safe_error_label(
        RuntimeError("Navigation timeout for https://linkedin/?token=secret")
    )
    assert label == "RuntimeError:timeout"
    assert "secret" not in label
    assert "linkedin" not in label
    assert _safe_error_label(
        RuntimeError("net::ERR_NAME_NOT_RESOLVED at https://secret.example")
    ) == "RuntimeError:dns"
    assert _safe_error_label(
        RuntimeError("net::ERR_CONNECTION_RESET at https://secret.example")
    ) == "RuntimeError:connection"
    assert _safe_error_label(
        RuntimeError("net::ERR_ABORTED at https://secret.example")
    ) == "RuntimeError:network_aborted"
    unknown_label = _safe_error_label(
        RuntimeError(
            "Page.goto: net::ERR_FAILED at "
            "https://secret.example/?token=private"
        )
    )
    assert unknown_label == "RuntimeError:chromium_err_failed"
    assert "secret.example" not in unknown_label
    assert "private" not in unknown_label

    class TargetClosedError(Exception):
        pass

    assert _safe_error_label(
        TargetClosedError("Page.goto failed")
    ) == "TargetClosedError:target_closed"

    monkeypatch.setenv("LINKEDIN_DETAIL_BUDGET", "31")
    with pytest.raises(ValueError, match="entre 0 y 30"):
        configured_linkedin_detail_budget()

    monkeypatch.setenv("LINKEDIN_MAX_QUERIES_PER_LOCATION", "3")
    with pytest.raises(ValueError, match="entre 1 y 2"):
        configured_linkedin_max_queries_per_location()

    monkeypatch.setenv("LINKEDIN_QUERY_INTERVAL_MS", "1999")
    with pytest.raises(ValueError, match="entre 2000 y 5000"):
        configured_linkedin_query_interval_ms()

    monkeypatch.setenv("LINKEDIN_DETAIL_CLICK_INTERVAL_MS", "749")
    with pytest.raises(ValueError, match="entre 750 y 3000"):
        configured_linkedin_detail_click_interval_ms()

    monkeypatch.setenv("LINKEDIN_DIRECT_DETAIL_FALLBACK", "sometimes")
    with pytest.raises(ValueError, match="true o false"):
        configured_linkedin_direct_detail_fallback()

    page = MagicMock()
    _respect_query_cadence(
        page,
        last_successful_query_at=10.0,
        interval_ms=2750,
        now_fn=lambda: 10.25,
    )
    page.wait_for_timeout.assert_called_once_with(2500)


def test_linkedin_slow_query_cooldown_starts_after_parsing(monkeypatch):
    from features.web_scraping.domain.linkedin_models import (
        LinkedInJobsRequest,
        LinkedInParseDiagnostics,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        scrape_linkedin_jobs,
    )

    monkeypatch.setenv("LINKEDIN_QUERY_INTERVAL_MS", "2750")
    clock = SimpleNamespace(value=10.0)
    waits: list[int] = []

    class FakePage:
        url = "https://www.linkedin.com/jobs/"

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url
            if "/jobs/search/" in url:
                clock.value += 2.452

        def wait_for_timeout(self, milliseconds: int) -> None:
            waits.append(milliseconds)
            clock.value += milliseconds / 1000

        def content(self) -> str:
            return "<html></html>"

        def is_closed(self) -> bool:
            return False

    class FakeSession:
        page = FakePage()

        def page_is_alive(self) -> bool:
            return True

        def close(self) -> None:
            return None

    class FakeStore:
        def load_browser_metadata(self):
            return {
                "browser": "chromium",
                "executable_path": "",
                "profile_path": "/private/browser-profile",
            }

        def resolve_profile_path(self, **_kwargs):
            return Path("/private/browser-profile")

        def record_runtime_failure(self, *_args, **_kwargs) -> None:
            raise AssertionError("unexpected runtime failure")

    with (
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.time.monotonic",
            side_effect=lambda: clock.value,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.open_persistent_authenticated_context",
            return_value=FakeSession(),
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._wait_for_search_results_hydration",
            return_value="results",
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.build_linkedin_search_queries",
            return_value=[
                (
                    "Primary @ South Korea",
                    "https://www.linkedin.com/jobs/search/?keywords=kr",
                ),
                (
                    "Primary @ Japan",
                    "https://www.linkedin.com/jobs/search/?keywords=jp",
                ),
            ],
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._parse_linkedin_jobs_html_with_diagnostics",
            return_value=([], LinkedInParseDiagnostics()),
        ),
    ):
        scrape_linkedin_jobs(
            LinkedInJobsRequest(
                query="AI jobs",
                locations=["South Korea", "Japan"],
                max_results=5,
                include_description=False,
            ),
            session_store=FakeStore(),  # type: ignore[arg-type]
        )

    assert waits == [2750]


def test_linkedin_enrichment_source_navigation_obeys_cooldown_and_reuses_dom():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _SearchNavigationState,
        _ensure_search_source,
    )

    clock = SimpleNamespace(value=20.0)

    class FakePage:
        def __init__(self) -> None:
            self.gotos: list[str] = []
            self.waits: list[int] = []

        def goto(self, url: str, **_kwargs) -> None:
            self.gotos.append(url)
            clock.value += 2.452

        def wait_for_timeout(self, milliseconds: int) -> None:
            self.waits.append(milliseconds)
            clock.value += milliseconds / 1000

    page = FakePage()
    state = _SearchNavigationState(
        active_source_url="https://www.linkedin.com/jobs/search/?keywords=kr",
        completed_at=clock.value,
    )
    japan_source = "https://www.linkedin.com/jobs/search/?keywords=jp"

    with (
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._wait_for_search_results_hydration",
            return_value="results",
        ),
    ):
        navigated = _ensure_search_source(
            page,
            source_url=japan_source,
            navigation_state=state,
            interval_ms=2750,
            now_fn=lambda: clock.value,
        )
        reused = _ensure_search_source(
            page,
            source_url=japan_source,
            navigation_state=state,
            interval_ms=2750,
            now_fn=lambda: clock.value,
        )

    assert navigated is True
    assert reused is False
    assert page.waits == [2750]
    assert page.gotos == [japan_source]
    assert state.completed_at == pytest.approx(25.202)


def test_linkedin_failed_source_navigation_invalidates_dom_cache():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _SearchNavigationState,
        _ensure_search_source,
    )

    clock = SimpleNamespace(value=30.0)

    class FakePage:
        def __init__(self) -> None:
            self.goto_count = 0

        def goto(self, _url: str, **_kwargs) -> None:
            self.goto_count += 1
            raise RuntimeError("navigation failed")

        def wait_for_timeout(self, milliseconds: int) -> None:
            clock.value += milliseconds / 1000

    page = FakePage()
    state = _SearchNavigationState(
        active_source_url="https://www.linkedin.com/jobs/search/?keywords=kr",
        completed_at=clock.value,
    )
    source = "https://www.linkedin.com/jobs/search/?keywords=jp"

    for _ in range(2):
        with pytest.raises(RuntimeError, match="navigation failed"):
            _ensure_search_source(
                page,
                source_url=source,
                navigation_state=state,
                interval_ms=2750,
                now_fn=lambda: clock.value,
            )

    assert page.goto_count == 2
    assert state.active_source_url is None
    assert state.completed_at == pytest.approx(35.5)


def test_linkedin_enrichment_source_http_failure_replaces_page_and_retries_once():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _SearchNavigationState,
        _ensure_search_source_with_single_retry,
    )

    source = "https://www.linkedin.com/jobs/search/?keywords=jp"
    navigation_urls: list[str] = []
    warnings: list[str] = []
    retry_reservations = 0

    class PlaywrightError(Exception):
        pass

    class FakePage:
        def __init__(self, *, degraded: bool) -> None:
            self.degraded = degraded

        def goto(self, url: str, **_kwargs) -> None:
            navigation_urls.append(url)
            if self.degraded:
                raise PlaywrightError(
                    "Page.goto: net::ERR_HTTP_RESPONSE_CODE_FAILURE at "
                    "https://www.linkedin.com/jobs/search/?token=secret"
                )

        def wait_for_timeout(self, _milliseconds: int) -> None:
            return None

    class FakeSession:
        def __init__(self) -> None:
            self.page = FakePage(degraded=True)
            self.replace_count = 0

        def page_is_alive(self) -> bool:
            return True

        def replace_page(self):
            self.replace_count += 1
            self.page = FakePage(degraded=False)
            return self.page

    session = FakeSession()
    state = _SearchNavigationState(
        active_source_url="https://www.linkedin.com/jobs/search/?keywords=kr",
    )

    def reserve_retry() -> None:
        nonlocal retry_reservations
        retry_reservations += 1

    with (
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._wait_for_search_results_hydration",
            return_value="results",
        ),
    ):
        page, navigated, retried = _ensure_search_source_with_single_retry(
            session,
            session.page,
            source_url=source,
            navigation_state=state,
            interval_ms=0,
            retry_allowed=True,
            warning_scope="Japan",
            warnings=warnings,
            reserve_retry=reserve_retry,
        )

    assert page is session.page
    assert navigated is True
    assert retried is True
    assert session.replace_count == 1
    assert retry_reservations == 1
    assert navigation_urls == [source, source]
    assert state.active_source_url == source
    assert (
        "detail_source_navigation_retry:Japan:"
        "http_response_code_failure"
    ) in warnings
    assert (
        "page_recovered:detail:Japan:http_response_code_failure"
        in warnings
    )
    assert "token=secret" not in json.dumps(warnings)


def test_linkedin_auth_required_retains_session_and_writes_safe_diagnostic(
    tmp_path,
    monkeypatch,
):
    from features.web_scraping.domain.linkedin_models import LinkedInJobsRequest
    from features.web_scraping.infrastructure.authenticated_browser import (
        AuthenticatedBrowserLaunchConfig,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        LinkedInAuthRequiredError,
        scrape_linkedin_jobs,
    )
    from features.web_scraping.infrastructure.linkedin_session_store import (
        LinkedInSessionStore,
    )

    monkeypatch.delenv("LINKEDIN_BROWSER", raising=False)
    monkeypatch.delenv("LINKEDIN_BROWSER_EXECUTABLE_PATH", raising=False)
    monkeypatch.setenv("LINKEDIN_HEADLESS", "false")
    state_path = tmp_path / "private" / "storage_state.json"
    executable = tmp_path / "brave"
    executable.write_text("", encoding="utf-8")
    executable.chmod(0o700)
    store = LinkedInSessionStore(state_path)
    launch_config = AuthenticatedBrowserLaunchConfig.from_env(
        browser="brave",
        executable_path=executable,
    )

    class FakeContext:
        def storage_state(self, *, path: str) -> None:
            Path(path).write_text(
                json.dumps(
                    {
                        "cookies": [{"name": "li_at", "value": "secret-cookie"}],
                        "origins": [],
                    }
                ),
                encoding="utf-8",
            )

    class FakePage:
        url = "https://www.linkedin.com/jobs/"

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url

    class FakeSession:
        page = FakePage()

        def close(self) -> None:
            return None

    profile_path = store.resolve_profile_path(create=True)
    store.save_from_context(
        FakeContext(),
        launch_config=launch_config,
        profile_path=profile_path,
    )

    with (
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.open_persistent_authenticated_context",
            return_value=FakeSession(),
        ) as open_context,
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page",
            side_effect=LinkedInAuthRequiredError("login"),
        ),
        pytest.raises(LinkedInAuthRequiredError),
    ):
        scrape_linkedin_jobs(
            LinkedInJobsRequest(query="vacantes AI de hoy", max_results=2),
            session_store=store,
        )

    assert state_path.exists()
    assert store.browser_metadata_path.exists()
    diagnostic = json.loads(
        store.runtime_diagnostic_path.read_text(encoding="utf-8")
    )
    assert diagnostic["reason"] == "LinkedInAuthRequiredError"
    assert diagnostic["browser"].startswith("brave")
    assert diagnostic["headless"] is False
    assert diagnostic["profile_mode"] == "persistent"
    assert diagnostic["profile_path"] == str(profile_path)
    assert diagnostic["storage_state_retained"] is True
    assert "secret-cookie" not in store.runtime_diagnostic_path.read_text(
        encoding="utf-8"
    )
    assert store.runtime_diagnostic_path.stat().st_mode & 0o777 == 0o600
    assert open_context.call_args.kwargs["launch_config"].browser == "brave"
    assert open_context.call_args.kwargs["headless"] is False
    assert open_context.call_args.kwargs["profile_path"] == profile_path


def test_linkedin_runtime_uses_persistent_profile_without_storage_snapshot(
    tmp_path,
    monkeypatch,
):
    from features.web_scraping.domain.linkedin_models import LinkedInJobsRequest
    from features.web_scraping.infrastructure.authenticated_browser import (
        AuthenticatedBrowserLaunchConfig,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        scrape_linkedin_jobs,
    )
    from features.web_scraping.infrastructure.linkedin_session_store import (
        LinkedInSessionStore,
    )

    monkeypatch.delenv("LINKEDIN_PROFILE_DIR", raising=False)
    state_path = tmp_path / "private" / "storage_state.json"
    store = LinkedInSessionStore(state_path)
    profile_path = store.resolve_profile_path(create=True)

    class SnapshotContext:
        def storage_state(self, *, path: str) -> None:
            Path(path).write_text(
                json.dumps({"cookies": [], "origins": []}),
                encoding="utf-8",
            )

    store.save_from_context(
        SnapshotContext(),
        launch_config=AuthenticatedBrowserLaunchConfig(browser="chromium"),
        profile_path=profile_path,
    )
    state_path.unlink()

    class FakePage:
        url = "https://www.linkedin.com/jobs/"

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url

    class FakeSession:
        page = FakePage()

        def close(self) -> None:
            return None

    with (
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.open_persistent_authenticated_context",
            return_value=FakeSession(),
        ) as open_context,
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.build_linkedin_search_queries",
            return_value=[],
        ),
    ):
        records, rejected, timings, warnings, queries = scrape_linkedin_jobs(
            LinkedInJobsRequest(query="vacantes AI de hoy", max_results=2),
            session_store=store,
        )

    assert (records, rejected, timings, warnings, queries) == ([], [], [], [], [])
    assert not state_path.exists()
    assert open_context.call_args.kwargs["profile_path"] == profile_path


def test_linkedin_auth_detection_stops_login_and_challenge_pages():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        LinkedInAuthRequiredError,
        LinkedInBlockedError,
        _validate_authenticated_page,
    )

    class FakeLocator:
        def __init__(self, text: str, count: int = 0) -> None:
            self._text = text
            self._count = count

        def inner_text(self, timeout: int) -> str:
            return self._text

        def count(self) -> int:
            return self._count

    class FakeContext:
        def __init__(self, authenticated: bool) -> None:
            self._authenticated = authenticated

        def cookies(self) -> list[dict[str, str]]:
            return [{"name": "li_at"}] if self._authenticated else []

    class FakePage:
        def __init__(self, url: str, text: str = "", authenticated: bool = False) -> None:
            self.url = url
            self._text = text
            self.context = FakeContext(authenticated)

        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator(self._text)

    with pytest.raises(LinkedInAuthRequiredError):
        _validate_authenticated_page(FakePage("https://www.linkedin.com/login"))
    with pytest.raises(LinkedInBlockedError):
        _validate_authenticated_page(
            FakePage(
                "https://www.linkedin.com/jobs/",
                "Security verification CAPTCHA",
            )
        )
    _validate_authenticated_page(
        FakePage("https://www.linkedin.com/jobs/", authenticated=True)
    )


def test_linkedin_audit_writes_validated_json_and_schema(tmp_path, monkeypatch):
    from features.web_scraping.application import linkedin_audit
    from features.web_scraping.application.linkedin_audit import (
        load_linkedin_audit_snapshot,
        persist_linkedin_audit_snapshot,
    )
    from features.web_scraping.domain.linkedin_models import (
        LinkedInQueryTiming,
        LinkedInRejectedRecord,
        LinkedInVacancyRecord,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        linkedin_audit,
        "get_request_runtime_config",
        lambda: SimpleNamespace(session_id="sess-1", request_id="req-1"),
    )
    record = LinkedInVacancyRecord(
        linkedin_job_id="111",
        title="AI Engineer",
        company_name="Example AI",
        posted_at_text="2 hours ago",
        published_at=datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc),
        freshness_confidence="high",
        is_within_24_hours=True,
        canonical_url="https://www.linkedin.com/jobs/view/ai-engineer-111",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        hard_skills=["Python", "PyTorch"],
        soft_skills=["Communication"],
        candidate_expectations=["5+ years of ML experience"],
        responsibilities=["Build production AI systems"],
        matched_terms=["ai"],
    )
    timing = LinkedInQueryTiming(
        query="AI Engineer @ Japan",
        started_at=datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc),
        completed_at=datetime(2026, 7, 28, 12, 0, 1, tzinfo=timezone.utc),
        elapsed_ms=1000,
        discovered_count=1,
        retained_count=1,
        diagnostics={
            "selector_counts": {".job-card-container": 7},
            "href_count": 9,
            "candidate_count": 16,
            "parseable_candidate_count": 1,
            "discard_reasons": {"duplicate_wrapper": 8},
        },
    )
    paths = persist_linkedin_audit_snapshot(
        original_query="vacantes AI de hoy",
        queries=["https://www.linkedin.com/jobs/search/?keywords=AI"],
        timings=[timing],
        vacancies=[record],
        rejected=[
            LinkedInRejectedRecord(
                source_url="https://www.linkedin.com/jobs/view/222",
                title="ML Engineer",
                reason="detail_network_failure",
            ),
            LinkedInRejectedRecord(
                source_url="https://www.linkedin.com/jobs/view/333",
                title="Data Scientist",
                reason="detail_budget_exhausted",
            ),
        ],
        warnings=[],
    )
    snapshot = load_linkedin_audit_snapshot(paths.json_path)

    assert snapshot.meta.session_id == "sess-1"
    assert snapshot.meta.schema_version == "1.3.0"
    assert snapshot.meta.result_count == 1
    assert snapshot.meta.rejected_count == 2
    assert [item.reason for item in snapshot.rejected] == [
        "detail_network_failure",
        "detail_budget_exhausted",
    ]
    assert snapshot.vacancies[0].title == "AI Engineer"
    assert snapshot.vacancies[0].language_requirements == []
    assert snapshot.vacancies[0].experience_requirements == []
    assert snapshot.vacancies[0].hard_skills == ["Python", "PyTorch"]
    assert snapshot.vacancies[0].soft_skills == ["Communication"]
    assert snapshot.vacancies[0].candidate_expectations == [
        "5+ years of ML experience"
    ]
    assert snapshot.vacancies[0].responsibilities == [
        "Build production AI systems"
    ]
    assert snapshot.vacancies[0].foreigner_acceptance == "unknown"
    assert snapshot.vacancies[0].visa_status == "unknown"
    assert snapshot.vacancies[0].relocation_support == "unknown"
    assert snapshot.timings[0].diagnostics.href_count == 9
    assert snapshot.timings[0].diagnostics.discard_reasons == {
        "duplicate_wrapper": 8
    }
    assert paths.schema_path.exists()
    serialized = paths.json_path.read_text(encoding="utf-8")
    assert "storage_state" not in serialized
    assert "cookies" not in serialized
    assert "<html" not in serialized.lower()
    summary = paths.summary_path.read_text(encoding="utf-8")
    assert "Selector counts" in summary
    assert "Hard skills: Python, PyTorch" in summary
    assert "Soft skills: Communication" in summary
    assert "Expectativas: 5+ years of ML experience" in summary
    assert "Responsabilidades: Build production AI systems" in summary
    schema = json.loads(paths.schema_path.read_text(encoding="utf-8"))
    vacancy_schema = schema["$defs"]["LinkedInVacancyRecord"]
    assert "language_requirements" in vacancy_schema["properties"]
    for field_name in (
        "hard_skills",
        "soft_skills",
        "candidate_expectations",
        "responsibilities",
    ):
        assert field_name in vacancy_schema["properties"]
    assert vacancy_schema["properties"]["candidate_expectations"]["maxItems"] == 6
    assert vacancy_schema["properties"]["responsibilities"]["maxItems"] == 6


@pytest.mark.asyncio
async def test_linkedin_dispatch_runs_input_guard_and_agentdog_before_return():
    if sys.version_info < (3, 10):
        pytest.skip("El flow existente usa PEP 604 sin future annotations.")
    from core.domain.models import AgentState
    from features.web_scraping.application.flow import run_web_scraping_flow
    from features.web_scraping.domain.linkedin_models import LinkedInJobsResult
    from langchain_core.messages import HumanMessage

    state = {
        "messages": [HumanMessage(content="buscá vacantes LinkedIn de AI Engineer de hoy")],
        "next_agent": "web_scraping_agent",
        "risk_flag": False,
        "blocked": False,
        "request_id": "req-linkedin",
        "session_id": "sess-linkedin",
        "scrape_tracker": {},
    }
    long_summary = (
        "Encontré vacantes verificadas de LinkedIn. "
        + " ".join(f"vacante_{idx}" for idx in range(260))
    )
    service_result = LinkedInJobsResult(
        status="ok",
        job_uid="linkedin-job-test",
        user_summary=long_summary,
    )
    guard = MagicMock(return_value={"blocked": False})
    agentdog = AsyncMock(return_value=(True, {"label": "safe", "verdict_source": "model"}))

    with (
        patch("features.web_scraping.application.flow.input_guard", guard),
        patch(
            "features.web_scraping.application.linkedin_service.run_linkedin_jobs_vertical",
            return_value=service_result,
        ),
        patch(
            "features.web_scraping.application.flow._select_strategy_context",
            return_value={"tracker": {}},
        ),
        patch("features.web_scraping.application.flow._emit_node_outcome"),
        patch("features.web_scraping.application.flow._emit_guard_audit"),
    ):
        result = await run_web_scraping_flow(
            state,  # type: ignore[arg-type]
            agent=MagicMock(),
            get_llm_fn=MagicMock(),
            get_runtime_policy=lambda: {},
            should_evaluate_guard_fn=lambda _: True,
            evaluate_trajectory_safe_fn=agentdog,
        )

    guard.assert_called_once()
    agentdog.assert_awaited_once()
    content = result["messages"][0].content
    assert "vacante_0" in content
    assert "vacante_259" in content
    assert len(content.split()) > 200


@pytest.mark.asyncio
async def test_linkedin_validation_error_does_not_trigger_public_fallback():
    if sys.version_info < (3, 10):
        pytest.skip("El flow existente usa PEP 604 sin future annotations.")
    from features.web_scraping.application.flow import run_web_scraping_flow
    from features.web_scraping.domain.linkedin_models import LinkedInJobsResult
    from langchain_core.messages import HumanMessage

    state = {
        "messages": [HumanMessage(content="buscá vacantes LinkedIn de AI Engineer de hoy")],
        "next_agent": "web_scraping_agent",
        "risk_flag": False,
        "blocked": False,
        "request_id": "req-linkedin-validation",
        "session_id": "sess-linkedin-validation",
        "scrape_tracker": {},
    }
    service_result = LinkedInJobsResult(
        status="validation_error",
        user_summary="`max_results` debe ser un número entre 1 y 50.",
    )
    public_fallback = AsyncMock()
    agentdog = AsyncMock(return_value=(True, {"label": "safe", "verdict_source": "model"}))

    with (
        patch("features.web_scraping.application.flow.input_guard", return_value={"blocked": False}),
        patch(
            "features.web_scraping.application.linkedin_service.run_linkedin_jobs_vertical",
            return_value=service_result,
        ),
        patch(
            "features.web_scraping.application.flow._run_generic_web_search_fetch",
            public_fallback,
        ),
        patch(
            "features.web_scraping.application.flow._select_strategy_context",
            return_value={"tracker": {}},
        ),
        patch("features.web_scraping.application.flow._emit_node_outcome"),
        patch("features.web_scraping.application.flow._emit_guard_audit"),
    ):
        result = await run_web_scraping_flow(
            state,  # type: ignore[arg-type]
            agent=MagicMock(),
            get_llm_fn=MagicMock(),
            get_runtime_policy=lambda: {},
            should_evaluate_guard_fn=lambda _: True,
            evaluate_trajectory_safe_fn=agentdog,
        )

    public_fallback.assert_not_awaited()
    agentdog.assert_awaited_once()
    assert "entre 1 y 50" in result["messages"][0].content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    ["extraction_incomplete", "auth_required", "blocked", "error"],
)
async def test_linkedin_failures_never_trigger_generic_public_fallback(status):
    if sys.version_info < (3, 10):
        pytest.skip("El flow existente usa PEP 604 sin future annotations.")
    from features.web_scraping.application.flow import run_web_scraping_flow
    from features.web_scraping.domain.linkedin_models import LinkedInJobsResult
    from langchain_core.messages import HumanMessage

    state = {
        "messages": [HumanMessage(content="buscá vacantes LinkedIn de AI Engineer de hoy")],
        "next_agent": "web_scraping_agent",
        "risk_flag": False,
        "blocked": False,
        "request_id": f"req-linkedin-{status}",
        "session_id": f"sess-linkedin-{status}",
        "scrape_tracker": {},
    }
    service_result = LinkedInJobsResult(
        status=status,
        user_summary=f"Error accionable LinkedIn: {status}.",
    )
    public_fallback = AsyncMock(
        return_value={"summary": "Noticias japonesas irrelevantes"}
    )
    agentdog = AsyncMock(
        return_value=(True, {"label": "safe", "verdict_source": "model"})
    )

    with (
        patch(
            "features.web_scraping.application.flow.input_guard",
            return_value={"blocked": False},
        ),
        patch(
            "features.web_scraping.application.linkedin_service.run_linkedin_jobs_vertical",
            return_value=service_result,
        ),
        patch(
            "features.web_scraping.application.flow._run_generic_web_search_fetch",
            public_fallback,
        ),
        patch(
            "features.web_scraping.application.flow._select_strategy_context",
            return_value={"tracker": {}},
        ),
        patch("features.web_scraping.application.flow._emit_node_outcome"),
        patch("features.web_scraping.application.flow._emit_guard_audit"),
    ):
        result = await run_web_scraping_flow(
            state,  # type: ignore[arg-type]
            agent=MagicMock(),
            get_llm_fn=MagicMock(),
            get_runtime_policy=lambda: {},
            should_evaluate_guard_fn=lambda _: True,
            evaluate_trajectory_safe_fn=agentdog,
        )

    public_fallback.assert_not_awaited()
    agentdog.assert_awaited_once()
    content = result["messages"][0].content
    assert f"Error accionable LinkedIn: {status}." in content
    assert "Noticias japonesas" not in content



def test_linkedin_structured_sections_detect_real_world_responsibility_headings():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _extract_structured_sections,
    )

    expectations, responsibilities = _extract_structured_sections(
        """
In This Role, You’ll Get to
Design, code, experiment and implement models and algorithms.
Mine big data and discover actionable insights.
What You’ll Need To Succeed
4+ years hands-on data science experience.
"""
    )
    assert any("4+ years" in item for item in expectations)
    assert any("Design, code" in item for item in responsibilities)

    expectations, responsibilities = _extract_structured_sections(
        """
What You'll Be Doing
Develop and demonstrate solutions based on NVIDIA GenAI technologies.
Drive pre-sales conversations and build architectures.
What We Need To See
Business level English communication is also a requirement.
"""
    )
    assert any("Business level English" in item for item in expectations)
    assert any("NVIDIA GenAI" in item for item in responsibilities)

    expectations, responsibilities = _extract_structured_sections(
        """
【仕事内容】
GPU/FPGA/AI Acceleratorなどのハードウェア特性に最適なDeep Neural Networks (DNN）の実装
【従事すべき業務の内容】
雇入れ直後： 本求人に記載のある業務
Requirements
Computer Architecture及びMachine Learningの知識
"""
    )
    assert any("Computer Architecture" in item for item in expectations)
    assert any("GPU/FPGA" in item for item in responsibilities)

    expectations, responsibilities = _extract_structured_sections(
        """
Key Responsibilities
• Design, develop, and evaluate enterprise-grade applications utilizing Large Language Models (LLMs)
• Build and support Retrieval-Augmented Generation (RAG) systems and AI Agent-based applications
Compensation & Benefits:
• Annual salary: ¥6M – ¥25M JPY
"""
    )
    assert any("Large Language Models" in item for item in responsibilities)
    assert not any("Annual salary" in item for item in responsibilities)


def test_linkedin_mobility_inference_detects_international_applications_phrase():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _infer_foreigner_acceptance,
        _infer_relocation_support,
        _infer_visa_status,
    )

    text = (
        "We welcome both local and international applications for this role. "
        "Full visa sponsorship and relocation assistance available for eligible candidates."
    )

    assert _infer_foreigner_acceptance(text) == "yes"
    assert _infer_visa_status(text) == "sponsorship"
    assert _infer_relocation_support(text) == "yes"


def test_linkedin_structured_items_truncate_on_word_boundary():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _extract_structured_sections,
    )

    expectations, responsibilities = _extract_structured_sections(
        """
Requirements
4+ years hands-on data science experience with production machine learning systems.
Responsibilities
Collaborating closely with key application developers to understand and address their current and future challenges while developing optimized systems.
"""
    )

    assert expectations[0].startswith("4+ years")
    assert all(not item.endswith((" p", " str", " tim", " imp")) for item in responsibilities)
    assert all(len(item) <= 180 for item in expectations + responsibilities)


def test_linkedin_country_group_separates_apac_remote_from_korea_search_context():
    from features.web_scraping.application.linkedin_service import _record_country_group
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord

    record = LinkedInVacancyRecord(
        title="AI Engineer | Remote",
        location="Asia-Pacífico y Japón",
        canonical_url="https://www.linkedin.com/jobs/view/1",
        source_url="https://www.linkedin.com/jobs/search/?location=South%20Korea",
    )

    assert _record_country_group(record) == "Asia-Pacífico / Remote"


def test_linkedin_incomplete_detail_body_is_rejected_before_rendering():
    from features.web_scraping.infrastructure.linkedin_scraper import _is_incomplete_detail_body

    assert _is_incomplete_detail_body("Acerca del empleo") is True
    assert _is_incomplete_detail_body("About the job") is True
    assert _is_incomplete_detail_body("Acerca del empleo\nDesign and build AI systems") is False


def test_linkedin_experience_inference_handles_real_world_years_lines():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _infer_experience_requirements,
    )

    text = """
What You’ll Need To Succeed
4+ years hands-on data science experience.
Excellent understanding of AI/ML/DL and Statistics.
Working Conditions
Annual salary: ¥6M – ¥25M JPY
What We Need To See
MS or PhD degree in AI computation or system optimization with a strong computational profile, or equivalent experience and 3+ years of relevant work.
"""

    requirements = _infer_experience_requirements(text)

    assert "4+ years hands-on data science experience." in requirements
    assert any("3+ years of relevant work" in item for item in requirements)
    assert not any("Annual salary" in item for item in requirements)
