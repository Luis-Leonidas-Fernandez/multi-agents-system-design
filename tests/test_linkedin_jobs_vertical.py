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


def test_linkedin_reusable_context_can_be_released_for_manual_refresh(tmp_path):
    from features.web_scraping.infrastructure.authenticated_browser import (
        AuthenticatedBrowserLaunchConfig,
        close_reusable_authenticated_contexts,
        open_persistent_authenticated_context,
    )

    profile_path = tmp_path / "private" / "browser-profile"
    profile_path.mkdir(parents=True)
    page = MagicMock(url="about:blank")
    context = MagicMock()
    context.pages = [page]
    chromium = MagicMock()
    chromium.launch_persistent_context.return_value = context
    playwright = SimpleNamespace(chromium=chromium)

    open_persistent_authenticated_context(
        playwright=playwright,
        profile_path=profile_path,
        launch_config=AuthenticatedBrowserLaunchConfig(browser="chromium"),
        reuse=True,
    )

    assert close_reusable_authenticated_contexts(profile_path=profile_path) == 1
    assert close_reusable_authenticated_contexts(profile_path=profile_path) == 0
    context.close.assert_called_once()


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


def test_linkedin_bootstrap_observer_detects_authenticated_jobs_ready():
    from scripts.bootstrap_linkedin_session import observe_linkedin_jobs_readiness

    class FakeLocator:
        def inner_text(self, *, timeout):
            return "LinkedIn Jobs"

        def count(self):
            return 0

    class FakeContext:
        def __init__(self):
            self.pages = []

        def cookies(self):
            return [{"name": "li_at"}]

    context = FakeContext()
    page = SimpleNamespace(
        url="https://www.linkedin.com/jobs/",
        context=context,
        locator=lambda _selector: FakeLocator(),
    )
    context.pages = [page]

    assert observe_linkedin_jobs_readiness(
        context,
        timeout_seconds=1,
        monotonic=lambda: 0.0,
        sleep=lambda _seconds: None,
    ) == "ready"


def test_linkedin_bootstrap_observer_times_out_without_automating_login():
    from scripts.bootstrap_linkedin_session import observe_linkedin_jobs_readiness

    class FakeLocator:
        def inner_text(self, *, timeout):
            return "Sign in"

        def count(self):
            return 1

    context = SimpleNamespace(pages=[])
    page = SimpleNamespace(
        url="https://www.linkedin.com/login",
        context=SimpleNamespace(cookies=lambda: []),
        locator=lambda _selector: FakeLocator(),
    )
    context.pages = [page]
    clock = iter((0.0, 2.0))

    assert observe_linkedin_jobs_readiness(
        context,
        timeout_seconds=1,
        monotonic=lambda: next(clock),
        sleep=lambda _seconds: None,
    ) == "timeout"
    assert not hasattr(page, "fill")


def test_linkedin_bootstrap_exit_4_is_reserved_for_profile_in_use(tmp_path):
    from features.web_scraping.infrastructure.authenticated_browser import (
        BrowserProfileInUseError,
    )
    from scripts import bootstrap_linkedin_session

    store = MagicMock()
    store.resolve_profile_path.return_value = tmp_path / "browser-profile"
    playwright_context = MagicMock()
    playwright_context.__enter__.return_value = SimpleNamespace()

    with patch.object(
        bootstrap_linkedin_session,
        "LinkedInSessionStore",
        return_value=store,
    ), patch.object(
        bootstrap_linkedin_session,
        "sync_playwright",
        return_value=playwright_context,
    ), patch.object(
        bootstrap_linkedin_session,
        "open_persistent_authenticated_context",
        side_effect=BrowserProfileInUseError("profile already in use"),
    ):
        returncode = bootstrap_linkedin_session.main(
            ["--observe-ready", "--ready-timeout-seconds", "1"]
        )

    assert returncode == 4


def test_linkedin_bootstrap_observer_persists_ready_state_and_closes(tmp_path):
    from scripts import bootstrap_linkedin_session

    profile_path = tmp_path / "browser-profile"
    store = MagicMock()
    store.resolve_profile_path.return_value = profile_path
    page = MagicMock()
    context = SimpleNamespace(pages=[page])
    session = SimpleNamespace(context=context, page=page, close=MagicMock())
    playwright_context = MagicMock()
    playwright_context.__enter__.return_value = SimpleNamespace()

    with patch.object(
        bootstrap_linkedin_session,
        "LinkedInSessionStore",
        return_value=store,
    ), patch.object(
        bootstrap_linkedin_session,
        "sync_playwright",
        return_value=playwright_context,
    ), patch.object(
        bootstrap_linkedin_session,
        "open_persistent_authenticated_context",
        return_value=session,
    ), patch.object(
        bootstrap_linkedin_session,
        "observe_linkedin_jobs_readiness",
        return_value="ready",
    ), patch.object(
        bootstrap_linkedin_session,
        "detect_google_oauth_automation_block_in_context",
        return_value=False,
    ):
        returncode = bootstrap_linkedin_session.main(
            ["--observe-ready", "--ready-timeout-seconds", "1"]
        )

    assert returncode == 0
    store.save_from_context.assert_called_once()
    assert store.save_from_context.call_args.args == (context,)
    assert store.save_from_context.call_args.kwargs["profile_path"] == profile_path
    session.close.assert_called_once()


def test_linkedin_bootstrap_observer_returns_explicit_timeout_and_closes(tmp_path):
    from scripts import bootstrap_linkedin_session

    store = MagicMock()
    store.resolve_profile_path.return_value = tmp_path / "browser-profile"
    page = MagicMock()
    session = SimpleNamespace(
        context=SimpleNamespace(pages=[page]),
        page=page,
        close=MagicMock(),
    )
    playwright_context = MagicMock()
    playwright_context.__enter__.return_value = SimpleNamespace()

    with patch.object(
        bootstrap_linkedin_session,
        "LinkedInSessionStore",
        return_value=store,
    ), patch.object(
        bootstrap_linkedin_session,
        "sync_playwright",
        return_value=playwright_context,
    ), patch.object(
        bootstrap_linkedin_session,
        "open_persistent_authenticated_context",
        return_value=session,
    ), patch.object(
        bootstrap_linkedin_session,
        "observe_linkedin_jobs_readiness",
        return_value="timeout",
    ):
        returncode = bootstrap_linkedin_session.main(
            ["--observe-ready", "--ready-timeout-seconds", "1"]
        )

    assert returncode == 5
    store.save_from_context.assert_not_called()
    session.close.assert_called_once()


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
    assert extract_linkedin_locations(
        "Buscá vacantes solo para Corea del Sur. No incluyas Japón ni otros países."
    ) == ("South Korea",)
    assert extract_linkedin_locations(
        "Buscá vacantes únicamente en Corea del Sur y descartá Japón."
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


def test_linkedin_entity_url_canonicalization_strips_query_and_fragment():
    from features.web_scraping.infrastructure.linkedin_url_policy import (
        canonicalize_linkedin_url,
        linkedin_job_id_from_url,
    )

    assert canonicalize_linkedin_url(
        "https://www.linkedin.com/jobs/view/4451609695#details"
    ) == "https://www.linkedin.com/jobs/view/4451609695"
    assert canonicalize_linkedin_url(
        "https://linkedin.com/company/openai#about"
    ) == "https://linkedin.com/company/openai"
    assert canonicalize_linkedin_url(
        "https://www.linkedin.com/company/hire-feed/about/?trk=public_jobs"
    ) == "https://www.linkedin.com/company/hire-feed/about"
    assert {
        linkedin_job_id_from_url(url)
        for url in (
            "https://www.linkedin.com/jobs/view/4451609695/",
            "https://www.linkedin.com/jobs/view/4451609695?trackingId=x",
            "https://www.linkedin.com/jobs/view/some-role-4451609695?refId=y",
        )
    } == {"4451609695"}


def test_linkedin_url_policy_accepts_only_official_search_results_redirect():
    from features.web_scraping.infrastructure.linkedin_url_policy import (
        canonicalize_linkedin_url,
        canonicalize_linkedin_job_url,
        validate_linkedin_jobs_url,
    )

    redirected = (
        "https://linkedin.com/jobs/search-results/?keywords=AI&f_TPR=r86400"
        "&location=Japan&infoNotice=job-search-rewrite&skipRedirect=true"
    )
    assert validate_linkedin_jobs_url(redirected) == redirected
    assert canonicalize_linkedin_job_url(redirected) == (
        "https://www.linkedin.com/jobs/search-results?"
        "keywords=AI&f_TPR=r86400&location=Japan&infoNotice=job-search-rewrite"
        "&skipRedirect=true"
    )
    with pytest.raises(ValueError, match="path_not_allowed"):
        canonicalize_linkedin_url(redirected)

    with pytest.raises(ValueError, match="query_parameter_not_allowed"):
        validate_linkedin_jobs_url(
            "https://www.linkedin.com/jobs/search-results?keywords=AI&token=secret"
        )
    with pytest.raises(ValueError, match="path_not_allowed"):
        validate_linkedin_jobs_url(
            "https://www.linkedin.com/jobs/search-results/anything"
        )


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


def test_linkedin_parser_extracts_plain_text_card_date_without_losing_fallback():
    from datetime import datetime, timezone

    from features.web_scraping.infrastructure.linkedin_scraper import (
        parse_linkedin_jobs_html,
    )

    html = """
    <ul class="jobs-search-results-list">
      <li data-occludable-job-id="4426794001">
        <a class="job-card-list__title--link" href="/jobs/view/4426794001/">
          Machine Learning Engineer (AdTech/MarTech)
        </a>
        <div class="job-card-container__primary-description">MUSINSA 무신사</div>
        <div class="job-card-container__metadata-item">Seúl, Corea del Sur (Presencial)</div>
        <span>Visto · Adelántate a solicitar el empleo · Hace 9 horas · Solicitud sencilla</span>
      </li>
    </ul>
    """

    records = parse_linkedin_jobs_html(
        html,
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI&f_TPR=r86400",
        now=datetime(2026, 8, 13, 18, 0, tzinfo=timezone.utc),
    )

    assert len(records) == 1
    assert records[0].linkedin_job_id == "4426794001"
    assert records[0].posted_at_text == "Hace 9 horas"
    assert records[0].published_at == datetime(2026, 8, 13, 9, 0, tzinfo=timezone.utc)
    assert records[0].is_within_24_hours is True


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


def test_linkedin_parser_uses_semantic_search_area_and_exact_id_dedupe():
    from features.web_scraping.infrastructure.linkedin_parser import (
        _parse_linkedin_jobs_html_with_diagnostics,
    )

    html = """
    <nav><a href="/jobs/view/900" aria-label="Global job link"></a></nav>
    <main>
      <ul class="jobs-search-results-list" role="listbox">
        <li role="option" data-occludable-job-id="00111">
          <a href="/jobs/view/111" aria-label="AI Engineer"></a>
          <a href="/jobs/view/111">AI Engineer duplicate</a>
        </li>
        <li role="listitem" data-job-id="112">
          <a href="/jobs/view/1112">Wrong exact ID</a>
        </li>
        <li role="listitem" data-job-id="222">
          <a href="/jobs/view/222">Machine Learning Engineer</a>
        </li>
      </ul>
      <aside class="jobs-search__job-details--container">
        <a href="/jobs/view/333">Detail panel title</a>
      </aside>
    </main>
    """

    records, diagnostics = _parse_linkedin_jobs_html_with_diagnostics(
        html,
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
    )

    assert [record.linkedin_job_id for record in records] == ["111", "222"]
    assert [record.title for record in records] == [
        "AI Engineer",
        "Machine Learning Engineer",
    ]
    assert diagnostics.candidate_count == 2
    assert "legacy_selector_fallback" not in diagnostics.discard_reasons


def test_linkedin_parser_rejects_standalone_job_link_by_default():
    from features.web_scraping.infrastructure.linkedin_parser import (
        _parse_linkedin_jobs_html_with_diagnostics,
    )

    records, diagnostics = _parse_linkedin_jobs_html_with_diagnostics(
        '<main><a href="/jobs/view/111">AI Engineer</a></main>',
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
    )

    assert records == []
    assert diagnostics.href_count == 1
    assert diagnostics.candidate_count == 0
    assert "standalone_link_fallback" not in diagnostics.discard_reasons


def test_linkedin_parser_opt_in_salvages_safe_standalone_job_links_in_order():
    from features.web_scraping.infrastructure.linkedin_parser import (
        _parse_linkedin_jobs_html_with_diagnostics,
    )

    html = """
    <main>
      <a href="/jobs/view/ai-engineer-111/?trackingId=secret"
         aria-label="AI Engineer"></a>
      <a href="/jobs/view/222">Machine Learning Engineer</a>
      <a href="/jobs/view/00111">Duplicate AI Engineer</a>
      <a href="https://www.linkedin.com/jobs/view/data-scientist-333">
        Data Scientist
      </a>
    </main>
    """

    records, diagnostics = _parse_linkedin_jobs_html_with_diagnostics(
        html,
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        allow_standalone_fallback=True,
    )

    assert [record.linkedin_job_id for record in records] == ["111", "222", "333"]
    assert [record.title for record in records] == [
        "AI Engineer",
        "Machine Learning Engineer",
        "Data Scientist",
    ]
    assert records[0].canonical_url == (
        "https://www.linkedin.com/jobs/view/ai-engineer-111"
    )
    assert diagnostics.candidate_count == 3
    assert diagnostics.parseable_candidate_count == 3
    assert diagnostics.discard_reasons == {"standalone_link_fallback": 3}


@pytest.mark.parametrize(
    "html",
    [
        '<nav><a href="/jobs/view/111">AI Engineer</a></nav>',
        '<main><a href="/jobs/view/not-a-numeric-id">AI Engineer</a></main>',
        '<main><a href="/jobs/view/111" aria-label=""></a></main>',
        '<main><a href="https://evil.example/jobs/view/111">AI Engineer</a></main>',
    ],
)
def test_linkedin_parser_opt_in_rejects_unsafe_standalone_job_links(html):
    from features.web_scraping.infrastructure.linkedin_parser import (
        _parse_linkedin_jobs_html_with_diagnostics,
    )

    records, diagnostics = _parse_linkedin_jobs_html_with_diagnostics(
        html,
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        allow_standalone_fallback=True,
    )

    assert records == []
    assert diagnostics.candidate_count == 0
    assert "standalone_link_fallback" not in diagnostics.discard_reasons


def test_linkedin_parser_discards_broken_card_without_aborting(monkeypatch):
    from bs4 import BeautifulSoup

    from features.web_scraping.infrastructure import linkedin_parser

    class BrokenCard:
        def select_one(self, _selector):
            raise RuntimeError("do not leak this message")

    soup = BeautifulSoup(
        """
        <ul class="jobs-search-results-list">
          <li data-job-id="4451609695">
            <a href="/jobs/view/4451609695">AI Engineer</a>
            <span class="job-card-container__primary-description">Example AI</span>
          </li>
        </ul>
        """,
        "html.parser",
    )
    good_card = soup.select_one("li[data-job-id]")
    good_link = good_card.select_one("a")

    monkeypatch.setattr(
        linkedin_parser,
        "_semantic_candidates",
        lambda _soup: [(BrokenCard(), good_link), (good_card, good_link)],
    )

    records, diagnostics = linkedin_parser._parse_linkedin_jobs_html_with_diagnostics(
        str(soup),
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
    )

    assert [record.linkedin_job_id for record in records] == ["4451609695"]
    assert diagnostics.discard_reasons["duplicate_wrapper"] >= 1


def test_linkedin_search_hydration_waits_progressively_for_late_cards():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _wait_for_search_results_hydration,
    )

    state = {"polls": 0}

    class FakeLocator:
        def __init__(self, selector: str, index: int = 0) -> None:
            self.selector = selector
            self.index = index

        @property
        def first(self):
            return self

        def count(self) -> int:
            if state["polls"] < 2:
                return 0
            if (
                self.selector
                == ".jobs-search-results-list [data-job-id] a[href*='/jobs/view/']"
            ):
                return 2
            if self.selector == "a[href*='/jobs/view/']":
                return 2
            return 0

        def nth(self, index: int):
            return FakeLocator(self.selector, index)

        def get_attribute(self, name: str):
            if name == "href" and self.count():
                return f"/jobs/view/{111 + self.index}"
            return None

        def evaluate(self, _script: str) -> None:
            return None

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


def test_linkedin_search_hydration_diagnostics_record_polling_and_timeout_safely():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _wait_for_search_results_hydration,
    )
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        LinkedInSearchHydrationDiagnosticsCollector,
    )

    state = {"polls": 0}

    class FakeLocator:
        def __init__(self, selector: str) -> None:
            self.selector = selector

        @property
        def first(self):
            return self

        def count(self) -> int:
            if self.selector == ".jobs-search-results-list [data-job-id]":
                return min(state["polls"], 2)
            return 0

        def nth(self, _index: int):
            return self

        def get_attribute(self, name: str):
            return "/jobs/view/secret-raw-url-111" if name == "href" else None

        def is_visible(self, timeout: int) -> bool:
            return bool(self.count())

        def inner_text(self, timeout: int) -> str:
            return "<html>Visible Job Title secret-token</html>"

        def evaluate(self, _script: str) -> None:
            return None

    class FakePage:
        url = "https://www.linkedin.com/jobs/search/?keywords=secret-token"

        def __init__(self) -> None:
            self.waits = []

        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator(selector)

        def wait_for_timeout(self, milliseconds: int) -> None:
            self.waits.append(milliseconds)
            state["polls"] += 1

    collector = LinkedInSearchHydrationDiagnosticsCollector()
    result = _wait_for_search_results_hydration(
        FakePage(),
        max_wait_ms=450,
        query="https://www.linkedin.com/jobs/search/?keywords=secret-token",
        diagnostics=collector,
    )

    assert result == "timeout"
    assert [event.sequence for event in collector.events] == [1, 2, 3, 4]
    assert [event.outcome for event in collector.events] == [
        "polling",
        "polling",
        "polling",
        "timeout",
    ]
    assert [event.card_count for event in collector.events] == [0, 1, 2, 2]
    assert {event.href_count for event in collector.events} == {0}
    serialized = json.dumps(
        [event.model_dump(mode="json") for event in collector.events]
    )
    assert "https://" not in serialized
    assert "jobs/search" not in serialized
    assert "secret-token" not in serialized
    assert "<html" not in serialized.lower()
    assert "Visible Job Title" not in serialized


def test_linkedin_search_hydration_diagnostics_skip_unlabeled_internal_waits():
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        search_hydration_diagnostics_context,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _wait_for_search_results_hydration,
    )

    class FakeLocator:
        @property
        def first(self):
            return self

        def count(self) -> int:
            return 0

        def nth(self, _index: int):
            return self

        def get_attribute(self, _name: str):
            return None

        def is_visible(self, timeout: int) -> bool:
            return False

        def inner_text(self, timeout: int) -> str:
            return ""

    class FakePage:
        url = "https://www.linkedin.com/jobs/search/?keywords=AI"

        def locator(self, _selector: str) -> FakeLocator:
            return FakeLocator()

        def wait_for_timeout(self, _milliseconds: int) -> None:
            return None

    with search_hydration_diagnostics_context() as collector:
        result = _wait_for_search_results_hydration(FakePage(), max_wait_ms=0)

    assert result == "timeout"
    assert collector.events == []


def test_linkedin_search_hydration_rejects_isolated_job_link():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _wait_for_search_results_hydration,
    )

    class FakeLocator:
        def __init__(self, selector: str) -> None:
            self.selector = selector

        @property
        def first(self):
            return self

        def count(self) -> int:
            return int(self.selector == "a[href*='/jobs/view/']")

        def nth(self, _index: int):
            return self

        def get_attribute(self, name: str):
            return "/jobs/view/111" if name == "href" else None

        def is_visible(self, timeout: int) -> bool:
            return bool(self.count())

        def evaluate(self, _script: str) -> None:
            return None

    class FakePage:
        url = "https://www.linkedin.com/jobs/search/?keywords=AI"

        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator(selector)

        def wait_for_timeout(self, _milliseconds: int) -> None:
            return None

    assert _wait_for_search_results_hydration(
        FakePage(),
        max_wait_ms=450,
    ) == "timeout"


def test_linkedin_search_hydration_separates_raw_signals_from_unique_candidates():
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        LinkedInSearchHydrationDiagnosticsCollector,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _wait_for_search_results_hydration,
    )

    class FakeLocator:
        def __init__(self, selector: str, index: int = 0) -> None:
            self.selector = selector
            self.index = index

        @property
        def first(self):
            return self

        def count(self) -> int:
            return {
                "a[href*='/jobs/view/']": 2,
                "[data-entity-urn*='jobPosting']": 1,
                "[data-job-id]": 1,
                "[data-occludable-job-id]": 1,
            }.get(self.selector, 0)

        def nth(self, index: int):
            return FakeLocator(self.selector, index)

        def get_attribute(self, name: str):
            if self.selector == "a[href*='/jobs/view/']" and name == "href":
                return "/jobs/view/123"
            if (
                self.selector == "[data-entity-urn*='jobPosting']"
                and name == "data-entity-urn"
            ):
                return "urn:li:jobPosting:123"
            if self.selector == "[data-job-id]" and name == "data-job-id":
                return "123"
            if (
                self.selector == "[data-occludable-job-id]"
                and name == "data-occludable-job-id"
            ):
                return "123"
            return None

        def is_visible(self, timeout: int) -> bool:
            return False

    class FakePage:
        url = "https://www.linkedin.com/jobs/search/?keywords=AI"

        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator(selector)

        def wait_for_timeout(self, _milliseconds: int) -> None:
            return None

        def evaluate(self, _script: str):
            return 0

    collector = LinkedInSearchHydrationDiagnosticsCollector()
    assert (
        _wait_for_search_results_hydration(
            FakePage(),
            max_wait_ms=0,
            query="AI @ Korea",
            diagnostics=collector,
        )
        == "timeout"
    )

    event = collector.events[-1]
    assert event.raw_signal_count == 5
    assert event.unique_candidate_count == 1
    assert event.raw_signal_count > event.unique_candidate_count


def test_linkedin_search_hydration_scroll_chooses_container_with_more_job_signals():
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        _scroll_search_results_incrementally,
    )

    class FakeLocator:
        def __init__(self, selector: str, page) -> None:
            self.selector = selector
            self.page = page

        def count(self) -> int:
            return int(self.selector in {".jobs-search-results-list", "main"})

        def nth(self, _index: int):
            return self

        def evaluate(self, script: str):
            if "scrollable" in script:
                return {
                    "jobSignalCount": (
                        5 if self.selector == "main" else 1
                    ),
                    "scrollHeight": 2000,
                    "clientHeight": 500,
                    "scrollTop": 0,
                    "scrollable": True,
                }
            if "scrollTopBefore" in script:
                self.page.scrolled_selector = self.selector
                return {
                    "scrollHeight": 2000,
                    "clientHeight": 500,
                    "scrollTopBefore": 0,
                    "scrollTopAfter": 500,
                }
            return None

    class FakePage:
        def __init__(self) -> None:
            self.scrolled_selector = ""

        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator(selector, self)

        def evaluate(self, _script: str):
            return {"found": False}

    page = FakePage()
    metrics = _scroll_search_results_incrementally(page)

    assert metrics.selected_scroll_container == "main"
    assert page.scrolled_selector == "main"


def test_linkedin_search_hydration_scroll_and_reload_are_bounded():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _wait_for_search_results_hydration,
    )

    class FakeLocator:
        def __init__(self, selector: str, page) -> None:
            self.selector = selector
            self.page = page

        @property
        def first(self):
            return self

        def count(self) -> int:
            return int(self.selector == "main")

        def nth(self, _index: int):
            return self

        def get_attribute(self, _name: str):
            return None

        def is_visible(self, timeout: int) -> bool:
            return bool(self.count())

        def evaluate(self, script: str):
            if "jobSignalCount" in script:
                return {
                    "jobSignalCount": 0,
                    "scrollHeight": 2000,
                    "clientHeight": 480,
                    "scrollTop": self.page.scroll_top,
                    "scrollable": True,
                }
            if "scrollTopBefore" in script:
                before = self.page.scroll_top
                self.page.scroll_top += 480
                self.page.scrolls += 1
                return {
                    "scrollHeight": 2000,
                    "clientHeight": 480,
                    "scrollTopBefore": before,
                    "scrollTopAfter": self.page.scroll_top,
                }
            return None

    class FakePage:
        url = "https://www.linkedin.com/jobs/search/?keywords=AI"

        def __init__(self) -> None:
            self.scrolls = 0
            self.scroll_top = 0
            self.reloads = 0
            self.waits = []

        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator(selector, self)

        def wait_for_timeout(self, milliseconds: int) -> None:
            self.waits.append(milliseconds)

        def reload(self, **_kwargs) -> None:
            self.reloads += 1

    page = FakePage()
    original_url = page.url

    assert _wait_for_search_results_hydration(page, max_wait_ms=1000) == "timeout"
    assert page.scrolls == 3
    assert page.reloads == 1
    assert sum(page.waits) == 1000
    assert page.url == original_url


def test_linkedin_search_hydration_stops_when_scroll_has_no_progress():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _wait_for_search_results_hydration,
    )

    class FakeLocator:
        def __init__(self, selector: str, page) -> None:
            self.selector = selector
            self.page = page

        @property
        def first(self):
            return self

        def count(self) -> int:
            return int(self.selector == "main")

        def nth(self, _index: int):
            return self

        def get_attribute(self, _name: str):
            return None

        def is_visible(self, timeout: int) -> bool:
            return False

        def evaluate(self, script: str):
            if "scrollable" in script:
                return {
                    "jobSignalCount": 0,
                    "scrollHeight": 2000,
                    "clientHeight": 480,
                    "scrollTop": 0,
                    "scrollable": True,
                }
            if "scrollTopBefore" in script:
                self.page.scrolls += 1
                return {
                    "scrollHeight": 2000,
                    "clientHeight": 480,
                    "scrollTopBefore": 0,
                    "scrollTopAfter": 0,
                }
            return None

    class FakePage:
        url = "https://www.linkedin.com/jobs/search/?keywords=AI"

        def __init__(self) -> None:
            self.scrolls = 0
            self.waits = []
            self.reloads = 0

        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator(selector, self)

        def wait_for_timeout(self, milliseconds: int) -> None:
            self.waits.append(milliseconds)

        def reload(self, **_kwargs) -> None:
            self.reloads += 1

        def evaluate(self, _script: str):
            return {"found": False}

    page = FakePage()
    assert _wait_for_search_results_hydration(page, max_wait_ms=1000) == "timeout"
    assert page.scrolls == 1
    assert page.reloads == 1


def test_linkedin_search_hydration_continues_bounded_when_scroll_moves_without_candidates():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _wait_for_search_results_hydration,
    )

    class FakeLocator:
        def __init__(self, selector: str, page) -> None:
            self.selector = selector
            self.page = page

        @property
        def first(self):
            return self

        def count(self) -> int:
            return int(self.selector == "main")

        def nth(self, _index: int):
            return self

        def get_attribute(self, _name: str):
            return None

        def is_visible(self, timeout: int) -> bool:
            return False

        def evaluate(self, script: str):
            if "scrollable" in script:
                return {
                    "jobSignalCount": 0,
                    "scrollHeight": 3000,
                    "clientHeight": 480,
                    "scrollTop": self.page.scroll_top,
                    "scrollable": True,
                }
            if "scrollTopBefore" in script:
                before = self.page.scroll_top
                self.page.scroll_top += 480
                self.page.scrolls += 1
                return {
                    "scrollHeight": 3000,
                    "clientHeight": 480,
                    "scrollTopBefore": before,
                    "scrollTopAfter": self.page.scroll_top,
                }
            return None

    class FakePage:
        url = "https://www.linkedin.com/jobs/search/?keywords=AI"

        def __init__(self) -> None:
            self.scrolls = 0
            self.scroll_top = 0
            self.waits = []

        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator(selector, self)

        def wait_for_timeout(self, milliseconds: int) -> None:
            self.waits.append(milliseconds)

        def reload(self, **_kwargs) -> None:
            return None

        def evaluate(self, _script: str):
            return {"found": False}

    page = FakePage()
    assert _wait_for_search_results_hydration(page, max_wait_ms=1000) == "timeout"
    assert page.scrolls == 3


def test_linkedin_visual_diagnostics_off_creates_no_artifacts(tmp_path, monkeypatch):
    from features.web_scraping.infrastructure.linkedin_visual_diagnostics import (
        visual_diagnostics_context,
    )

    monkeypatch.delenv("LINKEDIN_SEARCH_VISUAL_DIAGNOSTICS", raising=False)
    with visual_diagnostics_context(tmp_path / "audit") as collector:
        assert collector.start_run(SimpleNamespace(), query="AI @ Korea") is None

    assert not (tmp_path / "audit" / "visual-diagnostics").exists()
    assert collector.events == []


def test_linkedin_visual_diagnostics_bundle_is_local_relative_and_sanitized(
    tmp_path,
    monkeypatch,
):
    from features.web_scraping.infrastructure.linkedin_visual_diagnostics import (
        visual_diagnostics_context,
    )

    monkeypatch.setenv("LINKEDIN_SEARCH_VISUAL_DIAGNOSTICS", "true")

    class FakeTracing:
        def __init__(self) -> None:
            self.start_kwargs = None
            self.started_chunks = []
            self.stopped_chunks = []
            self.stop_called = False

        def start(self, **kwargs) -> None:
            self.start_kwargs = kwargs

        def start_chunk(self, **kwargs) -> None:
            self.started_chunks.append(kwargs)

        def stop_chunk(self, **kwargs) -> None:
            self.stopped_chunks.append(kwargs)
            Path(kwargs["path"]).write_text("trace", encoding="utf-8")

        def stop(self, **_kwargs) -> None:
            self.stop_called = True

    class FakeLocator:
        @property
        def first(self):
            return self

        def count(self) -> int:
            return 1

        def screenshot(self, *, path: str) -> None:
            Path(path).write_text("main", encoding="utf-8")

    class FakePage:
        def __init__(self) -> None:
            self.context = SimpleNamespace(tracing=FakeTracing())
            self.evaluate_calls = 0

        def screenshot(self, *, path: str, full_page: bool) -> None:
            assert full_page is True
            Path(path).write_text("full", encoding="utf-8")

        def locator(self, selector: str) -> FakeLocator:
            assert selector == "main"
            return FakeLocator()

        def evaluate(self, _script: str) -> dict:
            self.evaluate_calls += 1
            if self.evaluate_calls == 1:
                scroll_top = 415
            else:
                scroll_top = 895
            return {
                "schema_version": "1.0",
                "node_count": 10,
                "frame_count": 2,
                "body_scrollHeight": 2000,
                "viewport": {"width": 1280, "height": 720},
                "signal_counts": {"jobs_view": 0},
                "scrollables": [
                    {
                        "container_index": 2,
                        "tag": "div",
                        "class_tokens": ["scaffold-layout__list"],
                        "attributes": {
                            "data-job-id": "123",
                            "data-private-token": "secret",
                        },
                        "contains_main": False,
                        "scrollHeight": 1686,
                        "clientHeight": 700,
                        "scrollTop": scroll_top,
                        "anchor_count": 4,
                        "jobs_view_count": 0,
                        "urn_count": 0,
                        "data_job_id_count": 0,
                        "data_occludable_job_id_count": 0,
                        "job_signal_count": 0,
                    }
                ],
                "structural": [
                    {
                        "container_index": 2,
                        "tag": "div",
                        "class_tokens": ["scaffold-layout__list"],
                        "attributes": {"data-debug-id": "leak"},
                        "innerText": "must not persist",
                    }
                ],
            }

    audit_dir = tmp_path / "audit"
    page = FakePage()
    with visual_diagnostics_context(audit_dir) as collector:
        run = collector.start_run(page, query="AI @ Korea")
        assert run is not None
        run.capture_before(page)
        run.capture_after(page)
        run.stop_trace(page)
        run.finalize()
        assert collector.start_run(page, query="second query") is None

    visual_dir = audit_dir / "visual-diagnostics"
    expected_files = {
        "manifest.json",
        "before-full.png",
        "before-main.png",
        "structure-before.json",
        "after-full.png",
        "after-main.png",
        "structure-after.json",
        "trace.zip",
    }
    assert expected_files <= {item.name for item in visual_dir.iterdir()}
    assert page.context.tracing.start_kwargs == {
        "screenshots": True,
        "snapshots": True,
        "sources": False,
    }
    assert len(page.context.tracing.started_chunks) == 1
    assert len(page.context.tracing.stopped_chunks) == 1
    assert page.context.tracing.stop_called is True

    manifest = json.loads((visual_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["trace"]["sensitive_local_artifact"] is True
    assert manifest["trace"]["sources"] is False
    assert manifest["artifacts"]["structure_before"] == "structure-before.json"
    assert "/Users/" not in json.dumps(manifest)
    assert collector.events[0].manifest_path == "visual-diagnostics/manifest.json"

    before = json.loads((visual_dir / "structure-before.json").read_text(encoding="utf-8"))
    after = json.loads((visual_dir / "structure-after.json").read_text(encoding="utf-8"))
    assert before["scrollables"][0]["container_index"] == after["scrollables"][0]["container_index"] == 2
    assert after["scrollables"][0]["scrollTopBefore"] == 415
    assert after["scrollables"][0]["scrollTopAfter"] == 895
    assert after["scrollables"][0]["scroll_delta"] == 480
    assert after["scrollables"][0]["job_signal_count_before"] == 0
    assert after["scrollables"][0]["job_signal_count_after"] == 0
    serialized = json.dumps({"before": before, "after": after})
    assert "data-private-token" not in serialized
    assert "data-debug-id" not in serialized
    assert "innerText" not in serialized
    assert "must not persist" not in serialized


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


def test_linkedin_detail_hydration_waits_for_requested_description():
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
            return int(
                self.selector == ".jobs-description-content__text"
                and state["polls"] >= 2
            )

        def is_visible(self, timeout: int) -> bool:
            return bool(self.count())

        def inner_text(self, timeout: int) -> str:
            if self.selector == ".jobs-description-content__text":
                return "Description loaded asynchronously."
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

    assert _wait_for_detail_hydration(
        page,
        require_date=False,
        require_description=True,
        max_wait_ms=1000,
    ) == "ready"
    assert page.waits == [100, 150]


def test_linkedin_detail_hydration_ignores_description_placeholder_until_body_arrives():
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
            return int(
                self.selector.endswith("job-title h1")
                or self.selector == ".jobs-description-content__text"
            )

        def is_visible(self, timeout: int) -> bool:
            return bool(self.count())

        def inner_text(self, timeout: int) -> str:
            if self.selector == ".jobs-description-content__text":
                return "About the job" if state["polls"] < 2 else "Full description arrived."
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

    assert _wait_for_detail_hydration(
        page,
        require_date=False,
        require_description=True,
        max_wait_ms=1000,
    ) == "ready"
    assert page.waits == [100, 150]


def test_linkedin_detail_hydration_requires_date_and_description_together():
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
            if self.selector == ".jobs-description-content__text":
                return int(state["polls"] >= 1)
            return int(self.selector == "time" and state["polls"] >= 2)

        def is_visible(self, timeout: int) -> bool:
            return bool(self.count())

        def inner_text(self, timeout: int) -> str:
            if self.selector == "time":
                return "2 hours ago"
            return "Description arrived before the date."

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

    assert _wait_for_detail_hydration(
        page,
        require_date=True,
        require_description=True,
        max_wait_ms=1000,
    ) == "ready"
    assert page.waits == [100, 150]


def test_linkedin_detail_hydration_keeps_default_any_signal_behavior():
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _wait_for_detail_hydration,
    )

    class FakeLocator:
        def __init__(self, selector: str) -> None:
            self.selector = selector

        @property
        def first(self):
            return self

        def count(self) -> int:
            return int(self.selector.endswith("job-title h1"))

        def is_visible(self, timeout: int) -> bool:
            return bool(self.count())

        def inner_text(self, timeout: int) -> str:
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

    page = FakePage()

    assert _wait_for_detail_hydration(
        page,
        require_date=False,
        max_wait_ms=1000,
    ) == "ready"
    assert page.waits == []


@pytest.mark.parametrize(
    ("include_description", "expected_waits", "expected_description"),
    [
        (True, [100, 150], "Description loaded asynchronously."),
        (False, [], ""),
    ],
)
def test_linkedin_direct_detail_waits_only_for_requested_description(
    include_description,
    expected_waits,
    expected_description,
):
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _enrich_job_detail,
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
            return int(
                self.selector == ".jobs-description-content__text"
                and state["polls"] >= 2
            )

        def is_visible(self, timeout: int) -> bool:
            return bool(self.count())

        def inner_text(self, timeout: int) -> str:
            if self.selector == ".jobs-description-content__text":
                return "Description loaded asynchronously."
            return "AI Engineer"

        def get_attribute(self, name: str):
            return None

    class FakePage:
        url = "https://www.linkedin.com/jobs/view/111"

        def __init__(self) -> None:
            self.waits = []

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url

        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator(selector)

        def wait_for_timeout(self, milliseconds: int) -> None:
            self.waits.append(milliseconds)
            state["polls"] += 1

    candidate = LinkedInVacancyRecord(
        linkedin_job_id="111",
        title="AI Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/111",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        matched_terms=["ai"],
        published_at=datetime(2026, 7, 29, 10, 0, tzinfo=timezone.utc),
    )
    page = FakePage()

    with patch(
        "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
    ):
        enriched = _enrich_job_detail(
            page,
            candidate,
            include_description=include_description,
        )

    assert page.waits == expected_waits
    assert enriched.description_full_text == expected_description


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


def test_linkedin_detail_panel_hydration_ignores_description_placeholder_until_body_arrives():
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _wait_for_detail_panel_hydration,
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
                return "/jobs/view/111/"
            return None

        def inner_text(self, timeout: int) -> str:
            if self.selector == "time":
                return "1 hour ago"
            if self.selector == ".jobs-description-content__text":
                return "About this job" if state["polls"] < 2 else "Panel description arrived."
            return "AI Engineer"

    class FakePage:
        url = "https://www.linkedin.com/jobs/search/?keywords=AI&currentJobId=111"

        def __init__(self) -> None:
            self.waits = []

        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator(selector)

        def wait_for_timeout(self, milliseconds: int) -> None:
            self.waits.append(milliseconds)
            state["polls"] += 1

    candidate = LinkedInVacancyRecord(
        linkedin_job_id="111",
        title="AI Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/111",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        matched_terms=["ai"],
    )
    page = FakePage()

    assert _wait_for_detail_panel_hydration(
        page,
        candidate,
        include_description=True,
        max_wait_ms=1000,
    ) == "ready"
    assert page.waits == [100, 150]


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
    assert error.value.reason == "detail_click_failed"
    assert error.value.safe_label == "ValueError:runtime"


def test_linkedin_detail_panel_job_id_comparison_is_normalized_and_exact():
    from features.web_scraping.infrastructure.linkedin_detail_panel import (
        _detail_panel_job_id_matches,
    )

    class EmptyLocator:
        def count(self) -> int:
            return 0

        def nth(self, _index: int):
            return self

    class FakePage:
        def __init__(self, url: str) -> None:
            self.url = url

        def locator(self, _selector: str) -> EmptyLocator:
            return EmptyLocator()

    assert _detail_panel_job_id_matches(
        FakePage("https://www.linkedin.com/jobs/search/?currentJobId=00111"),
        "111",
    )
    assert not _detail_panel_job_id_matches(
        FakePage("https://www.linkedin.com/jobs/search/?currentJobId=1112"),
        "111",
    )


def test_linkedin_direct_detail_fallback_uses_exact_url_and_restores_search():
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_jobs_pipeline import (
        _run_guarded_direct_detail_fallback,
    )
    from features.web_scraping.infrastructure.linkedin_url_policy import (
        validate_linkedin_jobs_url,
    )

    source_url = "https://www.linkedin.com/jobs/search/?keywords=AI"
    candidate = LinkedInVacancyRecord(
        linkedin_job_id="00111",
        title="AI Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/ai-engineer-111",
        source_url=source_url,
        matched_terms=["ai"],
    )

    class FakePage:
        url = source_url

        def __init__(self) -> None:
            self.navigations = []

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url
            self.navigations.append(url)

    page = FakePage()
    direct_calls = []
    warnings = []

    def enrich(target_page, record, **_kwargs):
        direct_calls.append(record.canonical_url)
        target_page.url = record.canonical_url
        return record

    result = _run_guarded_direct_detail_fallback(
        page,
        candidate,
        source_url=source_url,
        include_description=False,
        warnings=warnings,
        validate_jobs_url=validate_linkedin_jobs_url,
        validate_authenticated_page=lambda _page: None,
        wait_for_search_results_hydration=lambda _page: "results",
        enrich_job_detail=enrich,
        safe_error_label=lambda exc: type(exc).__name__,
        terminal_error_types=(),
    )

    assert result.linkedin_job_id == "111"
    assert direct_calls == ["https://www.linkedin.com/jobs/view/ai-engineer-111"]
    assert page.navigations == [source_url]
    assert page.url == source_url
    assert warnings == [
        "direct_detail_fallback_used:111",
        "active_detail_date_selected:111:missing:date_none:candidates_0:score_0",
    ]


def test_linkedin_direct_detail_fallback_merges_active_detail_metadata(monkeypatch):
    from datetime import datetime, timezone

    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure import linkedin_jobs_pipeline as pipeline
    from features.web_scraping.infrastructure.linkedin_url_policy import (
        validate_linkedin_jobs_url,
    )

    source_url = "https://www.linkedin.com/jobs/search/?keywords=AI"
    candidate = LinkedInVacancyRecord(
        linkedin_job_id="4453249078",
        title="Machine Learning Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/4453249078",
        source_url=source_url,
        matched_terms=["ai"],
    )
    published = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)

    class FakePage:
        url = source_url

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url

    monkeypatch.setattr(
        pipeline,
        "_wait_for_active_detail_metadata",
        lambda *_args, **_kwargs: (
            "Machine Learning Engineer",
            "hace 11 horas",
            published,
            "medium",
            True,
        ),
    )
    result = pipeline._run_guarded_direct_detail_fallback(
        FakePage(),
        candidate,
        source_url=source_url,
        include_description=False,
        warnings=[],
        validate_jobs_url=validate_linkedin_jobs_url,
        validate_authenticated_page=lambda _page: None,
        wait_for_search_results_hydration=lambda _page: "results",
        enrich_job_detail=lambda _page, record, **_kwargs: record,
        safe_error_label=lambda exc: type(exc).__name__,
        terminal_error_types=(),
    )

    assert result.linkedin_job_id == "4453249078"
    assert result.posted_at_text == "hace 11 horas"
    assert result.published_at == published
    assert result.freshness_confidence == "medium"
    assert result.is_within_24_hours is True


def test_linkedin_direct_detail_fallback_reports_search_restore_failure():
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_jobs_pipeline import (
        _run_guarded_direct_detail_fallback,
    )
    from features.web_scraping.infrastructure.linkedin_url_policy import (
        validate_linkedin_jobs_url,
    )

    source_url = "https://www.linkedin.com/jobs/search/?keywords=AI"
    candidate = LinkedInVacancyRecord(
        linkedin_job_id="111",
        title="AI Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/111",
        source_url=source_url,
        matched_terms=["ai"],
    )
    page = MagicMock()
    warnings = []

    _run_guarded_direct_detail_fallback(
        page,
        candidate,
        source_url=source_url,
        include_description=False,
        warnings=warnings,
        validate_jobs_url=validate_linkedin_jobs_url,
        validate_authenticated_page=lambda _page: None,
        wait_for_search_results_hydration=lambda _page: "timeout",
        enrich_job_detail=lambda _page, record, **_kwargs: record,
        safe_error_label=lambda exc: type(exc).__name__,
        terminal_error_types=(),
    )

    assert warnings == [
        "direct_detail_fallback_used:111",
        "active_detail_date_selected:111:missing:date_none:candidates_0:score_0",
        "list_not_hydrated:111",
        "search_restore_failed:111:list_not_hydrated",
    ]


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



def test_linkedin_extracts_card_local_reposted_time_text():
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        extract_card_local_posted_time_text,
    )
    from features.web_scraping.infrastructure.linkedin_parser import (
        parse_linkedin_relative_time,
    )

    posted = extract_card_local_posted_time_text(
        "Machine Learning Engineer GOWARD Seoul Publicado de nuevo hace 23 minutos Visto"
    )

    assert posted == "Publicado de nuevo hace 23 minutos"
    published_at, confidence, within_24h = parse_linkedin_relative_time(posted)
    assert published_at is not None
    assert confidence == "medium"
    assert within_24h is True


def test_linkedin_extracts_card_local_reposted_time_text_english():
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        extract_card_local_posted_time_text,
    )

    assert (
        extract_card_local_posted_time_text(
            "AI Architect Company Seoul Reposted 2 hours ago Viewed"
        )
        == "Reposted 2 hours ago"
    )

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
    assert snapshot.search_hydration_diagnostics == []
    assert snapshot.visual_diagnostics == []


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


def test_linkedin_service_runs_bootstrap_and_retries_in_dev_on_auth_failure(
    tmp_path,
    monkeypatch,
):
    from features.web_scraping.application import linkedin_audit
    from features.web_scraping.application import linkedin_service
    from features.web_scraping.application.linkedin_service import (
        run_linkedin_jobs_vertical,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        LinkedInAuthRequiredError,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LINKEDIN_AUTO_BOOTSTRAP_ON_AUTH_FAILURE", "1")
    monkeypatch.setattr(
        linkedin_audit,
        "get_request_runtime_config",
        lambda: SimpleNamespace(session_id="sess-auth", request_id="req-auth"),
    )

    scrape_results = [
        LinkedInAuthRequiredError("login required"),
        ([], [], [], [], []),
    ]

    def fake_scrape(_request):
        item = scrape_results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    store = MagicMock()
    store.load_browser_metadata.return_value = {
        "profile_path": str(tmp_path / "browser-profile")
    }
    store.resolve_profile_path.return_value = tmp_path / "browser-profile"
    recovery_events = []

    def fake_close_contexts(*, profile_path):
        recovery_events.append(("close", profile_path))
        return 1

    def fake_run(_command, **_kwargs):
        recovery_events.append(("run", None))
        return SimpleNamespace(returncode=0)

    with patch(
        "features.web_scraping.application.linkedin_service.scrape_linkedin_jobs",
        side_effect=fake_scrape,
    ) as scrape, patch(
        "features.web_scraping.application.linkedin_service.subprocess.run",
        side_effect=fake_run,
    ) as run, patch(
        "features.web_scraping.application.linkedin_service.LinkedInSessionStore",
        return_value=store,
    ), patch(
        "features.web_scraping.application.linkedin_service.close_reusable_authenticated_contexts",
        side_effect=fake_close_contexts,
    ) as close_contexts:
        result = run_linkedin_jobs_vertical("Buscá vacantes LinkedIn de AI de hoy")

    assert result.status == "extraction_incomplete"
    assert "auth_refresh_bootstrap_started" in result.warnings
    assert "auth_refresh_runtime_session_closed" in result.warnings
    assert "auth_refresh_bootstrap_completed" in result.warnings
    assert "auth_refresh_retry_started" in result.warnings
    assert scrape.call_count == 2
    assert recovery_events == [
        ("close", tmp_path / "browser-profile"),
        ("run", None),
    ]
    close_contexts.assert_called_once_with(profile_path=tmp_path / "browser-profile")
    run.assert_called_once()
    command = run.call_args.args[0]
    assert command[1:] == [
        "scripts/bootstrap_linkedin_session.py",
        "--browser",
        "brave",
        "--executable-path",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        "--observe-ready",
        "--ready-timeout-seconds",
        "300",
    ]


def test_linkedin_service_reports_bootstrap_when_retry_still_requires_auth(
    tmp_path,
    monkeypatch,
):
    from features.web_scraping.application import linkedin_audit
    from features.web_scraping.application import linkedin_service
    from features.web_scraping.application.linkedin_service import (
        run_linkedin_jobs_vertical,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        LinkedInAuthRequiredError,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LINKEDIN_AUTO_BOOTSTRAP_ON_AUTH_FAILURE", "1")
    monkeypatch.setattr(
        linkedin_audit,
        "get_request_runtime_config",
        lambda: SimpleNamespace(session_id="sess-auth", request_id="req-auth"),
    )

    with patch(
        "features.web_scraping.application.linkedin_service.scrape_linkedin_jobs",
        side_effect=LinkedInAuthRequiredError("login required"),
    ) as scrape, patch(
        "features.web_scraping.application.linkedin_service.subprocess.run",
        return_value=SimpleNamespace(returncode=0),
    ):
        result = run_linkedin_jobs_vertical("Buscá vacantes LinkedIn de AI de hoy")

    assert result.status == "auth_required"
    assert "auth_refresh_bootstrap_completed" in result.warnings
    assert "auth_refresh_retry_started" in result.warnings
    assert scrape.call_count == 2
    assert "Abrí automáticamente el bootstrap" in result.user_summary


def test_linkedin_service_does_not_retry_when_observer_exits_profile_in_use(
    tmp_path,
    monkeypatch,
):
    from features.web_scraping.application import linkedin_audit
    from features.web_scraping.application.linkedin_service import (
        run_linkedin_jobs_vertical,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        LinkedInAuthRequiredError,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LINKEDIN_AUTO_BOOTSTRAP_ON_AUTH_FAILURE", "1")
    monkeypatch.setattr(
        linkedin_audit,
        "get_request_runtime_config",
        lambda: SimpleNamespace(session_id="sess-exit-4", request_id="req-exit-4"),
    )

    with patch(
        "features.web_scraping.application.linkedin_service.scrape_linkedin_jobs",
        side_effect=LinkedInAuthRequiredError("login required"),
    ) as scrape, patch(
        "features.web_scraping.application.linkedin_service.subprocess.run",
        return_value=SimpleNamespace(returncode=4),
    ):
        result = run_linkedin_jobs_vertical("Buscá vacantes LinkedIn de AI de hoy")

    assert result.status == "auth_required"
    assert "auth_refresh_bootstrap_failed:exit_4" in result.warnings
    assert "auth_refresh_retry_started" not in result.warnings
    assert scrape.call_count == 1


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
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        minutes_to_tpr,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        build_linkedin_search_queries,
    )

    queries = build_linkedin_search_queries("Argentina")

    assert minutes_to_tpr(1440) == "r86400"
    assert minutes_to_tpr(60) == "r3600"
    assert minutes_to_tpr(30) == "r1800"
    assert queries
    assert all("f_TPR=r86400" in url for _, url in queries)
    assert all("sortBy=DD" in url for _, url in queries)
    assert all("location=Argentina" in url for _, url in queries)


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5, "1440", None])
def test_linkedin_minutes_to_tpr_rejects_invalid_values(invalid):
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        minutes_to_tpr,
    )

    with pytest.raises(ValueError):
        minutes_to_tpr(invalid)


def test_linkedin_search_url_retains_allowed_query_params():
    from urllib.parse import parse_qs, urlparse

    from features.web_scraping.infrastructure.linkedin_url_policy import (
        canonicalize_linkedin_url,
        validate_linkedin_jobs_url,
    )

    url = validate_linkedin_jobs_url(
        "https://www.linkedin.com/jobs/search/?keywords=AI&location=Japan&f_TPR=r86400"
    )
    params = parse_qs(urlparse(url).query)

    assert params["keywords"] == ["AI"]
    assert params["location"] == ["Japan"]
    assert params["f_TPR"] == ["r86400"]
    with pytest.raises(ValueError, match="path_not_allowed"):
        canonicalize_linkedin_url(url)


def test_linkedin_query_builder_consolidates_topics_per_location():
    from urllib.parse import parse_qs, urlparse

    from features.web_scraping.infrastructure.linkedin_scraper import (
        CONSOLIDATED_QUERY_PLANS,
        build_linkedin_search_queries,
    )

    queries = build_linkedin_search_queries(["South Korea", "Japan"])

    assert len(CONSOLIDATED_QUERY_PLANS) == 2
    assert len(queries) == 6
    assert sum("location=Japan" in url for _, url in queries) == 3
    assert sum("location=South+Korea" in url for _, url in queries) == 3
    assert all("f_TPR=r86400" in url and "sortBy=DD" in url for _, url in queries)
    assert [label for label, _url in queries] == [
        "AI/ML/Data/GenAI @ South Korea",
        "AI/ML/Data/GenAI @ Japan",
        "AI Agents/Product/Architecture @ South Korea",
        "AI Agents/Product/Architecture @ Japan",
        "Country-focused AI/Data @ South Korea",
        "Country-focused AI/Data @ Japan",
    ]
    primary_keywords = [
        parse_qs(urlparse(url).query)["keywords"][0]
        for _, url in queries[:2]
    ]
    assert primary_keywords[0] == primary_keywords[1]
    country_focused_keywords = [
        parse_qs(urlparse(url).query)["keywords"][0]
        for label, url in queries
        if label.startswith("Country-focused")
    ]
    assert "South Korea" in country_focused_keywords[0]
    assert "Japón" in country_focused_keywords[1]
    assert "Tokyo" in country_focused_keywords[1]
    assert "Japan" not in country_focused_keywords[1]
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


def test_linkedin_semantic_dedupe_collapses_recruiter_campaign_variants():
    from datetime import datetime, timezone

    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _dedupe_linkedin_vacancies_semantically,
    )

    shared_prefix = "\n".join(
        [
            "Acerca del empleo",
            "About The Job",
            "The client is hiring for this role. Harper is an AI career agent that works for you — the candidate.",
            "Skip the application form: just talk to Harper, and it puts strong candidates straight in front of the client.",
        ]
    )
    ai_engineer = LinkedInVacancyRecord(
        linkedin_job_id="4435039660",
        title="Applied AI / ML Engineer at a frontier open-source AI lab",
        company_name="Harper",
        location="Seúl, Corea del Sur",
        canonical_url="https://www.linkedin.com/jobs/view/4435039660",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI&location=South+Korea",
        description_full_text="\n".join(
            [
                shared_prefix,
                "Job Title",
                "Applied AI, Machine Learning Engineer",
                "Company Description",
                "One of the world's most important AI labs — European, building open-weight, open-source frontier models that go head-to-head with the best closed labs. Teams across France, the US, UK, Germany, and Singapore — now building in Seoul.",
                "What You'll Do",
                "Take GenAI from research into production and own customer onboarding.",
            ]
        ),
        published_at=datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc),
    )
    research_engineer = LinkedInVacancyRecord(
        linkedin_job_id="4435053385",
        title="Applied Scientist / Research Engineer at an Nvidia-backed AI Lab",
        company_name="Harper",
        location="Seúl, Corea del Sur",
        canonical_url="https://www.linkedin.com/jobs/view/4435053385",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI&location=South+Korea",
        description_full_text="\n".join(
            [
                shared_prefix,
                "Job Title",
                "Senior/Staff Applied Scientist / Research Engineer",
                "Company Description",
                "An Nvidia-backed European AI lab, valued at ~$14B — one of the world's most important, building open-weight, open-source frontier models that go head-to-head with the best closed labs. Teams across France, the US, UK, Germany, and Singapore — now building in Seoul.",
                "What You'll Do",
                "Run pre-training, post-training, and deployment of SOTA models on clusters with thousands of GPUs.",
            ]
        ),
        published_at=datetime(2026, 8, 2, 13, 0, tzinfo=timezone.utc),
    )

    deduped, warnings = _dedupe_linkedin_vacancies_semantically(
        [ai_engineer, research_engineer]
    )

    assert [record.linkedin_job_id for record in deduped] == ["4435053385"]
    assert warnings == [
        "recruiter_campaign_duplicate_dropped:4435039660:kept:4435053385"
    ]


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
        "Country-focused AI/Data @ South Korea",
        "Country-focused AI/Data @ Japan",
    ]
    assert len(queries) == 6
    params = [parse_qs(urlparse(url).query) for _label, url in queries]
    assert [query["location"] for query in params] == [
        ["South Korea"],
        ["Japan"],
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
    assert [item.reason for item in rejected] == ["selector_drift/card_missing"]


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
    assert rejected == []
    assert panel_detail.call_count == 1
    direct_detail.assert_not_called()
    assert not any("/jobs/view/" in url for url in navigations)


def test_linkedin_direct_detail_fallback_fetches_deduped_candidate_once(
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

    monkeypatch.setenv("LINKEDIN_DIRECT_DETAIL_FALLBACK", "true")
    monkeypatch.setenv("LINKEDIN_DETAIL_BUDGET", "3")
    monkeypatch.setenv("LINKEDIN_MAX_QUERIES_PER_LOCATION", "2")
    candidate = LinkedInVacancyRecord(
        linkedin_job_id="111",
        title="AI Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/111",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        matched_terms=["ai"],
    )

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
        context = SimpleNamespace(
            request=SimpleNamespace(get=lambda *_args, **_kwargs: None)
        )

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

    recent = candidate.model_copy(
        update={
            "company_name": "Example AI",
            "location": "Tokyo, Japan",
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
                (
                    "Primary @ Japan",
                    "https://www.linkedin.com/jobs/search/?keywords=AI",
                ),
                (
                    "Fallback @ Japan",
                    "https://www.linkedin.com/jobs/search/?keywords=ML",
                ),
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
            return_value=None,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._enrich_job_detail",
            return_value=recent,
        ) as direct_detail,
    ):
        records, rejected, _timings, warnings, _queries = scrape_linkedin_jobs(
            LinkedInJobsRequest(
                query="AI jobs Japan",
                locations=["Japan"],
                max_results=5,
                include_description=False,
            ),
            session_store=FakeStore(),  # type: ignore[arg-type]
        )

    assert [record.linkedin_job_id for record in records] == ["111"]
    assert rejected == []
    assert direct_detail.call_count == 1
    assert warnings.count("direct_detail_fallback_used:111") == 1
    assert any("selector_drift/card_missing" in warning for warning in warnings)


@pytest.mark.parametrize(
    ("hydration_state", "expected_direct_calls"),
    [("timeout", 3), ("empty", 0), ("results", 0)],
)
def test_linkedin_standalone_fallback_is_plural_timeout_only_and_restores_search(
    monkeypatch,
    hydration_state,
    expected_direct_calls,
):
    from features.web_scraping.domain.linkedin_models import LinkedInJobsRequest
    from features.web_scraping.infrastructure.linkedin_scraper import (
        scrape_linkedin_jobs,
    )

    monkeypatch.setenv("LINKEDIN_DIRECT_DETAIL_FALLBACK", "true")
    monkeypatch.setenv("LINKEDIN_DETAIL_BUDGET", "3")
    monkeypatch.setenv("LINKEDIN_MAX_QUERIES_PER_LOCATION", "1")
    source_url = "https://www.linkedin.com/jobs/search/?keywords=AI"
    restored_source_url = "https://www.linkedin.com/jobs/search?keywords=AI"
    standalone_html = """
    <main>
      <a href="/jobs/view/ai-engineer-111/?trackingId=secret"
         aria-label="AI Engineer"></a>
      <a href="/jobs/view/222">Machine Learning Engineer</a>
      <a href="/jobs/view/333">Data Scientist</a>
    </main>
    """

    class FakePage:
        url = "https://www.linkedin.com/jobs/"

        def __init__(self) -> None:
            self.navigations: list[str] = []

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url
            self.navigations.append(url)

        def wait_for_timeout(self, _milliseconds: int) -> None:
            return None

        def content(self) -> str:
            return standalone_html

        def is_closed(self) -> bool:
            return False

    page = FakePage()

    class FakeSession:
        def __init__(self) -> None:
            self.page = page

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

    def enrich_direct(target_page, record, **_kwargs):
        target_page.goto(record.canonical_url)
        description = (
            ""
            if record.linkedin_job_id == "111"
            else f"Build AI systems for job {record.linkedin_job_id}."
        )
        return record.model_copy(
            update={
                "company_name": "Example AI",
                "location": "Tokyo, Japan",
                "workplace_type": "hybrid",
                "description_full_text": description,
                "posted_at_text": "2 hours ago",
                "published_at": datetime(
                    2026,
                    7,
                    29,
                    10,
                    0,
                    tzinfo=timezone.utc,
                ),
                "freshness_confidence": "medium",
                "is_within_24_hours": True,
            }
        )

    hydration_results = (
        ["timeout", "results", "results", "results"]
        if hydration_state == "timeout"
        else [hydration_state]
    )
    card_lookup = MagicMock(return_value=None)
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
            side_effect=hydration_results,
        ) as wait_hydration,
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.build_linkedin_search_queries",
            return_value=[("Primary @ Japan", source_url)],
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._safe_job_card_link",
            side_effect=card_lookup,
        ) as safe_card_link,
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._enrich_job_detail",
            side_effect=enrich_direct,
        ) as direct_detail,
    ):
        records, _rejected, timings, warnings, _queries = scrape_linkedin_jobs(
            LinkedInJobsRequest(
                query="AI jobs Japan",
                locations=["Japan"],
                max_results=5,
                include_description=True,
            ),
            session_store=FakeStore(),  # type: ignore[arg-type]
        )

    assert direct_detail.call_count == expected_direct_calls
    safe_card_link.assert_not_called()
    assert wait_hydration.call_count == (4 if hydration_state == "timeout" else 1)
    if hydration_state == "timeout":
        assert [record.linkedin_job_id for record in records] == ["222", "333"]
        assert [item.reason for item in _rejected] == [
            "missing_description_full_text"
        ]
        assert page.url == restored_source_url
        assert page.navigations[-1] == restored_source_url
        assert "https://www.linkedin.com/jobs/view/ai-engineer-111" in (
            page.navigations
        )
        assert warnings.count("direct_detail_fallback_used:111") == 1
        assert warnings.count("direct_detail_fallback_used:222") == 1
        assert warnings.count("direct_detail_fallback_used:333") == 1
        assert timings[0].diagnostics.candidate_count == 3
        assert timings[0].diagnostics.parseable_candidate_count == 3
        assert timings[0].diagnostics.discard_reasons == {
            "standalone_link_fallback": 3
        }
    else:
        assert records == []
        assert page.navigations.count(source_url) == 1
        assert not any(
            warning.startswith("direct_detail_fallback_used:")
            for warning in warnings
        )
        assert "standalone_link_fallback" not in (
            timings[0].diagnostics.discard_reasons
        )


def test_linkedin_timeout_no_signal_static_probe_recovers_unique_candidates(
    monkeypatch,
):
    from features.web_scraping.domain.linkedin_models import (
        LinkedInJobsRequest,
        LinkedInParseDiagnostics,
        LinkedInVacancyRecord,
    )
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        search_hydration_diagnostics_context,
    )
    from features.web_scraping.infrastructure.linkedin_static_probe_diagnostics import (
        static_probe_diagnostics_context,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        scrape_linkedin_jobs,
    )

    monkeypatch.setenv("LINKEDIN_DETAIL_BUDGET", "0")
    monkeypatch.setenv("LINKEDIN_MAX_QUERIES_PER_LOCATION", "1")
    source_url = "https://www.linkedin.com/jobs/search/?keywords=AI"

    class FakePage:
        url = "https://www.linkedin.com/jobs/"

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url

        def wait_for_timeout(self, _milliseconds: int) -> None:
            return None

        def content(self) -> str:
            return "<main></main>"

        def is_closed(self) -> bool:
            return False

    class FakeSession:
        page = FakePage()
        context = SimpleNamespace(
            request=SimpleNamespace(get=lambda *_args, **_kwargs: None)
        )

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

    def complete_static_record(job_id: str, title: str) -> LinkedInVacancyRecord:
        return LinkedInVacancyRecord(
            linkedin_job_id=job_id,
            title=title,
            company_name="Example AI",
            location="Seoul, South Korea",
            workplace_type="hybrid",
            posted_at_text="1 hour ago",
            published_at=datetime(2026, 7, 29, 10, 0, tzinfo=timezone.utc),
            freshness_confidence="high",
            is_within_24_hours=True,
            canonical_url=f"https://www.linkedin.com/jobs/view/{job_id}",
            source_url=source_url,
            hard_skills=["Python"],
            foreigner_acceptance="yes",
            matched_terms=["ai"],
        )

    static_records = [
        complete_static_record("111", "AI Engineer"),
        complete_static_record("222", "Machine Learning Engineer"),
        complete_static_record("333", "Data Scientist"),
    ]

    def wait_timeout_no_signals(_page, *, query="unknown", diagnostics=None):
        assert diagnostics is not None
        diagnostics.record(
            query=query,
            elapsed_ms=6500,
            card_count=0,
            href_count=0,
            empty_state_visible=False,
            auth_checkpoint_visible=False,
            outcome="timeout",
        )
        return "timeout"

    def static_probe(_session, **kwargs):
        assert kwargs["source_url"] == source_url
        assert kwargs["allow_standalone_fallback"] is True
        return SimpleNamespace(
            records=static_records,
            diagnostics=LinkedInParseDiagnostics(
                href_count=3,
                candidate_count=3,
                parseable_candidate_count=3,
                discard_reasons={"standalone_link_fallback": 3},
            ),
            status_code=200,
            category="ok",
            detail="",
        )

    with (
        search_hydration_diagnostics_context(),
        static_probe_diagnostics_context() as static_diagnostics,
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.open_persistent_authenticated_context",
            return_value=FakeSession(),
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._wait_for_search_results_hydration",
            side_effect=wait_timeout_no_signals,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.build_linkedin_search_queries",
            return_value=[("Primary @ South Korea", source_url)],
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._probe_linkedin_search_with_authenticated_request",
            side_effect=static_probe,
        ) as probe,
    ):
        records, rejected, timings, warnings, _queries = scrape_linkedin_jobs(
            LinkedInJobsRequest(
                query="AI jobs South Korea",
                locations=["South Korea"],
                max_results=5,
                include_description=False,
            ),
            session_store=FakeStore(),  # type: ignore[arg-type]
        )

    assert probe.call_count == 1
    assert [record.linkedin_job_id for record in records] == ["111", "222", "333"]
    assert rejected == []
    assert timings[0].diagnostics.parseable_candidate_count == 3
    assert "search_static_probe_attempt:South Korea" in warnings
    assert any(
        warning.startswith(
            "search_static_probe_result:South Korea:status_200:ok:candidates_3"
        )
        for warning in warnings
    )
    assert [
        (event.kind, event.outcome, event.candidate_count, event.accepted_count)
        for event in static_diagnostics.events
    ] == [("search_static_probe", "ok", 3, 3)]


@pytest.mark.parametrize("hydration_state", ["empty", "results"])
def test_linkedin_static_search_probe_is_timeout_no_signal_only(
    monkeypatch,
    hydration_state,
):
    from features.web_scraping.domain.linkedin_models import LinkedInJobsRequest
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        search_hydration_diagnostics_context,
    )
    from features.web_scraping.infrastructure.linkedin_static_probe_diagnostics import (
        static_probe_diagnostics_context,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        scrape_linkedin_jobs,
    )

    monkeypatch.setenv("LINKEDIN_DETAIL_BUDGET", "0")
    monkeypatch.setenv("LINKEDIN_MAX_QUERIES_PER_LOCATION", "1")
    source_url = "https://www.linkedin.com/jobs/search/?keywords=AI"

    class FakePage:
        url = "https://www.linkedin.com/jobs/"

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url

        def wait_for_timeout(self, _milliseconds: int) -> None:
            return None

        def content(self) -> str:
            return "<main></main>"

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

    def wait_terminal(_page, *, query="unknown", diagnostics=None):
        assert diagnostics is not None
        diagnostics.record(
            query=query,
            elapsed_ms=250,
            card_count=1 if hydration_state == "results" else 0,
            href_count=1 if hydration_state == "results" else 0,
            empty_state_visible=hydration_state == "empty",
            auth_checkpoint_visible=False,
            outcome=hydration_state,
        )
        return hydration_state

    with (
        search_hydration_diagnostics_context(),
        static_probe_diagnostics_context() as static_diagnostics,
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.open_persistent_authenticated_context",
            return_value=FakeSession(),
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._wait_for_search_results_hydration",
            side_effect=wait_terminal,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.build_linkedin_search_queries",
            return_value=[("Primary @ South Korea", source_url)],
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._probe_linkedin_search_with_authenticated_request",
        ) as probe,
    ):
        records, _rejected, _timings, _warnings, _queries = scrape_linkedin_jobs(
            LinkedInJobsRequest(
                query="AI jobs South Korea",
                locations=["South Korea"],
                max_results=5,
                include_description=False,
            ),
            session_store=FakeStore(),  # type: ignore[arg-type]
        )

    assert records == []
    probe.assert_not_called()
    assert static_diagnostics.events == []


def test_linkedin_static_probe_candidate_uses_static_detail_before_direct_detail(
    monkeypatch,
):
    from features.web_scraping.domain.linkedin_models import (
        LinkedInJobsRequest,
        LinkedInParseDiagnostics,
        LinkedInVacancyRecord,
    )
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        search_hydration_diagnostics_context,
    )
    from features.web_scraping.infrastructure.linkedin_static_probe_diagnostics import (
        static_probe_diagnostics_context,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import (
        scrape_linkedin_jobs,
    )

    monkeypatch.setenv("LINKEDIN_DIRECT_DETAIL_FALLBACK", "true")
    monkeypatch.setenv("LINKEDIN_DETAIL_BUDGET", "3")
    monkeypatch.setenv("LINKEDIN_MAX_QUERIES_PER_LOCATION", "1")
    source_url = "https://www.linkedin.com/jobs/search/?keywords=AI"
    candidate = LinkedInVacancyRecord(
        linkedin_job_id="111",
        title="AI Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/111",
        source_url=source_url,
        matched_terms=["ai"],
    )
    enriched = candidate.model_copy(
        update={
            "company_name": "Example AI",
            "location": "Seoul, South Korea",
            "workplace_type": "hybrid",
            "description_full_text": "Build production AI systems in Korea.",
            "description_excerpt": "Build production AI systems in Korea.",
            "posted_at_text": "2026-07-29T10:00:00Z",
            "published_at": datetime(2026, 7, 29, 10, 0, tzinfo=timezone.utc),
            "freshness_confidence": "high",
            "is_within_24_hours": True,
            "hard_skills": ["Python"],
            "foreigner_acceptance": "yes",
        }
    )

    class FakePage:
        url = "https://www.linkedin.com/jobs/"

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url

        def wait_for_timeout(self, _milliseconds: int) -> None:
            return None

        def content(self) -> str:
            return "<main></main>"

        def is_closed(self) -> bool:
            return False

    class FakeSession:
        page = FakePage()
        context = SimpleNamespace(
            request=SimpleNamespace(get=lambda *_args, **_kwargs: None)
        )

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

    def wait_hydration(_page, *, query="unknown", diagnostics=None):
        assert diagnostics is not None
        if query.startswith("restore:"):
            diagnostics.record(
                query=query,
                elapsed_ms=250,
                card_count=1,
                href_count=1,
                empty_state_visible=False,
                auth_checkpoint_visible=False,
                outcome="results",
            )
            return "results"
        diagnostics.record(
            query=query,
            elapsed_ms=6500,
            card_count=0,
            href_count=0,
            empty_state_visible=False,
            auth_checkpoint_visible=False,
            outcome="timeout",
        )
        return "timeout"

    search_probe = SimpleNamespace(
        records=[candidate],
        diagnostics=LinkedInParseDiagnostics(
            href_count=1,
            candidate_count=1,
            parseable_candidate_count=1,
            discard_reasons={"standalone_link_fallback": 1},
        ),
        status_code=200,
        category="ok",
        detail="",
    )
    detail_probe = SimpleNamespace(
        record=enriched,
        status_code=200,
        category="ok",
        detail="",
    )

    with (
        search_hydration_diagnostics_context(),
        static_probe_diagnostics_context() as static_diagnostics,
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.open_persistent_authenticated_context",
            return_value=FakeSession(),
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._wait_for_search_results_hydration",
            side_effect=wait_hydration,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.build_linkedin_search_queries",
            return_value=[("Primary @ South Korea", source_url)],
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._probe_linkedin_search_with_authenticated_request",
            return_value=search_probe,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._enrich_job_detail",
            return_value=candidate,
        ) as direct_detail,
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._probe_linkedin_detail_with_authenticated_request",
            return_value=detail_probe,
        ) as static_detail,
    ):
        records, rejected, _timings, warnings, _queries = scrape_linkedin_jobs(
            LinkedInJobsRequest(
                query="AI jobs South Korea",
                locations=["South Korea"],
                max_results=5,
                include_description=True,
            ),
            session_store=FakeStore(),  # type: ignore[arg-type]
        )

    assert [record.linkedin_job_id for record in records] == ["111"]
    assert records[0].description_full_text == "Build production AI systems in Korea."
    assert rejected == []
    assert direct_detail.call_count == 0
    assert static_detail.call_count == 1
    assert "detail_static_probe_attempt:111" in warnings
    assert "detail_static_probe_result:111:status_200:ok" in warnings
    assert [
        (event.kind, event.outcome, event.candidate_count, event.accepted_count)
        for event in static_diagnostics.events
    ] == [
        ("search_static_probe", "ok", 1, 1),
        ("detail_static_probe", "ok", 1, 1),
    ]



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
                    if record.linkedin_job_id in {"101", "102"}
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



def test_linkedin_rejected_records_dedupe_without_job_ids():
    from features.web_scraping.domain.linkedin_models import LinkedInRejectedRecord
    from features.web_scraping.infrastructure.linkedin_jobs_pipeline import (
        _dedupe_rejected_records,
    )

    deduped, warnings = _dedupe_rejected_records(
        [
            LinkedInRejectedRecord(
                title="Data Scientist (Remote)",
                reason="location_scope_mismatch",
            ),
            LinkedInRejectedRecord(
                title="Data Scientist   (Remote)",
                reason="stale_detail_panel",
            ),
            LinkedInRejectedRecord(
                title="ML Engineer",
                reason="low_topic_relevance",
            ),
        ]
    )

    assert [item.title for item in deduped] == [
        "Data Scientist (Remote)",
        "ML Engineer",
    ]
    assert warnings == [
        "rejected_duplicate_dropped:Data Scientist   (Remote):stale_detail_panel"
    ]


def test_linkedin_country_scope_filter_handles_regional_remote_spillover():
    from features.web_scraping.infrastructure.linkedin_jobs_pipeline import (
        _visible_location_matches_requested_scope,
    )

    assert not _visible_location_matches_requested_scope(
        "Asia-Pacífico",
        "South Korea",
        "Build AI systems remotely for customers across APAC.",
    )
    assert _visible_location_matches_requested_scope(
        "Asia-Pacífico",
        "South Korea",
        "Principal Forward Deployed AI Engineer, South Korea. Based in Seoul.",
    )
    assert not _visible_location_matches_requested_scope(
        "Asia-Pacífico",
        "South Korea",
        "https://www.linkedin.com/jobs/search?keywords=Machine+Learning&location=South+Korea",
    )
    assert _visible_location_matches_requested_scope(
        "Asia-Pacífico",
        "South Korea",
        "Remote role for the Korea market, working with the Seoul office.",
    )
    assert not _visible_location_matches_requested_scope(
        "Asia-Pacífico",
        "Japan",
        "Build AI systems remotely for customers across APAC.",
    )
    assert _visible_location_matches_requested_scope(
        "Asia-Pacífico",
        "Japan",
        "Generative AI role supporting Tokyo and Japan customers.",
    )



def test_linkedin_country_signal_text_requires_role_evidence_for_remote_apac():
    from types import SimpleNamespace

    from features.web_scraping.infrastructure.linkedin_jobs_pipeline import (
        _record_country_signal_text,
        _visible_location_matches_requested_scope,
    )

    record = SimpleNamespace(
        title="Machine Learning Engineer - AI (Remote)",
        company_name="",
        description_excerpt="Build remote AI products for APAC customers.",
        description_full_text="",
        canonical_url="https://www.linkedin.com/jobs/view/4453510505",
        source_url="https://www.linkedin.com/jobs/search?keywords=Machine+Learning&location=South+Korea",
    )

    signal_text = _record_country_signal_text(record)

    assert not _visible_location_matches_requested_scope(
        "Asia-Pacífico",
        "South Korea",
        signal_text,
    )


def test_linkedin_country_signal_text_accepts_remote_apac_with_korea_role_evidence():
    from types import SimpleNamespace

    from features.web_scraping.infrastructure.linkedin_jobs_pipeline import (
        _record_country_signal_text,
        _visible_location_matches_requested_scope,
    )

    record = SimpleNamespace(
        title="Machine Learning Engineer - AI (Remote)",
        company_name="",
        description_excerpt="Remote role supporting the Korea market from the Seoul office.",
        description_full_text="",
        canonical_url="https://www.linkedin.com/jobs/view/4453510505",
        source_url="https://www.linkedin.com/jobs/search?keywords=Machine+Learning&location=South+Korea",
    )

    signal_text = _record_country_signal_text(record)

    assert _visible_location_matches_requested_scope(
        "Asia-Pacífico",
        "South Korea",
        signal_text,
    )

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

    monkeypatch.setenv("LINKEDIN_DETAIL_BUDGET", "6")
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
        "outside_24_hours",
        "unverified_posted_date",
    ]
    assert "duplicate_candidate_skipped_before_detail:111" in warnings
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


def test_linkedin_static_detail_probe_accepts_complete_jobposting_jsonld():
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _probe_linkedin_detail_with_authenticated_request,
    )

    class FakeResponse:
        status = 200
        url = "https://www.linkedin.com/jobs/view/111"

        def __init__(self):
            self.disposed = False

        def text(self):
            return """
            <script type="application/ld+json">
            {
              "@context": "https://schema.org",
              "@type": "JobPosting",
              "title": "AI Engineer",
              "datePosted": "2026-07-29T10:00:00Z",
              "description": "<p>Build production AI systems with Python and LLMs.</p>",
              "hiringOrganization": {"name": "Example AI"},
              "jobLocation": {
                "address": {
                  "addressLocality": "Seoul",
                  "addressCountry": "South Korea"
                }
              },
              "jobLocationType": "HYBRID"
            }
            </script>
            """

        def dispose(self):
            self.disposed = True

    response = FakeResponse()
    session = SimpleNamespace(
        context=SimpleNamespace(
            request=SimpleNamespace(get=lambda *_args, **_kwargs: response)
        )
    )
    record = LinkedInVacancyRecord(
        linkedin_job_id="111",
        title="AI Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/111",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        matched_terms=["ai"],
    )

    result = _probe_linkedin_detail_with_authenticated_request(
        session,
        record,
        include_description=True,
        now=datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc),
    )

    assert result.category == "ok"
    assert result.detail == ""
    assert result.record.company_name == "Example AI"
    assert result.record.location == "Seoul, South Korea"
    assert result.record.description_full_text == (
        "Build production AI systems with Python and LLMs."
    )
    assert result.record.published_at == datetime(
        2026,
        7,
        29,
        10,
        0,
        tzinfo=timezone.utc,
    )
    assert result.record.is_within_24_hours is True
    assert response.disposed is True


def test_linkedin_static_detail_probe_prefers_jsonld_description_without_guest():
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _probe_linkedin_detail_with_authenticated_request,
    )

    class FakeResponse:
        status = 200
        url = "https://www.linkedin.com/jobs/view/111"

        def text(self):
            payload = json.dumps(
                {
                    "@context": "https://schema.org",
                    "@type": "JobPosting",
                    "title": "AI Engineer",
                    "datePosted": "2026-07-29T10:00:00Z",
                    "description": (
                        "<p>Structured JSON-LD body for production AI work.</p>"
                    ),
                    "hiringOrganization": {"name": "Example AI"},
                    "jobLocation": {
                        "address": {
                            "addressLocality": "Seoul",
                            "addressCountry": "South Korea",
                        }
                    },
                }
            )
            return (
                f'<script type="application/ld+json">{payload}</script>'
                '<div class="show-more-less-html__markup">'
                "Container body should not win."
                "</div>"
            )

        def dispose(self):
            return None

    calls: list[str] = []

    def fake_get(url, **_kwargs):
        calls.append(url)
        return FakeResponse()

    session = SimpleNamespace(
        context=SimpleNamespace(request=SimpleNamespace(get=fake_get))
    )
    record = LinkedInVacancyRecord(
        linkedin_job_id="111",
        title="AI Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/111",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        matched_terms=["ai"],
    )

    result = _probe_linkedin_detail_with_authenticated_request(
        session,
        record,
        include_description=True,
        now=datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc),
    )

    assert result.category == "ok"
    assert result.body_source == "jsonld_description"
    assert result.record.description_full_text == (
        "Structured JSON-LD body for production AI work."
    )
    assert calls == ["https://www.linkedin.com/jobs/view/111"]


def test_linkedin_static_detail_probe_recovers_container_without_jsonld():
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _probe_linkedin_detail_with_authenticated_request,
    )

    class FakeResponse:
        status = 200
        url = "https://www.linkedin.com/jobs/view/111"

        def text(self):
            return """
            <main>
              <h1 class="top-card-layout__title">AI Engineer</h1>
              <a class="topcard__org-name-link">Example AI</a>
              <div class="show-more-less-html__markup">
                <p>Build production AI systems.</p>
                <ul><li>Own LLM services.</li></ul>
              </div>
            </main>
            """

        def dispose(self):
            return None

    session = SimpleNamespace(
        context=SimpleNamespace(
            request=SimpleNamespace(get=lambda *_args, **_kwargs: FakeResponse())
        )
    )
    record = LinkedInVacancyRecord(
        linkedin_job_id="111",
        title="AI Engineer",
        company_name="Example AI",
        location="Seoul, South Korea",
        posted_at_text="2026-07-29T10:00:00Z",
        published_at=datetime(2026, 7, 29, 10, 0, tzinfo=timezone.utc),
        freshness_confidence="high",
        is_within_24_hours=True,
        canonical_url="https://www.linkedin.com/jobs/view/111",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        matched_terms=["ai"],
    )

    result = _probe_linkedin_detail_with_authenticated_request(
        session,
        record,
        include_description=True,
        now=datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc),
    )

    assert result.category == "ok"
    assert result.body_source == "static_html_container"
    assert "Build production AI systems." in result.record.description_full_text
    assert "Own LLM services." in result.record.description_full_text


def test_linkedin_static_detail_probe_uses_guest_body_with_absent_company():
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _probe_linkedin_detail_with_authenticated_request,
    )

    class FakeResponse:
        def __init__(self, text: str, url: str):
            self.status = 200
            self.url = url
            self._text = text

        def text(self):
            return self._text

        def dispose(self):
            return None

    same_url_payload = json.dumps(
        {
            "@context": "https://schema.org",
            "@type": "JobPosting",
            "title": "AI Engineer",
            "datePosted": "2026-07-29T10:00:00Z",
            "description": "",
            "hiringOrganization": {"name": "Example AI"},
            "jobLocation": {
                "address": {
                    "addressLocality": "Seoul",
                    "addressCountry": "South Korea",
                }
            },
        }
    )
    guest_html = """
    <main>
      <h1 class="top-card-layout__title">AI Engineer</h1>
      <div class="show-more-less-html__markup">
        <p>Recovered guest body for AI platform delivery.</p>
      </div>
    </main>
    """
    calls: list[str] = []

    def fake_get(url, **_kwargs):
        calls.append(url)
        if "jobs-guest" in url:
            return FakeResponse(guest_html, url)
        return FakeResponse(
            f'<script type="application/ld+json">{same_url_payload}</script>',
            "https://www.linkedin.com/jobs/view/111",
        )

    session = SimpleNamespace(
        context=SimpleNamespace(request=SimpleNamespace(get=fake_get)),
        page=SimpleNamespace(wait_for_timeout=lambda *_args: None),
    )
    record = LinkedInVacancyRecord(
        linkedin_job_id="111",
        title="AI Engineer",
        company_name="Example AI",
        canonical_url="https://www.linkedin.com/jobs/view/111",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        matched_terms=["ai"],
    )

    result = _probe_linkedin_detail_with_authenticated_request(
        session,
        record,
        include_description=True,
        now=datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc),
    )

    assert result.category == "ok"
    assert result.body_source == "guest_html_container"
    assert result.guest_status_code == 200
    assert result.identity_consistent is True
    assert result.record.description_full_text == (
        "Recovered guest body for AI platform delivery."
    )
    assert calls == [
        "https://www.linkedin.com/jobs/view/111",
        "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/111",
    ]


def test_linkedin_static_detail_probe_rejects_guest_identity_contradiction():
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _probe_linkedin_detail_with_authenticated_request,
    )

    class FakeResponse:
        def __init__(self, text: str, url: str):
            self.status = 200
            self.url = url
            self._text = text

        def text(self):
            return self._text

        def dispose(self):
            return None

    same_url_payload = json.dumps(
        {
            "@context": "https://schema.org",
            "@type": "JobPosting",
            "title": "AI Engineer",
            "datePosted": "2026-07-29T10:00:00Z",
            "description": "",
            "hiringOrganization": {"name": "Example AI"},
        }
    )
    guest_html = """
    <main>
      <h1 class="top-card-layout__title">Backend Engineer</h1>
      <a class="topcard__org-name-link">Wrong Corp</a>
      <div class="show-more-less-html__markup">
        <p>Wrong job body that must not be attached.</p>
      </div>
    </main>
    """

    def fake_get(url, **_kwargs):
        if "jobs-guest" in url:
            return FakeResponse(guest_html, url)
        return FakeResponse(
            f'<script type="application/ld+json">{same_url_payload}</script>',
            "https://www.linkedin.com/jobs/view/111",
        )

    session = SimpleNamespace(
        context=SimpleNamespace(request=SimpleNamespace(get=fake_get)),
        page=SimpleNamespace(wait_for_timeout=lambda *_args: None),
    )
    record = LinkedInVacancyRecord(
        linkedin_job_id="111",
        title="AI Engineer",
        company_name="Example AI",
        canonical_url="https://www.linkedin.com/jobs/view/111",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        matched_terms=["ai"],
    )

    result = _probe_linkedin_detail_with_authenticated_request(
        session,
        record,
        include_description=True,
        now=datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc),
    )

    assert result.category == "detail_incomplete"
    assert result.detail == "missing_description"
    assert result.identity_consistent is False
    assert result.record.description_full_text == ""


@pytest.mark.parametrize(
    ("description", "date_posted", "expected_detail"),
    [
        ("", "2026-07-29T10:00:00Z", "missing_description"),
        ("About the job", "2026-07-29T10:00:00Z", "incomplete_description"),
        ("Build AI systems.", "", "missing_date"),
        ("Build AI systems.", "2026-07-27T10:00:00Z", "outside_24_hours"),
    ],
)
def test_linkedin_static_detail_probe_rejects_incomplete_jobposting_jsonld(
    description,
    date_posted,
    expected_detail,
):
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_scraper import (
        _probe_linkedin_detail_with_authenticated_request,
    )

    class FakeResponse:
        status = 200
        url = "https://www.linkedin.com/jobs/view/111"

        def __init__(self):
            self.disposed = False

        def text(self):
            payload = json.dumps(
                {
                    "@context": "https://schema.org",
                    "@type": "JobPosting",
                    "title": "AI Engineer",
                    "datePosted": date_posted,
                    "description": description,
                    "hiringOrganization": {"name": "Example AI"},
                    "jobLocation": {
                        "address": {
                            "addressLocality": "Seoul",
                            "addressCountry": "South Korea",
                        }
                    },
                }
            )
            return f'<script type="application/ld+json">{payload}</script>'

        def dispose(self):
            self.disposed = True

    response = FakeResponse()
    session = SimpleNamespace(
        context=SimpleNamespace(
            request=SimpleNamespace(get=lambda *_args, **_kwargs: response)
        )
    )
    record = LinkedInVacancyRecord(
        linkedin_job_id="111",
        title="AI Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/111",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        matched_terms=["ai"],
    )

    result = _probe_linkedin_detail_with_authenticated_request(
        session,
        record,
        include_description=True,
        now=datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc),
    )

    assert result.category == "detail_incomplete"
    assert result.detail == expected_detail
    assert response.disposed is True


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

    monkeypatch.setenv("LINKEDIN_MAX_QUERIES_PER_LOCATION", "4")
    with pytest.raises(ValueError, match="entre 1 y 3"):
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
            LinkedInSearchHydrationDiagnostic,
            LinkedInStaticProbeDiagnostic,
            LinkedInVacancyRecord,
            LinkedInVisualDiagnosticArtifact,
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
        search_hydration_diagnostics=[
            LinkedInSearchHydrationDiagnostic(
                query="AI Engineer @ Japan",
                sequence=1,
                elapsed_ms=0,
                card_count=0,
                href_count=0,
                empty_state_visible=False,
                auth_checkpoint_visible=False,
                outcome="polling",
            ),
            LinkedInSearchHydrationDiagnostic(
                query="AI Engineer @ Japan",
                sequence=2,
                elapsed_ms=250,
                card_count=3,
                href_count=2,
                empty_state_visible=False,
                auth_checkpoint_visible=False,
                outcome="results",
            ),
        ],
        static_probe_diagnostics=[
            LinkedInStaticProbeDiagnostic(
                kind="search_static_probe",
                query="AI Engineer @ Japan",
                sequence=1,
                status_code=200,
                candidate_count=3,
                accepted_count=2,
                outcome="ok",
            )
        ],
        visual_diagnostics=[
            LinkedInVisualDiagnosticArtifact(
                query="AI Engineer @ Japan",
                manifest_path="visual-diagnostics/manifest.json",
                sensitive_local_artifact=True,
            )
        ],
    )
    snapshot = load_linkedin_audit_snapshot(paths.json_path)

    assert snapshot.meta.session_id == "sess-1"
    assert snapshot.meta.schema_version == "1.7.0"
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
    assert [event.outcome for event in snapshot.search_hydration_diagnostics] == [
        "polling",
        "results",
    ]
    assert snapshot.search_hydration_diagnostics[1].card_count == 3
    assert snapshot.search_hydration_diagnostics[1].href_count == 2
    assert snapshot.static_probe_diagnostics[0].kind == "search_static_probe"
    assert snapshot.static_probe_diagnostics[0].candidate_count == 3
    assert snapshot.static_probe_diagnostics[0].accepted_count == 2
    assert snapshot.visual_diagnostics[0].manifest_path == (
        "visual-diagnostics/manifest.json"
    )
    assert snapshot.visual_diagnostics[0].sensitive_local_artifact is True
    assert paths.schema_path.exists()
    serialized = paths.json_path.read_text(encoding="utf-8")
    assert "storage_state" not in serialized
    assert "cookies" not in serialized
    assert "<html" not in serialized.lower()
    summary = paths.summary_path.read_text(encoding="utf-8")
    assert "Search hydration diagnostics" in summary
    assert "Static probe diagnostics" in summary
    assert "Visual diagnostics" in summary
    assert "visual-diagnostics/manifest.json" in summary
    assert "**search_static_probe** `AI Engineer @ Japan` #1: ok" in summary
    assert "**AI Engineer @ Japan** #2: results at 250ms" in summary
    assert "cards=3" in summary
    assert "hrefs=2" in summary
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


def test_linkedin_service_persists_hydration_diagnostics_on_blocked_exception(
    monkeypatch,
    tmp_path,
):
    from features.web_scraping.application import linkedin_service
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        get_active_search_hydration_diagnostics,
    )

    captured: dict[str, object] = {}

    def fake_scrape(_request):
        collector = get_active_search_hydration_diagnostics()
        assert collector is not None
        collector.record(
            query="AI/ML/Data/GenAI @ South Korea",
            elapsed_ms=100,
            card_count=0,
            href_count=1,
            empty_state_visible=False,
            auth_checkpoint_visible=True,
            outcome="auth_checkpoint",
        )
        raise linkedin_service.LinkedInBlockedError("checkpoint")

    def fake_persist(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            job_uid="job-1",
            json_path=tmp_path / "audit.json",
            schema_path=tmp_path / "schema.json",
            summary_path=tmp_path / "summary.md",
        )

    monkeypatch.delenv("LINKEDIN_AUTO_BOOTSTRAP_ON_AUTH_FAILURE", raising=False)
    monkeypatch.setattr(linkedin_service, "scrape_linkedin_jobs", fake_scrape)
    monkeypatch.setattr(
        linkedin_service,
        "persist_linkedin_audit_snapshot",
        fake_persist,
    )

    result = linkedin_service.run_linkedin_jobs_vertical(
        "Buscá vacantes LinkedIn de AI publicadas hoy solo en Corea del Sur"
    )

    assert result.status == "blocked"
    assert result.records == []
    assert any(warning.startswith("blocked:") for warning in result.warnings)
    events = captured["search_hydration_diagnostics"]
    assert len(events) == 1
    assert events[0].query == "AI/ML/Data/GenAI @ South Korea"
    assert events[0].outcome == "auth_checkpoint"
    assert events[0].href_count == 1
    assert events[0].auth_checkpoint_visible is True


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


def _run_linkedin_static_detail_validation_case(
    monkeypatch,
    *,
    static_record_update: dict,
    static_detail: str,
    static_body_source: str = "guest_html_container",
    direct_detail_fallback: bool = False,
    direct_record_update: dict | None = None,
):
    from features.web_scraping.domain.linkedin_models import (
        LinkedInJobsRequest,
        LinkedInParseDiagnostics,
        LinkedInVacancyRecord,
    )
    from features.web_scraping.infrastructure.linkedin_detail_diagnostics import (
        detail_diagnostics_context,
    )
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        search_hydration_diagnostics_context,
    )
    from features.web_scraping.infrastructure.linkedin_static_probe_diagnostics import (
        static_probe_diagnostics_context,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import scrape_linkedin_jobs

    monkeypatch.setenv("LINKEDIN_DETAIL_BUDGET", "3")
    monkeypatch.setenv("LINKEDIN_MAX_QUERIES_PER_LOCATION", "1")
    monkeypatch.setenv(
        "LINKEDIN_DIRECT_DETAIL_FALLBACK",
        "true" if direct_detail_fallback else "false",
    )
    source_url = "https://www.linkedin.com/jobs/search/?keywords=AI"
    candidate = LinkedInVacancyRecord(
        linkedin_job_id="4452375291",
        title="AI Engineer",
        company_name="Example AI",
        location="Seoul, South Korea",
        workplace_type="hybrid",
        canonical_url="https://www.linkedin.com/jobs/view/4452375291",
        source_url=source_url,
        matched_terms=["ai"],
    )
    static_record = candidate.model_copy(update=static_record_update)
    direct_record = candidate.model_copy(update=direct_record_update or {})

    class FakePage:
        url = "https://www.linkedin.com/jobs/"

        def goto(self, url: str, **_kwargs) -> None:
            self.url = url

        def wait_for_timeout(self, _milliseconds: int) -> None:
            return None

        def content(self) -> str:
            return "<main></main>"

        def is_closed(self) -> bool:
            return False

    class FakeSession:
        page = FakePage()
        context = SimpleNamespace(
            request=SimpleNamespace(get=lambda *_args, **_kwargs: None)
        )

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

    def wait_hydration(_page, *, query="unknown", diagnostics=None):
        assert diagnostics is not None
        outcome = "results" if query.startswith("restore:") else "timeout"
        diagnostics.record(
            query=query,
            elapsed_ms=250,
            card_count=0 if outcome == "timeout" else 1,
            href_count=0 if outcome == "timeout" else 1,
            empty_state_visible=False,
            auth_checkpoint_visible=False,
            outcome=outcome,
        )
        return outcome

    search_probe = SimpleNamespace(
        records=[candidate],
        diagnostics=LinkedInParseDiagnostics(
            href_count=1,
            candidate_count=1,
            parseable_candidate_count=1,
            discard_reasons={"standalone_link_fallback": 1},
        ),
        status_code=200,
        category="ok",
        detail="",
    )
    detail_probe = SimpleNamespace(
        record=static_record,
        status_code=200,
        category="ok" if static_detail == "" else "detail_incomplete",
        detail=static_detail,
        body_source=static_body_source,
        description_length=len(static_record.description_full_text or ""),
        guest_status_code=200 if static_body_source.startswith("guest_") else 0,
        guest_retry_count=1 if static_body_source.startswith("guest_") else 0,
        identity_consistent=True,
    )

    with (
        search_hydration_diagnostics_context(),
        static_probe_diagnostics_context() as static_diagnostics,
        detail_diagnostics_context() as detail_diagnostics,
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.open_persistent_authenticated_context",
            return_value=FakeSession(),
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._wait_for_search_results_hydration",
            side_effect=wait_hydration,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper.build_linkedin_search_queries",
            return_value=[("Primary @ South Korea", source_url)],
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._parse_linkedin_jobs_html_with_diagnostics",
            return_value=([], LinkedInParseDiagnostics()),
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._probe_linkedin_search_with_authenticated_request",
            return_value=search_probe,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._safe_job_card_link",
            return_value=None,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._probe_linkedin_detail_with_authenticated_request",
            return_value=detail_probe,
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._enrich_job_detail",
            return_value=direct_record,
        ) as direct_detail,
    ):
        records, rejected, _timings, warnings, _queries = scrape_linkedin_jobs(
            LinkedInJobsRequest(
                query="AI jobs South Korea",
                locations=["South Korea"],
                max_results=5,
                include_description=True,
            ),
            session_store=FakeStore(),  # type: ignore[arg-type]
        )

    return SimpleNamespace(
        records=records,
        rejected=rejected,
        warnings=warnings,
        direct_detail=direct_detail,
        static_diagnostics=static_diagnostics,
        detail_diagnostics=detail_diagnostics,
    )


def test_linkedin_static_guest_body_missing_date_rejects_unverified_date(monkeypatch):
    result = _run_linkedin_static_detail_validation_case(
        monkeypatch,
        static_record_update={
            "description_full_text": "Recovered guest body for AI platform delivery.",
            "description_excerpt": "Recovered guest body for AI platform delivery.",
        },
        static_detail="missing_date",
    )

    assert result.records == []
    assert [item.reason for item in result.rejected] == ["unverified_posted_date"]
    assert "missing_description_full_text" not in {
        item.reason for item in result.rejected
    }


def test_linkedin_static_guest_body_survives_later_direct_incomplete(monkeypatch):
    result = _run_linkedin_static_detail_validation_case(
        monkeypatch,
        static_record_update={
            "description_full_text": "Recovered guest body for AI platform delivery.",
            "description_excerpt": "Recovered guest body for AI platform delivery.",
        },
        static_detail="missing_date",
        direct_detail_fallback=True,
        direct_record_update={"description_full_text": "", "description_excerpt": ""},
    )

    assert result.direct_detail.call_count == 1
    assert [item.reason for item in result.rejected] == ["unverified_posted_date"]
    validation_events = [
        event
        for event in result.detail_diagnostics.events
        if event.phase == "validation"
    ]
    assert validation_events[-1].description_ready is True
    assert validation_events[-1].rejection == "missing_date"


def test_linkedin_static_guest_body_verified_recent_korea_is_accepted(monkeypatch):
    result = _run_linkedin_static_detail_validation_case(
        monkeypatch,
        static_record_update={
            "description_full_text": "Build production AI systems for Korean customers.",
            "description_excerpt": "Build production AI systems for Korean customers.",
            "posted_at_text": "2026-07-29T10:00:00Z",
            "published_at": datetime(2026, 7, 29, 10, 0, tzinfo=timezone.utc),
            "freshness_confidence": "high",
            "is_within_24_hours": True,
            "hard_skills": ["Python"],
            "foreigner_acceptance": "yes",
        },
        static_detail="",
    )

    assert [record.linkedin_job_id for record in result.records] == ["4452375291"]
    assert result.rejected == []


def test_linkedin_static_detail_without_plausible_body_rejects_missing_description(
    monkeypatch,
):
    result = _run_linkedin_static_detail_validation_case(
        monkeypatch,
        static_record_update={
            "description_full_text": "",
            "posted_at_text": "2026-07-29T10:00:00Z",
            "published_at": datetime(2026, 7, 29, 10, 0, tzinfo=timezone.utc),
            "freshness_confidence": "high",
            "is_within_24_hours": True,
        },
        static_detail="missing_description",
        static_body_source="",
    )

    assert result.records == []
    assert [item.reason for item in result.rejected] == [
        "missing_description_full_text"
    ]


def test_linkedin_static_guest_body_old_date_rejects_outside_24_hours(monkeypatch):
    result = _run_linkedin_static_detail_validation_case(
        monkeypatch,
        static_record_update={
            "description_full_text": "Build production AI systems for Korean customers.",
            "description_excerpt": "Build production AI systems for Korean customers.",
            "posted_at_text": "2026-07-27T10:00:00Z",
            "published_at": datetime(2026, 7, 27, 10, 0, tzinfo=timezone.utc),
            "freshness_confidence": "high",
            "is_within_24_hours": False,
        },
        static_detail="outside_24_hours",
    )

    assert result.records == []
    assert [item.reason for item in result.rejected] == ["outside_24_hours"]


def test_linkedin_audit_regression_guest_body_missing_date_final_rejection(
    monkeypatch,
):
    result = _run_linkedin_static_detail_validation_case(
        monkeypatch,
        static_record_update={
            "description_full_text": "Recovered guest body for AI platform delivery.",
            "description_excerpt": "Recovered guest body for AI platform delivery.",
        },
        static_detail="missing_date",
        static_body_source="guest_html_container",
    )

    detail_static_events = [
        event
        for event in result.static_diagnostics.events
        if event.kind == "detail_static_probe"
    ]
    assert detail_static_events[-1].outcome == "missing_date"
    assert detail_static_events[-1].body_source == "guest_html_container"
    assert detail_static_events[-1].description_length > 0
    assert [item.reason for item in result.rejected] == ["unverified_posted_date"]
    assert [
        event.rejection
        for event in result.detail_diagnostics.events
        if event.phase == "validation"
    ] == ["missing_date"]



def test_linkedin_parser_accepts_provisional_job_id_without_title():
    from features.web_scraping.infrastructure.linkedin_parser import (
        _parse_linkedin_jobs_html_with_diagnostics,
    )

    html = """
    <ul class="jobs-search-results-list">
      <li role="listitem" data-job-id="123">
        <a href="/jobs/view/123" aria-label=""></a>
      </li>
    </ul>
    """

    records, diagnostics = _parse_linkedin_jobs_html_with_diagnostics(
        html,
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
    )

    assert [record.linkedin_job_id for record in records] == ["123"]
    assert records[0].title == ""
    assert records[0].candidate_metadata_incomplete is True
    assert diagnostics.unique_candidate_count == 1
    assert diagnostics.raw_signal_count >= 1


def test_linkedin_parser_merges_card_href_urn_sources_for_one_job():
    from features.web_scraping.infrastructure.linkedin_parser import (
        _parse_linkedin_jobs_html_with_diagnostics,
    )

    html = """
    <div class="job-card-container" data-job-id="123" data-entity-urn="urn:li:jobPosting:123">
      <a class="job-card-list__title--link" href="/jobs/view/ai-engineer-123">
        <span class="job-card-list__title">AI Engineer</span>
      </a>
      <span class="job-card-container__primary-description">Example AI</span>
      <span class="job-card-container__metadata-item">Tokyo, Japan</span>
    </div>
    """

    records, diagnostics = _parse_linkedin_jobs_html_with_diagnostics(
        html,
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
    )

    assert [record.linkedin_job_id for record in records] == ["123"]
    assert records[0].discovery_sources == ["card", "job_href", "urn"]
    assert diagnostics.raw_signal_count >= 3
    assert diagnostics.unique_candidate_count == 1
    assert diagnostics.duplicate_candidate_count == 0


def test_linkedin_parser_recovers_multiple_job_hrefs_without_cards():
    from features.web_scraping.infrastructure.linkedin_parser import (
        _parse_linkedin_jobs_html_with_diagnostics,
    )

    links = "".join(
        f'<li role="listitem"><a href="/jobs/view/{job_id}">AI Engineer {job_id}</a></li>'
        for job_id in range(100, 106)
    )
    html = f'<ul class="jobs-search-results-list">{links}</ul>'

    records, diagnostics = _parse_linkedin_jobs_html_with_diagnostics(
        html,
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
    )

    assert [record.linkedin_job_id for record in records] == [str(job_id) for job_id in range(100, 106)]
    assert diagnostics.selector_counts[".job-card-container"] == 0
    assert diagnostics.job_href_signal_count == 6
    assert diagnostics.unique_candidate_count == 6


def test_linkedin_pipeline_dedupes_same_run_before_detail_and_marks_retained(monkeypatch):
    from features.web_scraping.domain.linkedin_models import (
        LinkedInJobsRequest,
        LinkedInParseDiagnostics,
        LinkedInVacancyRecord,
    )
    from features.web_scraping.infrastructure.linkedin_scraper import scrape_linkedin_jobs

    monkeypatch.setenv("LINKEDIN_DETAIL_BUDGET", "2")
    monkeypatch.setenv("LINKEDIN_DIRECT_DETAIL_FALLBACK", "true")
    monkeypatch.setenv("LINKEDIN_MAX_QUERIES_PER_LOCATION", "2")
    source_a = "https://www.linkedin.com/jobs/search/?keywords=AI"
    source_b = "https://www.linkedin.com/jobs/search/?keywords=ML"
    candidate_a = LinkedInVacancyRecord(
        linkedin_job_id="123",
        title="",
        canonical_url="https://www.linkedin.com/jobs/view/123",
        source_url=source_a,
        matched_terms=[],
        candidate_metadata_incomplete=True,
        discovery_sources=["job_href"],
    )
    candidate_b = candidate_a.model_copy(
        update={"source_url": source_b, "discovery_sources": ["card", "urn"]}
    )
    enriched = candidate_a.model_copy(
        update={
            "title": "AI Engineer",
            "company_name": "Example AI",
            "location": "Tokyo, Japan",
            "workplace_type": "hybrid",
            "posted_at_text": "1 hour ago",
            "published_at": datetime(2026, 7, 29, 10, 0, tzinfo=timezone.utc),
            "freshness_confidence": "medium",
            "is_within_24_hours": True,
            "candidate_metadata_incomplete": False,
        }
    )

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
            return {"browser": "chromium", "executable_path": "", "profile_path": "/private/browser-profile"}
        def resolve_profile_path(self, **_kwargs):
            return Path("/private/browser-profile")
        def record_runtime_failure(self, *_args, **_kwargs) -> None:
            raise AssertionError("unexpected runtime failure")

    with (
        patch("features.web_scraping.infrastructure.linkedin_scraper.open_persistent_authenticated_context", return_value=FakeSession()),
        patch("features.web_scraping.infrastructure.linkedin_scraper._validate_authenticated_page"),
        patch("features.web_scraping.infrastructure.linkedin_scraper._wait_for_search_results_hydration", side_effect=["results", "timeout"]),
        patch("features.web_scraping.infrastructure.linkedin_scraper.build_linkedin_search_queries", return_value=[("A @ Japan", source_a), ("B @ Japan", source_b)]),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._parse_linkedin_jobs_html_with_diagnostics",
            side_effect=[
                ([candidate_a], LinkedInParseDiagnostics(raw_signal_count=1, unique_candidate_count=1, candidate_count=1)),
                ([candidate_b], LinkedInParseDiagnostics(raw_signal_count=2, unique_candidate_count=1, candidate_count=1, discard_reasons={"standalone_link_fallback": 1})),
            ],
        ),
        patch(
            "features.web_scraping.infrastructure.linkedin_scraper._ensure_search_source_with_single_retry",
            side_effect=lambda _session, page, **_kwargs: (page, False, False),
        ),
        patch("features.web_scraping.infrastructure.linkedin_scraper._safe_job_card_link", return_value=MagicMock()),
        patch("features.web_scraping.infrastructure.linkedin_scraper._enrich_job_detail_via_panel", return_value=enriched) as detail,
    ):
        records, rejected, timings, warnings, _queries = scrape_linkedin_jobs(
            LinkedInJobsRequest(query="AI jobs Japan", locations=["Japan"], max_results=5, include_description=False),
            session_store=FakeStore(),  # type: ignore[arg-type]
        )

    assert [record.linkedin_job_id for record in records] == ["123"]
    assert records[0].title == "AI Engineer"
    assert records[0].candidate_metadata_incomplete is False
    assert detail.call_count == 1
    assert [timing.retained_count for timing in timings] == [1, 0]
    assert timings[0].diagnostics.new_candidate_count == 1
    assert timings[0].diagnostics.duplicate_candidate_count == 0
    assert timings[1].diagnostics.new_candidate_count == 0
    assert timings[1].diagnostics.duplicate_candidate_count == 1
    assert rejected == []
    assert "duplicate_candidate_skipped_before_detail:123" in warnings




def test_linkedin_row_activation_outcomes_map_to_valid_search_hydration_outcomes():
    from features.web_scraping.domain.linkedin_models import LinkedInSearchHydrationDiagnostic
    from features.web_scraping.infrastructure.linkedin_jobs_pipeline import (
        _search_hydration_outcome_for_row_activation,
    )

    success = _search_hydration_outcome_for_row_activation("row_activation_success")
    failure = _search_hydration_outcome_for_row_activation("row_activation_no_job_id")

    assert success == "results"
    assert failure == "failed"
    LinkedInSearchHydrationDiagnostic(query="AI", sequence=1, elapsed_ms=0, outcome=success)
    LinkedInSearchHydrationDiagnostic(query="AI", sequence=2, elapsed_ms=0, outcome=failure)

def test_linkedin_row_identity_change_is_detected_without_url_change():
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        LinkedInDetailIdentity,
        _detail_identity_changed,
    )

    before = LinkedInDetailIdentity(
        job_id="111",
        canonical_detail_href="https://www.linkedin.com/jobs/view/111",
        allowlisted_attributes=(("data-job-id", "111"),),
    )
    after = LinkedInDetailIdentity(
        job_id="222",
        canonical_detail_href="https://www.linkedin.com/jobs/view/111",
        allowlisted_attributes=(("data-job-id", "222"),),
    )

    assert _detail_identity_changed(before, after) is True


def test_linkedin_detail_identity_falls_back_to_visible_right_panel_job_link():
    import features.web_scraping.infrastructure.linkedin_query_navigation as navigation

    class LinkCollection:
        def __init__(self, links):
            self.links = links

        def count(self):
            return len(self.links)

        def nth(self, index):
            return self.links[index]

    class EmptyCollection:
        def count(self):
            return 0

        def nth(self, _index):
            raise IndexError

    class Link:
        def __init__(self, href, box):
            self.href = href
            self.box = box

        def get_attribute(self, name):
            return self.href if name == "href" else None

        def bounding_box(self):
            return self.box

    class Page:
        url = "https://www.linkedin.com/jobs/search/?keywords=AI"

        def evaluate(self, _script):
            return 1366

        def locator(self, selector):
            if selector == "a[href*='/jobs/view/']":
                return LinkCollection([
                    Link("/jobs/view/111", {"x": 190, "y": 220, "width": 280, "height": 24}),
                    Link("/jobs/view/222", {"x": 594, "y": 267, "width": 613, "height": 30}),
                    Link("/jobs/view/333/apply/", {"x": 594, "y": 426, "width": 245, "height": 32}),
                ])
            return EmptyCollection()

    identity = navigation._detail_identity_from_page(Page())

    assert identity.job_id == "222"
    assert identity.canonical_detail_href == "https://www.linkedin.com/jobs/view/222"




def test_linkedin_duplicate_candidate_merges_date_evidence_before_detail():
    from datetime import datetime, timezone

    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_jobs_pipeline import (
        _merge_candidate_discovery_evidence,
    )

    current = LinkedInVacancyRecord(
        linkedin_job_id="4453249078",
        title="Machine Learning Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/4453249078",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        discovery_sources=["job_href"],
        candidate_metadata_incomplete=True,
    )
    published = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
    duplicate = current.model_copy(
        update={
            "company_name": "MUSINSA 무신사",
            "location": "Seúl, Seúl, Corea del Sur",
            "posted_at_text": "hace 11 horas",
            "published_at": published,
            "freshness_confidence": "medium",
            "is_within_24_hours": True,
            "discovery_sources": ["row_activation"],
        }
    )

    merged = _merge_candidate_discovery_evidence(current, duplicate)

    assert merged.linkedin_job_id == "4453249078"
    assert merged.title == "Machine Learning Engineer"
    assert merged.company_name == "MUSINSA 무신사"
    assert merged.location == "Seúl, Seúl, Corea del Sur"
    assert merged.posted_at_text == "hace 11 horas"
    assert merged.published_at == published
    assert merged.freshness_confidence == "medium"
    assert merged.is_within_24_hours is True
    assert merged.discovery_sources == ["job_href", "row_activation"]
    assert merged.candidate_metadata_incomplete is False


def test_linkedin_row_activation_date_warnings_are_safe_per_job():
    from features.web_scraping.infrastructure.linkedin_jobs_pipeline import (
        _row_activation_date_warning,
    )
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        LinkedInRowActivationOutcome,
    )

    assert _row_activation_date_warning(
        LinkedInRowActivationOutcome(
            "row_activation_success",
            (),
            job_id="4450481657",
            date_detected=True,
            date_verified=True,
            date_within_24_hours=True,
        )
    ) == "row_date_verified:4450481657:within_24_hours"
    assert _row_activation_date_warning(
        LinkedInRowActivationOutcome(
            "row_activation_duplicate",
            (),
            job_id="4450481657",
            date_detected=True,
            date_verified=True,
            date_within_24_hours=False,
        )
    ) == "row_date_verified:4450481657:outside_24_hours"
    assert _row_activation_date_warning(
        LinkedInRowActivationOutcome(
            "row_activation_success",
            (),
            job_id="4450481657",
            date_detected=False,
            date_verified=False,
        )
    ) == "row_date_missing:4450481657"
    assert _row_activation_date_warning(
        LinkedInRowActivationOutcome(
            "row_activation_no_job_id",
            (),
            date_detected=True,
            date_verified=False,
        )
    ) == "row_date_unattributed"

def test_linkedin_visible_card_date_collector_parses_left_card_dates(monkeypatch):
    from datetime import datetime, timezone

    from features.web_scraping.infrastructure import linkedin_query_navigation as navigation

    class Page:
        def evaluate(self, script):
            assert "data-occludable-job-id" in script
            assert "jobIdFromNode" in script
            assert "findDateWithin" in script
            return {"4426794001": "Visto · Hace 9 horas · Solicitud sencilla"}

    monkeypatch.setattr(
        navigation,
        "parse_linkedin_relative_time",
        lambda text: (datetime(2026, 8, 13, 9, 0, tzinfo=timezone.utc), "medium", True),
    )

    dates = navigation.collect_visible_search_card_dates(Page())

    assert dates == {
        "4426794001": (
            "Hace 9 horas",
            datetime(2026, 8, 13, 9, 0, tzinfo=timezone.utc),
            "medium",
            True,
        )
    }


def test_linkedin_visible_card_date_evidence_overrides_existing_outside_date():
    from datetime import datetime, timezone

    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_jobs_pipeline import (
        _apply_visible_card_date_evidence,
    )

    stale = datetime(2026, 8, 12, 9, 0, tzinfo=timezone.utc)
    visible = datetime(2026, 8, 13, 23, 0, tzinfo=timezone.utc)
    record = LinkedInVacancyRecord(
        linkedin_job_id="4426794001",
        title="Machine Learning Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/4426794001",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        posted_at_text="hace 2 días",
        published_at=stale,
        freshness_confidence="low",
        is_within_24_hours=False,
        discovery_sources=["job_href"],
    )

    updated, transition = _apply_visible_card_date_evidence(
        record,
        "4426794001",
        {
            "4426794001": (
                "Hace 10 horas",
                visible,
                "medium",
                True,
            )
        },
    )

    assert transition == "outside_24_hours_to_within_24_hours"
    assert updated.posted_at_text == "Hace 10 horas"
    assert updated.published_at == visible
    assert updated.freshness_confidence == "medium"
    assert updated.is_within_24_hours is True
    assert updated.discovery_sources == ["job_href", "visible_card"]


def test_linkedin_visible_card_date_evidence_never_degrades_existing_within_date():
    from datetime import datetime, timezone

    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_jobs_pipeline import (
        _apply_visible_card_date_evidence,
    )

    current = datetime(2026, 8, 13, 22, 0, tzinfo=timezone.utc)
    outside = datetime(2026, 8, 11, 22, 0, tzinfo=timezone.utc)
    record = LinkedInVacancyRecord(
        linkedin_job_id="4453503447",
        title="Artificial Intelligence Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/4453503447",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        posted_at_text="hace 41 minutos",
        published_at=current,
        freshness_confidence="medium",
        is_within_24_hours=True,
        discovery_sources=["job_href"],
    )

    updated, transition = _apply_visible_card_date_evidence(
        record,
        "4453503447",
        {
            "4453503447": (
                "Hace 2 días",
                outside,
                "medium",
                False,
            )
        },
    )

    assert transition == ""
    assert updated is record
    assert updated.posted_at_text == "hace 41 minutos"
    assert updated.published_at == current
    assert updated.is_within_24_hours is True
    assert updated.discovery_sources == ["job_href"]


def test_linkedin_detail_priority_spends_budget_on_verified_card_dates_first():
    from datetime import datetime, timezone

    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_jobs_pipeline import (
        _detail_priority_label,
        _prioritize_candidates_for_detail,
    )

    source = "https://www.linkedin.com/jobs/search/?keywords=AI"
    missing_date = LinkedInVacancyRecord(
        linkedin_job_id="111",
        title="Machine Learning Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/111",
        source_url=source,
        matched_terms=["machine learning"],
        discovery_sources=["job_href"],
    )
    verified_card = LinkedInVacancyRecord(
        linkedin_job_id="222",
        title="AI Product Team Leader",
        canonical_url="https://www.linkedin.com/jobs/view/222",
        source_url=source,
        matched_terms=["ai"],
        posted_at_text="Hace 10 horas",
        published_at=datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc),
        freshness_confidence="medium",
        is_within_24_hours=True,
        discovery_sources=["job_href", "visible_card"],
    )
    verified_other = LinkedInVacancyRecord(
        linkedin_job_id="333",
        title="Deep Learning Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/333",
        source_url=source,
        matched_terms=["deep learning"],
        posted_at_text="hace 15 horas",
        published_at=datetime(2026, 8, 13, 7, 0, tzinfo=timezone.utc),
        freshness_confidence="medium",
        is_within_24_hours=True,
        discovery_sources=["row_activation"],
    )

    prioritized = _prioritize_candidates_for_detail(
        [missing_date, verified_other, verified_card]
    )

    assert [record.linkedin_job_id for record in prioritized] == ["222", "333", "111"]
    assert _detail_priority_label(prioritized[0]) == "verified_card_date"
    assert _detail_priority_label(prioritized[1]) == "verified_date"
    assert _detail_priority_label(prioritized[2]) == "strong_metadata_missing_date"


def test_linkedin_active_detail_date_requires_matching_top_card_title(monkeypatch):
    from datetime import datetime, timezone

    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure import linkedin_jobs_pipeline as pipeline

    published = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
    record = LinkedInVacancyRecord(
        linkedin_job_id="4453249078",
        title="Machine Learning Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/4453249078",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        description_full_text="A real full job description with enough detail for validation.",
    )
    monkeypatch.setattr(
        pipeline,
        "_wait_for_active_detail_metadata",
        lambda *_args, **_kwargs: (
            "Data Scientist (Remote)",
            "hace 2 días",
            published,
            "medium",
            False,
        ),
    )
    warnings: list[str] = []

    merged = pipeline._merge_active_detail_metadata(
        record,
        object(),
        warnings=warnings,
    )

    assert merged.published_at is None
    assert merged.posted_at_text == ""
    assert merged.is_within_24_hours is False
    assert warnings[-1] == (
        "active_detail_top_card_identity:"
        "4453249078:top_card_title_mismatch"
    )


def test_linkedin_active_detail_date_applies_when_top_card_title_matches(monkeypatch):
    from datetime import datetime, timezone

    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure import linkedin_jobs_pipeline as pipeline

    published = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
    record = LinkedInVacancyRecord(
        linkedin_job_id="4453249078",
        title="Machine Learning Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/4453249078",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        description_full_text="A real full job description with enough detail for validation.",
    )
    monkeypatch.setattr(
        pipeline,
        "_wait_for_active_detail_metadata",
        lambda *_args, **_kwargs: (
            "Machine Learning Engineer",
            "hace 8 horas",
            published,
            "medium",
            True,
        ),
    )
    warnings: list[str] = []

    merged = pipeline._merge_active_detail_metadata(
        record,
        object(),
        warnings=warnings,
    )

    assert merged.published_at == published
    assert merged.posted_at_text == "hace 8 horas"
    assert merged.is_within_24_hours is True
    assert warnings[-1] == (
        "active_detail_top_card_identity:"
        "4453249078:top_card_title_match"
    )


def test_linkedin_row_date_extractor_walks_to_row_ancestor():
    import features.web_scraping.infrastructure.linkedin_query_navigation as navigation

    class TitleOnlyLocator:
        def evaluate(self, script):
            assert "parentElement" in script
            assert "rowLike" in script
            return "Visto · Adelántate a solicitar el empleo · hace 22 horas"

    posted_text, published_at, confidence, within_24h = (
        navigation._posted_at_from_row_locator(TitleOnlyLocator())
    )

    assert posted_text == "hace 22 horas"
    assert published_at is not None
    assert confidence == "medium"
    assert within_24h is True


def test_linkedin_active_detail_metadata_extracts_safe_title_and_date():
    import features.web_scraping.infrastructure.linkedin_query_navigation as navigation

    class Page:
        def evaluate(self, script):
            assert "visibleRight" in script
            assert "main h1" in script
            assert "body *" in script
            assert "dateCandidates" in script
            assert "titleDistance" in script
            assert "matchDate" in script
            return {
                "title": "Forward Deployed Engineer 글로벌 AI 유니콘 한국 초기 멤버",
                "posted": "Compartido hace 22 horas",
            }

    title, posted_text, published_at, confidence, within_24h = (
        navigation._active_detail_metadata_from_page(Page())
    )

    assert title == "Forward Deployed Engineer 글로벌 AI 유니콘 한국 초기 멤버"
    assert posted_text == "hace 22 horas"
    assert published_at is not None
    assert confidence == "medium"
    assert within_24h is True


def test_linkedin_row_activation_falls_back_to_active_detail_date_and_title(monkeypatch):
    from datetime import datetime, timezone

    import features.web_scraping.infrastructure.linkedin_query_navigation as navigation

    published = datetime(2026, 8, 13, 1, 0, tzinfo=timezone.utc)

    class Row:
        def evaluate(self, _script):
            return ""

        def click(self, **_kwargs):
            return None

    signature = (4, "job", (("data-job-id", "111"),))
    monkeypatch.setattr(navigation, "_enumerate_visible_job_rows", lambda _page: [("rows", 0, signature)])
    monkeypatch.setattr(navigation, "_resolve_row_locator", lambda _page, _signature: Row())
    monkeypatch.setattr(navigation, "_detail_identity_from_page", lambda _page: navigation.LinkedInDetailIdentity(job_id="111", canonical_detail_href="https://www.linkedin.com/jobs/view/111"))
    monkeypatch.setattr(navigation, "_wait_for_changed_detail_identity", lambda *_args, **_kwargs: (navigation.LinkedInDetailIdentity(job_id="4304546006", canonical_detail_href="https://www.linkedin.com/jobs/view/4304546006"), True))
    monkeypatch.setattr(navigation, "_wait_for_active_detail_metadata", lambda *_args, **_kwargs: ("Forward Deployed Engineer", "hace 22 horas", published, "medium", True))
    monkeypatch.setattr(navigation, "_scroll_results_panel_for_rows", lambda _page: False)

    result = navigation.discover_job_rows_via_activation(
        object(),
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        max_activations=1,
    )

    assert result.success_count == 1
    assert result.records[0].linkedin_job_id == "4304546006"
    assert result.records[0].title == "Forward Deployed Engineer"
    assert result.records[0].posted_at_text == "hace 22 horas"
    assert result.records[0].published_at == published
    assert result.records[0].is_within_24_hours is True
    assert result.outcomes[0].date_verified is True


def test_linkedin_row_activation_preserves_visible_row_posted_date(monkeypatch):
    import features.web_scraping.infrastructure.linkedin_query_navigation as navigation

    class Row:
        def evaluate(self, _script):
            return "Visto · Adelántate a solicitar el empleo · hace 3 horas"

        def click(self, **_kwargs):
            return None

    signature = (4, "job", (("data-job-id", "111"),))
    monkeypatch.setattr(navigation, "_enumerate_visible_job_rows", lambda _page: [("rows", 0, signature)])
    monkeypatch.setattr(navigation, "_resolve_row_locator", lambda _page, _signature: Row())
    monkeypatch.setattr(navigation, "_detail_identity_from_page", lambda _page: navigation.LinkedInDetailIdentity(job_id="111", canonical_detail_href="https://www.linkedin.com/jobs/view/111"))
    monkeypatch.setattr(navigation, "_wait_for_changed_detail_identity", lambda *_args, **_kwargs: (navigation.LinkedInDetailIdentity(job_id="222", canonical_detail_href="https://www.linkedin.com/jobs/view/222"), True))
    monkeypatch.setattr(navigation, "_scroll_results_panel_for_rows", lambda _page: False)

    result = navigation.discover_job_rows_via_activation(
        object(),
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        max_activations=1,
    )

    assert result.success_count == 1
    assert result.records[0].posted_at_text == "hace 3 horas"
    assert result.records[0].published_at is not None
    assert result.records[0].freshness_confidence == "medium"
    assert result.records[0].is_within_24_hours is True


@pytest.mark.parametrize(
    ("changed", "after", "expected"),
    [
        (True, SimpleNamespace(job_id="", canonical_detail_href="", allowlisted_attributes=(("data-job-id", "x"),)), "row_activation_no_job_id"),
        (False, SimpleNamespace(job_id="111", canonical_detail_href="https://www.linkedin.com/jobs/view/111", allowlisted_attributes=()), "row_activation_no_change"),
    ],
)


def test_linkedin_row_activation_emits_distinct_identity_failures(
    monkeypatch,
    changed,
    after,
    expected,
):
    import features.web_scraping.infrastructure.linkedin_query_navigation as navigation

    class Row:
        def click(self, **_kwargs):
            return None

    signature = (4, "job", (("data-job-id", "111"),))
    monkeypatch.setattr(navigation, "_enumerate_visible_job_rows", lambda _page: [("rows", 0, signature)])
    monkeypatch.setattr(navigation, "_resolve_row_locator", lambda _page, _signature: Row())
    monkeypatch.setattr(navigation, "_detail_identity_from_page", lambda _page: navigation.LinkedInDetailIdentity(job_id="111", canonical_detail_href="https://www.linkedin.com/jobs/view/111"))
    monkeypatch.setattr(navigation, "_wait_for_changed_detail_identity", lambda *_args, **_kwargs: (after, changed))
    monkeypatch.setattr(navigation, "_scroll_results_panel_for_rows", lambda _page: False)

    result = navigation.discover_job_rows_via_activation(
        object(),
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        max_activations=1,
    )

    assert result.outcomes[0].outcome == expected
    assert result.records == []


def test_linkedin_row_activation_has_hard_cap_and_reenumerates_virtualized_rows(monkeypatch):
    import features.web_scraping.infrastructure.linkedin_query_navigation as navigation

    class Row:
        def click(self, **_kwargs):
            return None

    rows = [("rows", index, (index, f"job-{index}", ())) for index in range(25)]
    monkeypatch.setattr(navigation, "_enumerate_visible_job_rows", lambda _page: rows)
    monkeypatch.setattr(navigation, "_resolve_row_locator", lambda _page, _signature: Row())
    monkeypatch.setattr(navigation, "_detail_identity_from_page", lambda _page: navigation.LinkedInDetailIdentity())
    counter = {"value": 0}

    def changed_identity(*_args, **_kwargs):
        counter["value"] += 1
        job_id = str(counter["value"])
        return navigation.LinkedInDetailIdentity(
            job_id=job_id,
            canonical_detail_href=f"https://www.linkedin.com/jobs/view/{job_id}",
        ), True

    monkeypatch.setattr(navigation, "_wait_for_changed_detail_identity", changed_identity)
    monkeypatch.setattr(navigation, "_scroll_results_panel_for_rows", lambda _page: False)

    result = navigation.discover_job_rows_via_activation(
        object(),
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
    )

    assert result.activation_count == navigation.MAX_ROW_ACTIVATIONS_PER_QUERY == 20
    assert result.success_count == 20


def test_linkedin_row_signature_changes_with_virtualized_content_not_visual_index():
    import features.web_scraping.infrastructure.linkedin_query_navigation as navigation

    class Link:
        def __init__(self, href):
            self.href = href

        @property
        def first(self):
            return self

        def get_attribute(self, name):
            return self.href if name == "href" else None

    class Row:
        def __init__(self, href, job_id):
            self.href = href
            self.job_id = job_id

        def evaluate(self, _script):
            return 12

        def locator(self, _selector):
            return Link(self.href)

        def get_attribute(self, name):
            return self.job_id if name == "data-job-id" else None

    first = navigation._row_signature(Row("/jobs/view/111", "111"))
    second = navigation._row_signature(Row("/jobs/view/222", "222"))

    assert first[0] == second[0] == 12
    assert first != second
    assert first[1] == "111"
    assert second[1] == "222"


def test_linkedin_row_ids_merge_before_detail_and_mark_duplicate_source():
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        merge_row_activation_records,
    )

    dom = LinkedInVacancyRecord(
        linkedin_job_id="111",
        title="AI Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/111",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
        discovery_sources=["card"],
    )
    row = dom.model_copy(update={"discovery_sources": ["row_activation"]})

    merged, dom_contributed, row_contributed = merge_row_activation_records([dom], [row])

    assert len(merged) == 1
    assert merged[0].discovery_sources == ["card", "row_activation"]
    assert dom_contributed is True
    assert row_contributed is True


def test_linkedin_row_merge_preserves_verified_row_date_for_duplicate_dom_candidate():
    from datetime import datetime, timezone

    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        merge_row_activation_records,
    )

    source = "https://www.linkedin.com/jobs/search/?keywords=AI"
    published = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
    dom = LinkedInVacancyRecord(
        linkedin_job_id="4453249078",
        title="Machine Learning Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/4453249078",
        source_url=source,
        discovery_sources=["job_href"],
    )
    row = LinkedInVacancyRecord(
        linkedin_job_id="4453249078",
        title="Machine Learning Engineer",
        posted_at_text="hace 12 horas",
        published_at=published,
        freshness_confidence="medium",
        is_within_24_hours=True,
        canonical_url="https://www.linkedin.com/jobs/view/4453249078",
        source_url=source,
        discovery_sources=["row_activation"],
    )

    merged, dom_contributed, row_contributed = merge_row_activation_records([dom], [row])

    assert len(merged) == 1
    assert merged[0].posted_at_text == "hace 12 horas"
    assert merged[0].published_at == published
    assert merged[0].freshness_confidence == "medium"
    assert merged[0].is_within_24_hours is True
    assert merged[0].discovery_sources == ["job_href", "row_activation"]
    assert dom_contributed is True
    assert row_contributed is True


def test_linkedin_discovery_modes_are_explicit():
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        discovery_mode_for_sources,
    )

    assert discovery_mode_for_sources(dom_contributed=True, row_contributed=True) == "multi_source_with_row_activation"
    assert discovery_mode_for_sources(dom_contributed=False, row_contributed=True) == "row_activation"
    assert discovery_mode_for_sources(dom_contributed=True, row_contributed=False) == "standard"


def test_linkedin_visual_manifest_omits_missing_before_main_capture(tmp_path):
    from features.web_scraping.infrastructure.linkedin_visual_diagnostics import (
        LinkedInVisualDiagnosticsCollector,
        _VisualSearchRun,
    )

    collector = LinkedInVisualDiagnosticsCollector(tmp_path)
    collector.base_dir.mkdir(parents=True, exist_ok=True)
    run = _VisualSearchRun(collector, query="AI")
    run.before_main_capture = False
    (tmp_path / "visual-diagnostics" / "after-main.png").write_text("image", encoding="utf-8")
    run.after_main_capture = True
    run.finalize()

    manifest = json.loads((tmp_path / "visual-diagnostics" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["before_main"] is None
    assert manifest["artifacts"]["after_main"] == "after-main.png"


def test_linkedin_rejected_card_visual_debug_writes_focused_artifacts(tmp_path):
    from features.web_scraping.infrastructure.linkedin_visual_diagnostics import (
        capture_rejected_job_card_visual_debug,
    )

    class Locator:
        def __init__(self, selector):
            self.selector = selector

        @property
        def first(self):
            return self

        def count(self):
            return 1 if "4453249078" in self.selector else 0

        def screenshot(self, *, path):
            Path(path).write_bytes(b"png")

    class Page:
        def evaluate(self, script, job_id):
            assert "innerText" in script
            assert job_id == "4453249078"
            return {
                "found": True,
                "job_id": "4453249078",
                "selector_kind": "attribute",
                "tag": "li",
                "class_tokens": ["scaffold-layout__list-item"],
                "has_href": False,
                "has_data_job_id": False,
                "has_data_occludable_job_id": True,
                "title_present": False,
                "date_texts": [],
                "bbox": {"x": 112, "y": 1246, "width": 488, "height": 130},
                "ancestor_chain": [],
            }

        def locator(self, selector):
            return Locator(selector)

    created = capture_rejected_job_card_visual_debug(
        Page(),
        tmp_path,
        job_id="4453249078",
        reason="outside_24_hours",
    )

    assert sorted(created) == [
        "rejected-card-4453249078-outside_24_hours.json",
        "rejected-card-4453249078-outside_24_hours.png",
    ]
    payload = json.loads(
        (tmp_path / "rejected-card-4453249078-outside_24_hours.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["job_id"] == "4453249078"
    assert payload["found"] is True
    assert "innerText" not in json.dumps(payload)





def test_linkedin_active_detail_identity_accepts_ordered_title_prefix():
    from types import SimpleNamespace

    from features.web_scraping.infrastructure.linkedin_jobs_pipeline import (
        _active_detail_top_card_date_identity_status,
    )

    matched, reason = _active_detail_top_card_date_identity_status(
        SimpleNamespace(title="Machine Learning Engineer"),
        "Machine Learning Engineer - AI (Remote)",
    )

    assert matched is True
    assert reason == "top_card_title_prefix_compatible"


def test_linkedin_active_detail_identity_rejects_too_short_title_prefix():
    from types import SimpleNamespace

    from features.web_scraping.infrastructure.linkedin_jobs_pipeline import (
        _active_detail_top_card_date_identity_status,
    )

    matched, reason = _active_detail_top_card_date_identity_status(
        SimpleNamespace(title="AI Engineer"),
        "AI Engineer Manager",
    )

    assert matched is False
    assert reason == "top_card_title_mismatch"

def test_linkedin_card_activation_date_evidence_merges_when_title_matches(monkeypatch):
    from datetime import datetime, timezone

    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    import features.web_scraping.infrastructure.linkedin_jobs_pipeline as pipeline

    class Locator:
        @property
        def first(self):
            return self

        def count(self):
            return 1

        def scroll_into_view_if_needed(self, **_kwargs):
            return None

        def click(self, **_kwargs):
            return None

    class Page:
        def locator(self, selector):
            assert "4453249078" in selector
            return Locator()

    published_at = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        pipeline,
        "_wait_for_active_detail_metadata",
        lambda page, require_date=True: (
            "Machine Learning Engineer",
            "Publicado de nuevo hace 37 minutos",
            published_at,
            "medium",
            True,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_active_detail_metadata_warning",
        lambda record: f"active_detail_date_selected:{record.linkedin_job_id}:within_24_hours",
    )
    record = LinkedInVacancyRecord(
        linkedin_job_id="4453249078",
        title="Machine Learning Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/4453249078",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
    )
    warnings = []

    updated = pipeline._activate_visible_search_card_date_evidence(
        Page(),
        record,
        warnings=warnings,
    )

    assert updated.published_at == published_at
    assert updated.posted_at_text == "Publicado de nuevo hace 37 minutos"
    assert updated.is_within_24_hours is True
    assert "card_activation_date_identity:4453249078:top_card_title_match" in warnings
    assert "card_activation_date_verified:4453249078:within_24_hours" in warnings


def test_linkedin_card_activation_date_evidence_rejects_mismatched_active_detail(monkeypatch):
    from datetime import datetime, timezone

    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    import features.web_scraping.infrastructure.linkedin_jobs_pipeline as pipeline

    class Locator:
        @property
        def first(self):
            return self

        def count(self):
            return 1

        def scroll_into_view_if_needed(self, **_kwargs):
            return None

        def click(self, **_kwargs):
            return None

    class Page:
        def locator(self, _selector):
            return Locator()

    monkeypatch.setattr(
        pipeline,
        "_wait_for_active_detail_metadata",
        lambda page, require_date=True: (
            "[Tech] AI Engineer - AI Agent 엔지니어",
            "Publicado de nuevo hace 37 minutos",
            datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc),
            "medium",
            True,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_active_detail_metadata_warning",
        lambda record: f"active_detail_date_selected:{record.linkedin_job_id}:within_24_hours",
    )
    record = LinkedInVacancyRecord(
        linkedin_job_id="4453249078",
        title="Machine Learning Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/4453249078",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
    )
    warnings = []

    updated = pipeline._activate_visible_search_card_date_evidence(
        Page(),
        record,
        warnings=warnings,
    )

    assert updated is record
    assert updated.published_at is None
    assert "card_activation_date_identity:4453249078:top_card_title_mismatch" in warnings
    assert not any(item.startswith("card_activation_date_verified:") for item in warnings)


def test_linkedin_card_title_activation_date_evidence_merges_when_title_matches(monkeypatch):
    from datetime import datetime, timezone

    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    import features.web_scraping.infrastructure.linkedin_jobs_pipeline as pipeline

    class LinkLocator:
        @property
        def first(self):
            return self

        def count(self):
            return 1

        def click(self, **_kwargs):
            return None

    class RowLocator:
        def __init__(self, text):
            self.text = text

        def inner_text(self, **_kwargs):
            return self.text

        def scroll_into_view_if_needed(self, **_kwargs):
            return None

        def locator(self, _selector):
            return LinkLocator()

        def click(self, **_kwargs):
            return None

    class RowsLocator:
        def __init__(self):
            self.rows = [
                RowLocator("Other Job Company Seoul"),
                RowLocator("Machine Learning Engineer GOWARD Seoul"),
            ]

        def count(self):
            return len(self.rows)

        def nth(self, index):
            return self.rows[index]

    class Page:
        def locator(self, _selector):
            return RowsLocator()

    published_at = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        pipeline,
        "_wait_for_active_detail_metadata",
        lambda page, require_date=True: (
            "Machine Learning Engineer",
            "Publicado de nuevo hace 12 minutos",
            published_at,
            "medium",
            True,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_active_detail_metadata_warning",
        lambda record: f"active_detail_date_selected:{record.linkedin_job_id}:within_24_hours",
    )
    record = LinkedInVacancyRecord(
        linkedin_job_id="4453249078",
        title="Machine Learning Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/4453249078",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
    )
    warnings = []

    updated = pipeline._activate_visible_search_card_by_title_date_evidence(
        Page(),
        record,
        warnings=warnings,
    )

    assert updated.published_at == published_at
    assert updated.posted_at_text == "Publicado de nuevo hace 12 minutos"
    assert updated.is_within_24_hours is True
    assert "card_title_activation_identity:4453249078:top_card_title_match" in warnings
    assert "card_title_activation_verified:4453249078:within_24_hours" in warnings


def test_linkedin_card_title_activation_date_evidence_skips_without_title():
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    import features.web_scraping.infrastructure.linkedin_jobs_pipeline as pipeline

    class Page:
        def locator(self, _selector):
            raise AssertionError("should not inspect rows without title")

    record = LinkedInVacancyRecord(
        linkedin_job_id="4453249078",
        title="",
        canonical_url="https://www.linkedin.com/jobs/view/4453249078",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
    )
    warnings = []

    updated = pipeline._activate_visible_search_card_by_title_date_evidence(
        Page(),
        record,
        warnings=warnings,
    )

    assert updated is record
    assert warnings == ["card_title_activation_failed:4453249078:missing_title"]


def test_linkedin_card_title_activation_date_evidence_rejects_wrong_active_title(monkeypatch):
    from datetime import datetime, timezone

    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    import features.web_scraping.infrastructure.linkedin_jobs_pipeline as pipeline

    class LinkLocator:
        @property
        def first(self):
            return self

        def count(self):
            return 1

        def click(self, **_kwargs):
            return None

    class RowLocator:
        def inner_text(self, **_kwargs):
            return "Machine Learning Engineer GOWARD Seoul"

        def scroll_into_view_if_needed(self, **_kwargs):
            return None

        def locator(self, _selector):
            return LinkLocator()

        def click(self, **_kwargs):
            return None

    class RowsLocator:
        def count(self):
            return 1

        def nth(self, _index):
            return RowLocator()

    class Page:
        def locator(self, _selector):
            return RowsLocator()

    monkeypatch.setattr(
        pipeline,
        "_wait_for_active_detail_metadata",
        lambda page, require_date=True: (
            "[Tech] AI Engineer - LLM 엔지니어",
            "Publicado de nuevo hace 12 minutos",
            datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc),
            "medium",
            True,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_active_detail_metadata_warning",
        lambda record: f"active_detail_date_selected:{record.linkedin_job_id}:within_24_hours",
    )
    record = LinkedInVacancyRecord(
        linkedin_job_id="4453249078",
        title="Machine Learning Engineer",
        canonical_url="https://www.linkedin.com/jobs/view/4453249078",
        source_url="https://www.linkedin.com/jobs/search/?keywords=AI",
    )
    warnings = []

    updated = pipeline._activate_visible_search_card_by_title_date_evidence(
        Page(),
        record,
        warnings=warnings,
    )

    assert updated is record
    assert updated.published_at is None
    assert "card_title_activation_identity:4453249078:top_card_title_mismatch" in warnings
    assert not any(item.startswith("card_title_activation_verified:") for item in warnings)

def test_linkedin_unverified_date_visual_evidence_captures_detail_and_card():
    from types import SimpleNamespace

    from features.web_scraping.infrastructure.linkedin_jobs_pipeline import (
        _capture_unverified_date_visual_evidence,
    )

    class VisualDiagnostics:
        def __init__(self):
            self.calls = []

        def capture_active_detail_date(self, page, *, job_id, reason):
            self.calls.append(("active", page, job_id, reason))

        def capture_rejected_candidate_card(self, page, *, job_id, reason):
            self.calls.append(("card", page, job_id, reason))

    visual = VisualDiagnostics()
    page = object()
    warnings = []

    _capture_unverified_date_visual_evidence(
        visual,
        page,
        SimpleNamespace(linkedin_job_id="4453249078"),
        warnings=warnings,
    )

    assert visual.calls == [
        ("active", page, "4453249078", "unverified_posted_date"),
        ("card", page, "4453249078", "unverified_posted_date"),
    ]
    assert warnings == ["unverified_date_visual_debug:4453249078"]


def test_linkedin_visual_diagnostics_caps_cover_current_unverified_batch():
    from features.web_scraping.infrastructure.linkedin_visual_diagnostics import (
        MAX_ACTIVE_DETAIL_DATE_VISUAL_CAPTURES,
        MAX_REJECTED_CARD_VISUAL_CAPTURES,
    )

    assert MAX_ACTIVE_DETAIL_DATE_VISUAL_CAPTURES >= 7
    assert MAX_REJECTED_CARD_VISUAL_CAPTURES >= 7

def test_linkedin_html_collector_adds_rejected_card_artifacts_to_manifest(tmp_path):
    from features.web_scraping.infrastructure.linkedin_visual_diagnostics import (
        LinkedInHTMLDiagnosticsCollector,
    )

    class Locator:
        @property
        def first(self):
            return self

        def count(self):
            return 1

        def screenshot(self, *, path):
            Path(path).write_bytes(b"png")

    class Page:
        def evaluate(self, _script, job_id):
            return {
                "found": True,
                "job_id": job_id,
                "selector_kind": "attribute",
                "tag": "li",
                "bbox": {"x": 0, "y": 0, "width": 10, "height": 10},
            }

        def locator(self, _selector):
            return Locator()

    collector = LinkedInHTMLDiagnosticsCollector(tmp_path, job_uid="job-1")
    collector.base_dir.mkdir(parents=True, exist_ok=True)
    (collector.base_dir / "manifest.json").write_text(
        json.dumps({"artifacts": {}}, ensure_ascii=False),
        encoding="utf-8",
    )

    collector.capture_rejected_candidate_card(
        Page(),
        job_id="4453249078",
        reason="outside_24_hours",
    )

    manifest = json.loads(
        (collector.base_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifacts"][
        "rejected-card-4453249078-outside_24_hours.json"
    ] == "rejected-card-4453249078-outside_24_hours.json"
    assert manifest["artifacts"][
        "rejected-card-4453249078-outside_24_hours.png"
    ] == "rejected-card-4453249078-outside_24_hours.png"


def test_linkedin_active_detail_date_visual_debug_writes_focused_artifacts(tmp_path):
    from features.web_scraping.infrastructure.linkedin_visual_diagnostics import (
        LinkedInHTMLDiagnosticsCollector,
    )

    class Locator:
        @property
        def first(self):
            return self

        def count(self):
            return 1

        def screenshot(self, *, path):
            Path(path).write_bytes(b"png")

    class Page:
        def evaluate(self, script):
            assert "date_candidates" in script
            return {
                "found": True,
                "title_present": True,
                "title_bbox": {"x": 640, "y": 190, "width": 400, "height": 40},
                "detail_bbox": {"x": 620, "y": 120, "width": 600, "height": 700},
                "date_candidates": [
                    {
                        "date_text": "hace 3 días",
                        "tag": "span",
                        "class_tokens": ["tvm__text"],
                        "bbox": {"x": 730, "y": 234, "width": 70, "height": 20},
                        "distance_from_title": 44,
                    }
                ],
            }

        def locator(self, _selector):
            return Locator()

    collector = LinkedInHTMLDiagnosticsCollector(tmp_path, job_uid="job-1")
    collector.base_dir.mkdir(parents=True, exist_ok=True)
    (collector.base_dir / "manifest.json").write_text(
        json.dumps({"artifacts": {}}, ensure_ascii=False),
        encoding="utf-8",
    )

    collector.capture_active_detail_date(
        Page(),
        job_id="4450481657",
        reason="outside_24_hours",
    )

    manifest = json.loads(
        (collector.base_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifacts"][
        "active-detail-date-4450481657-outside_24_hours.json"
    ] == "active-detail-date-4450481657-outside_24_hours.json"
    assert manifest["artifacts"][
        "active-detail-date-4450481657-outside_24_hours.png"
    ] == "active-detail-date-4450481657-outside_24_hours.png"
    payload = json.loads(
        (
            collector.base_dir
            / "active-detail-date-4450481657-outside_24_hours.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["date_candidates"][0]["date_text"] == "hace 3 días"
    assert "body" not in json.dumps(payload).casefold()







def test_linkedin_recruiter_profile_url_from_visible_recruiter_section():
    import features.web_scraping.infrastructure.linkedin_jobs_pipeline as pipeline

    assert pipeline._safe_linkedin_profile_url_from_href(
        "https://www.linkedin.com/in/seoul-recruiter?miniProfileUrn=secret"
    ) == "https://www.linkedin.com/in/seoul-recruiter"

    class Page:
        def evaluate(self, _script):
            return [
                "https://evil.example/in/bad",
                "https://www.linkedin.com/in/seoul-recruiter?trackingId=secret",
            ]

    assert pipeline._safe_recruiter_profile_url_from_active_detail(Page()) == (
        "https://www.linkedin.com/in/seoul-recruiter"
    )



def test_linkedin_recruiter_profile_url_from_hiring_team_text_block():
    import features.web_scraping.infrastructure.linkedin_jobs_pipeline as pipeline

    class Page:
        def evaluate(self, script):
            assert "equipo de contrataci" in script
            return ["https://www.linkedin.com/in/seoul-recruiter?trk=jobs"]

    assert pipeline._safe_recruiter_profile_url_from_active_detail(Page()) == (
        "https://www.linkedin.com/in/seoul-recruiter"
    )


def test_linkedin_recruiter_location_evidence_accepts_korea(monkeypatch):
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    import features.web_scraping.infrastructure.linkedin_jobs_pipeline as pipeline

    class RecruiterPage:
        def goto(self, url, **_kwargs):
            assert url == "https://www.linkedin.com/in/seoul-recruiter"

        def wait_for_timeout(self, _ms):
            return None

        def evaluate(self, _script):
            return {"has_korea": True, "has_japan": False, "has_united_states": False}

        def close(self):
            return None

    class Page:
        context = SimpleNamespace(new_page=lambda: RecruiterPage())

        def evaluate(self, _script):
            return ["https://www.linkedin.com/in/seoul-recruiter?trk=job"]

    monkeypatch.setattr(
        pipeline,
        "_wait_for_active_detail_metadata",
        lambda page, require_date=False: ("Machine Learning Engineer", "", None, "", False),
    )
    record = LinkedInVacancyRecord(
        linkedin_job_id="4453510505",
        title="Machine Learning Engineer",
        location="Asia-Pacífico",
        canonical_url="https://www.linkedin.com/jobs/view/4453510505",
        source_url="https://www.linkedin.com/jobs/search?location=South+Korea",
    )
    warnings = []

    signal = pipeline._recruiter_location_evidence_signal(
        Page(),
        record,
        "South Korea",
        None,
        warnings=warnings,
    )

    assert signal == "korea office"
    assert warnings == ["recruiter_location_evidence:4453510505:recruiter_profile_korea"]


def test_linkedin_recruiter_negative_does_not_block_company_positive(monkeypatch):
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    import features.web_scraping.infrastructure.linkedin_jobs_pipeline as pipeline

    class RecruiterPage:
        def goto(self, _url, **_kwargs):
            return None

        def wait_for_timeout(self, _ms):
            return None

        def evaluate(self, _script):
            return {"has_korea": False, "has_japan": False, "has_united_states": True}

        def close(self):
            return None

    class Page:
        context = SimpleNamespace(new_page=lambda: RecruiterPage())

        def evaluate(self, _script):
            return ["https://www.linkedin.com/in/us-recruiter"]

    monkeypatch.setattr(
        pipeline,
        "_wait_for_active_detail_metadata",
        lambda page, require_date=False: ("Machine Learning Engineer", "", None, "", False),
    )
    record = LinkedInVacancyRecord(
        linkedin_job_id="4453510505",
        title="Machine Learning Engineer",
        location="Asia-Pacífico",
        canonical_url="https://www.linkedin.com/jobs/view/4453510505",
        source_url="https://www.linkedin.com/jobs/search?location=South+Korea",
    )
    warnings = []

    recruiter_signal = pipeline._recruiter_location_evidence_signal(
        Page(),
        record,
        "South Korea",
        None,
        warnings=warnings,
    )

    country_signal_text = "korea office"
    if recruiter_signal:
        country_signal_text += f"\n{recruiter_signal}"

    assert recruiter_signal == ""
    assert pipeline._visible_location_matches_requested_scope(
        record.location,
        "South Korea",
        country_signal_text,
    )
    assert warnings == [
        "recruiter_location_evidence:4453510505:recruiter_profile_united_states"
    ]


def test_linkedin_recruiter_location_visual_debug_writes_artifacts(tmp_path):
    from features.web_scraping.infrastructure.linkedin_visual_diagnostics import (
        LinkedInHTMLDiagnosticsCollector,
    )

    class Page:
        def evaluate(self, script):
            assert "locationMatch" in script
            return {
                "page_path": "/in/seoul-recruiter/",
                "has_korea": True,
                "has_japan": False,
                "has_united_states": False,
                "location_text": "Seoul, South Korea",
                "text_length": 800,
            }

        def screenshot(self, *, path, full_page):
            assert full_page is False
            Path(path).write_bytes(b"png")

    collector = LinkedInHTMLDiagnosticsCollector(tmp_path, job_uid="job-1")
    collector.base_dir.mkdir(parents=True, exist_ok=True)
    (collector.base_dir / "manifest.json").write_text(
        json.dumps({"artifacts": {}}, ensure_ascii=False),
        encoding="utf-8",
    )

    collector.capture_recruiter_location(
        Page(),
        job_id="4453510505",
        reason="remote_scope_review",
    )

    manifest = json.loads(
        (collector.base_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifacts"][
        "recruiter-location-4453510505-remote_scope_review.json"
    ] == "recruiter-location-4453510505-remote_scope_review.json"
    assert manifest["artifacts"][
        "recruiter-location-4453510505-remote_scope_review.png"
    ] == "recruiter-location-4453510505-remote_scope_review.png"

def test_linkedin_company_about_url_accepts_company_subpages():
    import features.web_scraping.infrastructure.linkedin_jobs_pipeline as pipeline

    assert pipeline._safe_linkedin_company_slug_from_href(
        "/company/hire-feed/life/?trk=jobs"
    ) == "hire-feed"
    assert pipeline._safe_company_about_url_from_slug("hire-feed") == (
        "https://www.linkedin.com/company/hire-feed/about"
    )


def test_linkedin_company_about_url_from_active_detail_uses_first_safe_company_link():
    import features.web_scraping.infrastructure.linkedin_jobs_pipeline as pipeline

    class Page:
        def evaluate(self, _script):
            return [
                "https://evil.example/company/bad/life",
                "https://www.linkedin.com/company/hire-feed/life/?trackingId=secret",
            ]

    assert pipeline._safe_company_about_url_from_active_detail(Page()) == (
        "https://www.linkedin.com/company/hire-feed/about"
    )

def test_linkedin_company_about_location_evidence_accepts_korea(monkeypatch):
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    import features.web_scraping.infrastructure.linkedin_jobs_pipeline as pipeline

    class CompanyPage:
        def __init__(self):
            self.closed = False

        def goto(self, url, **_kwargs):
            assert url == "https://www.linkedin.com/company/example-ai/about"

        def wait_for_timeout(self, _ms):
            return None

        def evaluate(self, _script):
            return {
                "has_korea": True,
                "has_japan": False,
                "has_united_states": False,
                "text_length": 500,
            }

        def close(self):
            self.closed = True

    class Context:
        def __init__(self):
            self.company_page = CompanyPage()

        def new_page(self):
            return self.company_page

    class Page:
        def __init__(self):
            self.context = Context()

        def evaluate(self, _script):
            return "https://www.linkedin.com/company/example-ai?trk=jobs"

    class VisualDiagnostics:
        def __init__(self):
            self.calls = []

        def capture_company_about_location(self, page, *, job_id, reason):
            self.calls.append((page, job_id, reason))

    monkeypatch.setattr(
        pipeline,
        "_wait_for_active_detail_metadata",
        lambda page, require_date=False: ("Machine Learning Engineer", "", None, "", False),
    )
    record = LinkedInVacancyRecord(
        linkedin_job_id="4453510505",
        title="Machine Learning Engineer",
        location="Asia-Pacífico",
        canonical_url="https://www.linkedin.com/jobs/view/4453510505",
        source_url="https://www.linkedin.com/jobs/search?location=South+Korea",
    )
    page = Page()
    visual = VisualDiagnostics()
    warnings = []

    signal = pipeline._company_about_location_evidence_signal(
        page,
        record,
        "South Korea",
        visual,
        warnings=warnings,
    )

    assert signal == "korea office"
    assert warnings == ["company_about_location_evidence:4453510505:company_about_korea"]
    assert visual.calls == [(page.context.company_page, "4453510505", "remote_scope_review")]
    assert page.context.company_page.closed is True


def test_linkedin_company_about_location_evidence_rejects_united_states(monkeypatch):
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    import features.web_scraping.infrastructure.linkedin_jobs_pipeline as pipeline

    class CompanyPage:
        def goto(self, url, **_kwargs):
            assert url == "https://www.linkedin.com/company/hire-feed/about"

        def wait_for_timeout(self, _ms):
            return None

        def evaluate(self, _script):
            return {
                "has_korea": False,
                "has_japan": False,
                "has_united_states": True,
                "text_length": 800,
            }

        def close(self):
            return None

    class Page:
        context = SimpleNamespace(new_page=lambda: CompanyPage())

        def evaluate(self, _script):
            return "/company/hire-feed/about/?trackingId=secret"

    monkeypatch.setattr(
        pipeline,
        "_wait_for_active_detail_metadata",
        lambda page, require_date=False: ("Machine Learning Engineer - AI (Remote)", "", None, "", False),
    )
    record = LinkedInVacancyRecord(
        linkedin_job_id="4453510505",
        title="Machine Learning Engineer",
        location="Asia-Pacífico",
        canonical_url="https://www.linkedin.com/jobs/view/4453510505",
        source_url="https://www.linkedin.com/jobs/search?location=South+Korea",
    )
    warnings = []

    signal = pipeline._company_about_location_evidence_signal(
        Page(),
        record,
        "South Korea",
        None,
        warnings=warnings,
    )

    assert signal == ""
    assert warnings == [
        "company_about_location_evidence:4453510505:company_about_united_states"
    ]


def test_linkedin_company_about_location_evidence_requires_active_identity(monkeypatch):
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    import features.web_scraping.infrastructure.linkedin_jobs_pipeline as pipeline

    class Context:
        def new_page(self):
            raise AssertionError("should not open company page on identity mismatch")

    class Page:
        context = Context()

        def evaluate(self, _script):
            raise AssertionError("should not inspect company link on identity mismatch")

    monkeypatch.setattr(
        pipeline,
        "_wait_for_active_detail_metadata",
        lambda page, require_date=False: ("Unrelated AI Engineer", "", None, "", False),
    )
    record = LinkedInVacancyRecord(
        linkedin_job_id="4453510505",
        title="Machine Learning Engineer",
        location="Asia-Pacífico",
        canonical_url="https://www.linkedin.com/jobs/view/4453510505",
        source_url="https://www.linkedin.com/jobs/search?location=South+Korea",
    )
    warnings = []

    signal = pipeline._company_about_location_evidence_signal(
        Page(),
        record,
        "South Korea",
        None,
        warnings=warnings,
    )

    assert signal == ""
    assert warnings == [
        "company_about_location_evidence:4453510505:identity_top_card_title_mismatch"
    ]


def test_linkedin_company_about_location_visual_debug_writes_artifacts(tmp_path):
    from features.web_scraping.infrastructure.linkedin_visual_diagnostics import (
        LinkedInHTMLDiagnosticsCollector,
    )

    class Page:
        def evaluate(self, script):
            assert "headquarters" in script
            return {
                "page_path": "/company/hire-feed/about/",
                "has_korea": False,
                "has_japan": False,
                "has_united_states": True,
                "headquarters_text": "Menlo Park, California, United States",
                "location_text": "Menlo Park, California",
                "text_length": 1200,
            }

        def screenshot(self, *, path, full_page):
            assert full_page is False
            Path(path).write_bytes(b"png")

    collector = LinkedInHTMLDiagnosticsCollector(tmp_path, job_uid="job-1")
    collector.base_dir.mkdir(parents=True, exist_ok=True)
    (collector.base_dir / "manifest.json").write_text(
        json.dumps({"artifacts": {}}, ensure_ascii=False),
        encoding="utf-8",
    )

    collector.capture_company_about_location(
        Page(),
        job_id="4453510505",
        reason="remote_scope_review",
    )

    manifest = json.loads(
        (collector.base_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifacts"][
        "company-about-location-4453510505-remote_scope_review.json"
    ] == "company-about-location-4453510505-remote_scope_review.json"
    assert manifest["artifacts"][
        "company-about-location-4453510505-remote_scope_review.png"
    ] == "company-about-location-4453510505-remote_scope_review.png"
    payload = json.loads(
        (
            collector.base_dir
            / "company-about-location-4453510505-remote_scope_review.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["has_united_states"] is True
    assert payload["location_text"] == "Menlo Park, California"
    assert "trackingId" not in json.dumps(payload)

def test_linkedin_remote_scope_page_evidence_accepts_company_korea_when_identity_matches(monkeypatch):
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    import features.web_scraping.infrastructure.linkedin_jobs_pipeline as pipeline

    class Page:
        def evaluate(self, _script):
            return {
                "company_has_korea": True,
                "recruiter_has_korea": False,
                "location_has_korea": False,
                "company_has_japan": False,
                "recruiter_has_japan": False,
                "location_has_japan": False,
                "company_node_count": 3,
                "recruiter_node_count": 1,
                "location_node_count": 4,
            }

    monkeypatch.setattr(
        pipeline,
        "_wait_for_active_detail_metadata",
        lambda page, require_date=False: ("Machine Learning Engineer - AI (Remote)", "", None, "", False),
    )
    record = LinkedInVacancyRecord(
        linkedin_job_id="4453510505",
        title="Machine Learning Engineer",
        location="Asia-Pacífico",
        canonical_url="https://www.linkedin.com/jobs/view/4453510505",
        source_url="https://www.linkedin.com/jobs/search?location=South+Korea",
    )
    warnings = []

    signal = pipeline._remote_scope_page_country_evidence_signal(
        Page(),
        record,
        "South Korea",
        warnings=warnings,
    )

    assert signal == "korea office"
    assert "remote_scope_structured_evidence:4453510505:company_location_korea" in warnings
    assert pipeline._visible_location_matches_requested_scope(
        record.location,
        "South Korea",
        signal,
    )


def test_linkedin_remote_scope_page_evidence_rejects_when_identity_mismatches(monkeypatch):
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    import features.web_scraping.infrastructure.linkedin_jobs_pipeline as pipeline

    class Page:
        def evaluate(self, _script):
            raise AssertionError("should not inspect page text after identity mismatch")

    monkeypatch.setattr(
        pipeline,
        "_wait_for_active_detail_metadata",
        lambda page, require_date=False: ("Unrelated AI Engineer", "", None, "", False),
    )
    record = LinkedInVacancyRecord(
        linkedin_job_id="4453510505",
        title="Machine Learning Engineer",
        location="Asia-Pacífico",
        canonical_url="https://www.linkedin.com/jobs/view/4453510505",
        source_url="https://www.linkedin.com/jobs/search?location=South+Korea",
    )
    warnings = []

    signal = pipeline._remote_scope_page_country_evidence_signal(
        Page(),
        record,
        "South Korea",
        warnings=warnings,
    )

    assert signal == ""
    assert warnings == [
        "remote_scope_structured_evidence:4453510505:identity_top_card_title_mismatch"
    ]


def test_linkedin_remote_scope_page_evidence_accepts_recruiter_korea_when_identity_matches(monkeypatch):
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    import features.web_scraping.infrastructure.linkedin_jobs_pipeline as pipeline

    class Page:
        def evaluate(self, _script):
            return {
                "company_has_korea": False,
                "recruiter_has_korea": True,
                "location_has_korea": False,
                "company_node_count": 1,
                "recruiter_node_count": 2,
                "location_node_count": 3,
            }

    monkeypatch.setattr(
        pipeline,
        "_wait_for_active_detail_metadata",
        lambda page, require_date=False: ("Machine Learning Engineer", "", None, "", False),
    )
    record = LinkedInVacancyRecord(
        linkedin_job_id="4453510505",
        title="Machine Learning Engineer",
        location="Asia-Pacífico",
        canonical_url="https://www.linkedin.com/jobs/view/4453510505",
        source_url="https://www.linkedin.com/jobs/search?location=South+Korea",
    )
    warnings = []

    signal = pipeline._remote_scope_page_country_evidence_signal(
        Page(),
        record,
        "South Korea",
        warnings=warnings,
    )

    assert signal == "korea office"
    assert "remote_scope_structured_evidence:4453510505:recruiter_location_korea" in warnings

def test_linkedin_company_recruiter_location_visual_debug_writes_artifacts(tmp_path):
    from features.web_scraping.infrastructure.linkedin_visual_diagnostics import (
        LinkedInHTMLDiagnosticsCollector,
    )

    class Locator:
        def __init__(self, selector):
            self.selector = selector

        @property
        def first(self):
            return self

        def count(self):
            return 1 if "company" in self.selector or self.selector == "main, .jobs-search__job-details, .job-details-jobs-unified-top-card, .jobs-unified-top-card" else 0

        def screenshot(self, *, path):
            Path(path).write_bytes(b"png")

    class Page:
        def evaluate(self, script):
            assert "recruiter_candidate_count" in script
            return {
                "schema_version": "1.0",
                "top_card": {"found": True},
                "company": {"found": True},
                "recruiter_candidate_count": 0,
                "location_candidate_count": 1,
            }

        def locator(self, selector):
            return Locator(selector)

    collector = LinkedInHTMLDiagnosticsCollector(tmp_path, job_uid="job-1")
    collector.base_dir.mkdir(parents=True, exist_ok=True)
    (collector.base_dir / "manifest.json").write_text(
        json.dumps({"artifacts": {}}, ensure_ascii=False),
        encoding="utf-8",
    )

    collector.capture_company_recruiter_location(
        Page(),
        job_id="4453510505",
        reason="remote_scope_review",
    )

    manifest = json.loads(
        (collector.base_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifacts"][
        "company-recruiter-location-4453510505-remote_scope_review.json"
    ] == "company-recruiter-location-4453510505-remote_scope_review.json"
    assert manifest["artifacts"][
        "company-recruiter-location-4453510505-remote_scope_review-top-card.png"
    ] == "company-recruiter-location-4453510505-remote_scope_review-top-card.png"
    payload = json.loads(
        (
            collector.base_dir
            / "company-recruiter-location-4453510505-remote_scope_review.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["company"]["found"] is True
    assert "innerText" not in json.dumps(payload)


def test_linkedin_remote_scope_visual_evidence_only_for_regional_remote():
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_jobs_pipeline import (
        _capture_remote_scope_visual_evidence,
    )

    class VisualDiagnostics:
        def __init__(self):
            self.calls = []

        def capture_company_recruiter_location(self, page, *, job_id, reason):
            self.calls.append((page, job_id, reason))

    visual = VisualDiagnostics()
    page = object()
    warnings = []

    _capture_remote_scope_visual_evidence(
        visual,
        page,
        LinkedInVacancyRecord(
            linkedin_job_id="4453510505",
            title="Machine Learning Engineer - AI (Remote)",
            location="Asia-Pacífico",
            canonical_url="https://www.linkedin.com/jobs/view/4453510505",
            source_url="https://www.linkedin.com/jobs/search?location=South+Korea",
        ),
        warnings=warnings,
    )
    _capture_remote_scope_visual_evidence(
        visual,
        page,
        LinkedInVacancyRecord(
            linkedin_job_id="4394456615",
            title="AI Engineer",
            location="Seúl, Corea del Sur",
            canonical_url="https://www.linkedin.com/jobs/view/4394456615",
            source_url="https://www.linkedin.com/jobs/search?location=South+Korea",
        ),
        warnings=warnings,
    )

    assert visual.calls == [(page, "4453510505", "remote_scope_review")]
    assert warnings == ["remote_scope_visual_debug:4453510505:remote_scope_review"]

def test_linkedin_html_diagnostics_sanitize_text_and_sensitive_attributes():
    from features.web_scraping.infrastructure.linkedin_visual_diagnostics import (
        sanitize_linkedin_html,
    )

    sanitized, report = sanitize_linkedin_html(
        """
        <main class="results private-token" aria-label="Sensitive name">
          <div class="job-card" data-job-id="1234" data-private-token="secret"
               aria-selected="true">
            Candidate name, prompt injection, and cookie=secret
            <script>window.localStorage.secret = 'token'</script>
          </div>
        </main>
        """
    )

    assert "Candidate name" not in sanitized
    assert "cookie=secret" not in sanitized
    assert "localStorage" not in sanitized
    assert "aria-label" not in sanitized
    assert "data-private-token" not in sanitized
    assert 'data-job-id="1234"' in sanitized
    assert 'aria-selected="true"' in sanitized
    assert "private-token" in sanitized
    assert report["removed_text"] is True
    assert report["removed_attributes"] >= 2
    assert report["status"] == "ok"


def test_linkedin_html_diagnostics_are_bounded_and_manifest_is_local_only(
    tmp_path,
    monkeypatch,
):
    from features.web_scraping.infrastructure.linkedin_visual_diagnostics import (
        MAX_ACTIVATION_HTML_CAPTURES,
        visual_diagnostics_context,
    )

    monkeypatch.setenv("LINKEDIN_SEARCH_VISUAL_DIAGNOSTICS", "true")

    class FakeTracing:
        def start(self, **_kwargs):
            return None

        def start_chunk(self, **_kwargs):
            return None

        def stop_chunk(self, **kwargs):
            Path(kwargs["path"]).write_text("trace", encoding="utf-8")

        def stop(self, **_kwargs):
            return None

    class FakePage:
        context = SimpleNamespace(tracing=FakeTracing())

        def screenshot(self, *, path, full_page):
            assert full_page is True
            Path(path).write_text("image", encoding="utf-8")

        def locator(self, _selector):
            return SimpleNamespace(first=SimpleNamespace(count=lambda: 0))

        def evaluate(self, _script):
            return {"schema_version": "1.0"}

        def content(self):
            return "<main class='panel'><div aria-label='private'>secret text</div></main>"

    audit_dir = tmp_path / "audit"
    with visual_diagnostics_context(audit_dir) as collector:
        run = collector.start_run(FakePage(), query="AI Korea")
        assert run is not None
        run.capture_before(FakePage())
        run.capture_after(FakePage())
        for outcome in range(MAX_ACTIVATION_HTML_CAPTURES + 4):
            run.capture_activation(FakePage(), "row_activation_success")
        run.finalize()

    visual_dir = audit_dir / "visual-diagnostics"
    manifest = json.loads((visual_dir / "manifest.json").read_text(encoding="utf-8"))
    activation_files = list(visual_dir.glob("activation-*.html"))
    assert len(activation_files) == MAX_ACTIVATION_HTML_CAPTURES
    assert len(manifest["html"]["sanitization_reports"]) == 2 + MAX_ACTIVATION_HTML_CAPTURES
    assert manifest["artifacts"]["panel_before"] == "panel-before.html"
    assert manifest["artifacts"]["panel_after"] == "panel-after.html"
    assert all("/Users/" not in json.dumps(item) for item in manifest.values())
    assert "secret text" not in (visual_dir / "panel-before.html").read_text(encoding="utf-8")
    assert "aria-label" not in (visual_dir / "panel-before.html").read_text(encoding="utf-8")


def test_linkedin_public_snapshot_model_has_no_html_diagnostic_payload():
    from features.web_scraping.domain.linkedin_models import LinkedInAuditSnapshot

    payload = LinkedInAuditSnapshot.model_json_schema()
    assert "html" not in json.dumps(payload)


def test_linkedin_structural_panel_scoring_prefers_left_repeated_rows_over_navigation():
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        _choose_structural_results_panel,
    )

    navigation = {
        "container_index": 1,
        "x": 460,
        "width": 700,
        "row_candidate_count": 20,
        "row_repetition": 1,
        "visible_row_count": 20,
        "interactive_count": 40,
        "scrollHeight": 1200,
        "clientHeight": 500,
    }
    left_panel = {
        "container_index": 2,
        "x": 24,
        "width": 390,
        "row_candidate_count": 8,
        "row_repetition": 8,
        "visible_row_count": 8,
        "interactive_count": 8,
        "scrollHeight": 1800,
        "clientHeight": 500,
    }

    selected, _score = _choose_structural_results_panel(
        [navigation, left_panel], viewport_width=1280
    )
    assert selected["container_index"] == 2




def test_linkedin_structural_activation_filters_global_nav_and_detail_column():
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        _descriptor_is_safe_activation_row,
    )

    assert not _descriptor_is_safe_activation_row(
        {
            "x": 650,
            "y": 10,
            "width": 90,
            "height": 52,
            "href_fragment": "",
            "has_descendant_anchor": True,
        },
        viewport_width=1366,
    )
    assert not _descriptor_is_safe_activation_row(
        {
            "x": 594,
            "y": 267,
            "width": 613,
            "height": 30,
            "href_fragment": "4453249078",
            "has_descendant_anchor": True,
        },
        viewport_width=1366,
    )
    assert _descriptor_is_safe_activation_row(
        {
            "x": 24,
            "y": 180,
            "width": 390,
            "height": 92,
            "href_fragment": "",
            "role_button": True,
        },
        viewport_width=1366,
    )

def test_linkedin_structural_rows_include_divs_with_role_tabindex_or_cursor():
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        _dedupe_structural_rows,
    )

    rows = _dedupe_structural_rows(
        [
            {
                "container_index": 5,
                "x": 20,
                "y": 100,
                "width": 360,
                "height": 72,
                "row_candidate": True,
                "role_button": True,
                "href_fragment": "",
                "structural_path": ["div:0", "div:2"],
            }
        ]
    )
    assert len(rows) == 1
    assert rows[0]["role_button"] is True


def test_linkedin_structural_row_dedupe_prefers_smallest_compatible_ancestor():
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        _dedupe_structural_rows,
    )

    rows = _dedupe_structural_rows(
        [
            {
                "container_index": 1,
                "x": 20,
                "y": 100,
                "width": 390,
                "height": 90,
                "row_candidate": True,
                "href_fragment": "1234",
                "structural_path": ["div:0"],
            },
            {
                "container_index": 2,
                "x": 25,
                "y": 105,
                "width": 370,
                "height": 70,
                "row_candidate": True,
                "href_fragment": "1234",
                "structural_path": ["div:0", "div:0"],
            },
        ]
    )
    assert [row["container_index"] for row in rows] == [2]


def test_linkedin_row_signature_ignores_tiny_bbox_microchanges():
    import features.web_scraping.infrastructure.linkedin_query_navigation as navigation

    class Link:
        first = None

        def get_attribute(self, name):
            return "/jobs/view/1234" if name == "href" else None

    Link.first = Link()

    class Row:
        def __init__(self, box):
            self.box = box

        def locator(self, _selector):
            return Link()

        def get_attribute(self, name):
            return "1234" if name == "data-job-id" else None

        def bounding_box(self):
            return self.box

        def evaluate(self, _script):
            return ["div:0", "div:1"]

    first = navigation._row_signature(
        Row({"x": 20.1, "y": 100.2, "width": 360.1, "height": 72.2})
    )
    second = navigation._row_signature(
        Row({"x": 24.9, "y": 104.8, "width": 364.9, "height": 74.8})
    )
    assert navigation._row_identity_key(first) == navigation._row_identity_key(second)


def test_linkedin_structural_rows_without_ids_report_unresolved(monkeypatch):
    import features.web_scraping.infrastructure.linkedin_query_navigation as navigation

    rows = [("body *", index, (index, "", (), (20, index * 80, 360, 70), (f"div:{index}",))) for index in range(10)]
    monkeypatch.setattr(navigation, "_enumerate_visible_job_rows", lambda _page: rows)
    monkeypatch.setattr(navigation, "_resolve_row_locator", lambda _page, _signature: SimpleNamespace(click=lambda **_kwargs: None))
    monkeypatch.setattr(navigation, "_detail_identity_from_page", lambda _page: navigation.LinkedInDetailIdentity())
    monkeypatch.setattr(navigation, "_wait_for_changed_detail_identity", lambda *_args, **_kwargs: (navigation.LinkedInDetailIdentity(), False))
    monkeypatch.setattr(navigation, "_scroll_results_panel_for_rows", lambda _page: False)

    result = navigation.discover_job_rows_via_activation(
        object(), source_url="https://www.linkedin.com/jobs/search/?keywords=AI"
    )
    assert result.structural_rows_found == 10
    assert navigation.discovery_mode_for_sources(
        dom_contributed=False,
        row_contributed=False,
        structural_rows_found=True,
        row_job_ids_resolved=result.job_ids_resolved,
    ) == "structural_rows_found_but_unresolved"


def test_linkedin_row_activation_stops_after_two_moving_scrolls_without_new_ids(monkeypatch):
    import features.web_scraping.infrastructure.linkedin_query_navigation as navigation

    signature = (1, "", (), (20, 100, 360, 70), ("div:0",))
    monkeypatch.setattr(navigation, "_enumerate_visible_job_rows", lambda _page: [("body *", 1, signature)])
    monkeypatch.setattr(navigation, "_resolve_row_locator", lambda _page, _signature: SimpleNamespace(click=lambda **_kwargs: None))
    monkeypatch.setattr(navigation, "_detail_identity_from_page", lambda _page: navigation.LinkedInDetailIdentity())
    monkeypatch.setattr(navigation, "_wait_for_changed_detail_identity", lambda *_args, **_kwargs: (navigation.LinkedInDetailIdentity(), False))
    monkeypatch.setattr(navigation, "_scroll_results_panel_for_rows", lambda _page: True)

    result = navigation.discover_job_rows_via_activation(
        object(), source_url="https://www.linkedin.com/jobs/search/?keywords=AI"
    )
    assert result.scroll_count == 2
    assert result.stop_reason == "two_scrolls_without_new_job_ids"



def test_linkedin_row_ids_merge_one_dom_and_five_row_ids_into_five_unique_candidates():
    from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
    from features.web_scraping.infrastructure.linkedin_query_navigation import merge_row_activation_records

    source = "https://www.linkedin.com/jobs/search/?keywords=AI"
    dom = [LinkedInVacancyRecord(linkedin_job_id="1", canonical_url="https://www.linkedin.com/jobs/view/1", source_url=source)]
    rows = [
        LinkedInVacancyRecord(linkedin_job_id=str(index), canonical_url=f"https://www.linkedin.com/jobs/view/{index}", source_url=source, discovery_sources=["row_activation"])
        for index in range(1, 6)
    ]
    merged, _dom, _rows = merge_row_activation_records(dom, rows)
    assert len(merged) == 5


def test_linkedin_card_structure_debug_is_opt_in_and_sanitized(tmp_path, monkeypatch):
    import json

    monkeypatch.delenv("LINKEDIN_CARD_STRUCTURE_DEBUG", raising=False)

    from features.web_scraping.infrastructure.linkedin_card_structure_debug import (
        CARD_STRUCTURE_DEBUG_FILENAME,
        capture_visible_linkedin_card_structure_debug,
    )

    class Page:
        def evaluate(self, _script):
            return {
                "card_count": 1,
                "viewport": {"width": 1280, "height": 720},
                "cards": [
                    {
                        "card_index": 0,
                        "resolved_job_id": "4426794001",
                        "title_present": True,
                        "date_texts": ["Hace 10 horas", "secret<script>"],
                        "has_href": True,
                        "has_data_job_id": False,
                        "has_data_occludable_job_id": True,
                        "bbox": {"x": 120, "y": 590, "width": 480, "height": 120},
                        "ancestor_chain": [
                            {
                                "depth": 0,
                                "tag": "li",
                                "class_tokens": ["job-card", "token<script>"],
                                "has_href": True,
                                "has_data_job_id": False,
                                "has_data_occludable_job_id": True,
                                "bbox": {"x": 120, "y": 590, "width": 480, "height": 120},
                                "date_texts": ["Hace 10 horas"],
                                "child_count": 8,
                            }
                        ],
                    }
                ],
            }

    assert capture_visible_linkedin_card_structure_debug(Page(), tmp_path) is None

    monkeypatch.setenv("LINKEDIN_CARD_STRUCTURE_DEBUG", "true")
    path = capture_visible_linkedin_card_structure_debug(Page(), tmp_path)

    assert path == tmp_path / CARD_STRUCTURE_DEBUG_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["cards"][0]["resolved_job_id"] == "4426794001"
    assert payload["cards"][0]["date_texts"][0] == "Hace 10 horas"
    assert "<script>" not in path.read_text(encoding="utf-8")


def test_linkedin_html_diagnostics_manifest_includes_card_structure_debug(tmp_path, monkeypatch):
    from features.web_scraping.infrastructure.linkedin_card_structure_debug import (
        CARD_STRUCTURE_DEBUG_FILENAME,
    )
    from features.web_scraping.infrastructure.linkedin_visual_diagnostics import (
        LinkedInHTMLDiagnosticsCollector,
        _HTMLDiagnosticRun,
    )

    collector = LinkedInHTMLDiagnosticsCollector(tmp_path, job_uid="job-1")
    run = _HTMLDiagnosticRun(collector, query="AI")
    run.base_dir.mkdir(parents=True, exist_ok=True)
    (run.base_dir / CARD_STRUCTURE_DEBUG_FILENAME).write_text(
        '{"schema_version":"1.0","cards":[]}',
        encoding="utf-8",
    )

    run.finalize()

    assert CARD_STRUCTURE_DEBUG_FILENAME in (run.base_dir / "manifest.json").read_text(encoding="utf-8")


def test_linkedin_html_diagnostics_new_bundle_is_local_bounded_and_manifest_safe(tmp_path, monkeypatch):
    from features.web_scraping.infrastructure.linkedin_visual_diagnostics import (
        MAX_HTML_DIAGNOSTIC_ACTIVATIONS,
        visual_diagnostics_context,
    )
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        MAX_ROW_ACTIVATIONS_PER_QUERY,
    )

    monkeypatch.setenv("LINKEDIN_HTML_DIAGNOSTICS", "true")
    monkeypatch.delenv("LINKEDIN_CARD_STRUCTURE_DEBUG", raising=False)
    assert MAX_HTML_DIAGNOSTIC_ACTIVATIONS == 3
    assert MAX_ROW_ACTIVATIONS_PER_QUERY == 20

    class Page:
        def __init__(self):
            self.capture = 0

        def content(self):
            return (
                '<main class="row ' + "x" * 120 + '" aria-label="secret" '
                'onclick="leak()"><a href="/jobs/view/123?trk=secret#x" '
                'data-job-id="123">private text</a><script>token</script>'
                '<style>secret</style></main>'
            )

        def evaluate(self, _script):
            self.capture += 1
            row_index = 1 if self.capture == 1 else 2
            return {
                "row_count": row_index,
                "viewport": {"width": 1280, "height": 720},
                "rows": [{
                    "index": row_index,
                    "tag": "li",
                    "class_tokens": ["x" * 120],
                    "role": "listitem",
                    "tabindex": "0",
                    "aria_selected": "true",
                    "allowlisted_attrs": {"data-job-id": "123"},
                    "href": "/jobs/view/123",
                    "bounds": {"x": 1, "y": 2, "width": 300, "height": 80},
                    "vertical_band": 0,
                    "structural_path": ["li:0"],
                    "visible": True,
                    "row_candidate": True,
                }],
            }

        def screenshot(self, *, path, full_page):
            assert full_page is True
            Path(path).write_bytes(b"png")

    page = Page()
    root = tmp_path / "diagnostics"
    with visual_diagnostics_context(root, job_uid="job-123", diagnostics_root=root) as collector:
        run = collector.start_run(page, query="one query")
        assert run is not None
        run.capture_after_hydration(page)
        run.capture_before_scroll(page)
        run.capture_after_scroll(page, 1)
        run.capture_after_scroll(page, 2)
        run.capture_detail(page, "before_click")
        for _ in range(8):
            run.capture_detail(page, "after_click")
        run.capture_after(page)
        run.finalize()

    bundle = root / "job-123"
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert len(list(bundle.glob("results-panel-after-scroll-*.html"))) == 1
    assert len(list(bundle.glob("detail-after-click-*.html"))) == 2
    assert all((bundle / value).is_file() for value in manifest["artifacts"].values())
    assert all("secret" not in path.read_text(encoding="utf-8") for path in bundle.glob("*.html"))
    assert manifest["sanitization_report"]["scripts_removed"] > 0
    assert manifest["sanitization_report"]["query_strings_removed"] > 0
    assert "aria-label" not in (bundle / "search-after-hydration.html").read_text(encoding="utf-8")


def test_linkedin_html_diagnostics_sanitize_failure_writes_no_html(tmp_path, monkeypatch):
    from features.web_scraping.infrastructure.linkedin_visual_diagnostics import (
        visual_diagnostics_context,
    )

    monkeypatch.setenv("LINKEDIN_HTML_DIAGNOSTICS", "true")

    class Page:
        def content(self):
            raise RuntimeError("content unavailable")

    with visual_diagnostics_context(tmp_path, job_uid="job-failure", diagnostics_root=tmp_path) as collector:
        run = collector.start_run(Page(), query="one")
        run.capture_after_hydration(Page())
        run.finalize()

    bundle = tmp_path / "job-failure"
    assert not (bundle / "search-after-hydration.html").exists()
    assert (bundle / "manifest.json").exists()


def test_linkedin_html_diagnostics_gitignored_without_creating_test_file():
    import subprocess

    result = subprocess.run(
        ["/usr/bin/git", "check-ignore", "data/private/linkedin/diagnostics/test/file.html"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
