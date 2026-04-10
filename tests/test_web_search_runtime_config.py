import json


def test_get_web_search_runtime_config_uses_explicit_json_path(tmp_path, monkeypatch):
    from application.helpers.config_flow_helpers import get_web_search_runtime_config

    config_path = tmp_path / "web-search.json"
    config_path.write_text(json.dumps({"selected_provider": "duckduckgo", "provider_configured": "tavily"}), encoding="utf-8")
    monkeypatch.setenv("WEB_SEARCH_CONFIG", str(config_path))
    get_web_search_runtime_config.cache_clear()

    cfg = get_web_search_runtime_config()

    assert cfg.selected_provider == "duckduckgo"
    assert cfg.provider_configured == "tavily"


def test_web_search_registry_uses_runtime_config_file(tmp_path, monkeypatch):
    from application.helpers.config_flow_helpers import get_web_search_runtime_config
    from application.services.web_search_registry import resolve_web_search_provider_name

    config_path = tmp_path / "web-search.json"
    config_path.write_text(json.dumps({"provider_configured": "duckduckgo"}), encoding="utf-8")
    monkeypatch.setenv("WEB_SEARCH_CONFIG", str(config_path))
    monkeypatch.delenv("WEB_SEARCH_PROVIDER", raising=False)
    get_web_search_runtime_config.cache_clear()

    assert resolve_web_search_provider_name() == "duckduckgo"
