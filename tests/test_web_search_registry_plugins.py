import json


def test_web_search_registry_discovers_plugin_providers(tmp_path, monkeypatch):
    from features.web_scraping.infrastructure.web_search_registry import list_web_search_provider_specs

    plugin_dir = tmp_path / "web_search_provider_plugins"
    plugin_dir.mkdir()
    (plugin_dir / "custom.json").write_text(
        json.dumps({
            "providers": [
                {
                    "name": "custom-keyless",
                    "label": "Custom Keyless",
                    "kind": "tavily",
                    "env_vars": [],
                    "auto_detect_order": 5,
                    "requires_credential": False,
                }
            ]
        }),
        encoding="utf-8",
    )

    monkeypatch.setenv("WEB_SEARCH_PROVIDER_PLUGIN_DIR", str(plugin_dir))
    list_web_search_provider_specs.cache_clear()

    specs = list_web_search_provider_specs()

    assert any(spec.name == "custom-keyless" for spec in specs)


def test_web_search_registry_deduplicates_plugin_names(tmp_path, monkeypatch):
    from features.web_scraping.infrastructure.web_search_registry import list_web_search_provider_specs

    plugin_dir = tmp_path / "web_search_provider_plugins"
    plugin_dir.mkdir()
    (plugin_dir / "first.json").write_text(
        json.dumps({"providers": [{"name": "tavily", "label": "Override", "kind": "tavily", "env_vars": [], "auto_detect_order": 1, "requires_credential": False}]}),
        encoding="utf-8",
    )

    monkeypatch.setenv("WEB_SEARCH_PROVIDER_PLUGIN_DIR", str(plugin_dir))
    list_web_search_provider_specs.cache_clear()

    specs = [spec for spec in list_web_search_provider_specs() if spec.name == "tavily"]

    assert len(specs) == 1
