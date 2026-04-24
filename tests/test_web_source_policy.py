"""Tests de política y ranking de fuentes web por país."""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.parametrize(
    ("query", "expected_group"),
    [
        ("security news in the united states", "united_states"),
        ("seguridad en eeuu", "united_states"),
        ("seguridad en brazil", "brazil"),
        ("seguridad en brasil", "brazil"),
        ("seguridad en uruguay", "uruguay"),
        ("seguridad en colombia", "colombia"),
        ("seguridad en paraguay", "paraguay"),
        ("seguridad en bolivia", "bolivia"),
        ("seguridad en españa", "spain"),
        ("seguridad en ireland", "ireland"),
        ("seguridad en islandia", "iceland"),
        ("seguridad en eslovenia", "slovenia"),
        ("seguridad en italia", "italy"),
        ("seguridad en ucrania", "ukraine"),
        ("seguridad en rusia", "russia"),
        ("seguridad en china", "china"),
        ("seguridad en argentina", "argentina"),
        ("seguridad en chile", "chile"),
        ("seguridad en méxico", "mexico"),
        ("seguridad en japon", "japan"),
        ("seguridad en corea del sur", "south_korea"),
        ("seguridad en corea del norte", "north_korea"),
        ("seguridad en nueva zelanda", "new_zealand"),
    ],
)
def test_detect_query_source_group_recognizes_multiple_countries(query: str, expected_group: str) -> None:
    from application.policies.web_source_policy import detect_query_source_group

    assert detect_query_source_group(query) == expected_group


@pytest.mark.parametrize(
    ("country_group", "preferred_url", "local_url", "trusted_url"),
    [
        (
            "united_states",
            "https://www.nytimes.com/2026/04/10/us/politics/security-update.html",
            "https://www.example.us/security-update.html",
            "https://www.reuters.com/world/us/security-2026-04-10/",
        ),
        (
            "brazil",
            "https://g1.globo.com/politica/seguranca-no-brasil/",
            "https://www.diarioregional.br/politica/seguranca-no-brasil",
            "https://www.reuters.com/world/americas/brazil-security-2026-04-10/",
        ),
        (
            "argentina",
            "https://www.clarin.com/politica/seguridad-en-buenos-aires",
            "https://www.diarioregional.com.ar/politica/seguridad-en-buenos-aires",
            "https://www.reuters.com/world/americas/argentina-security-2026-04-10/",
        ),
        (
            "chile",
            "https://www.latercera.com/nacional/seguridad-en-santiago",
            "https://www.diarioregional.cl/politica/seguridad-en-santiago",
            "https://www.reuters.com/world/americas/chile-security-2026-04-10/",
        ),
        (
            "mexico",
            "https://www.eluniversal.com.mx/nacion/seguridad-en-mexico",
            "https://www.diarioregional.mx/politica/seguridad-en-mexico",
            "https://www.reuters.com/world/americas/mexico-security-2026-04-10/",
        ),
    ],
)
def test_score_domain_boost_prefers_country_sources_over_global_fallback(
    country_group: str,
    preferred_url: str,
    local_url: str,
    trusted_url: str,
) -> None:
    from application.policies.web_source_policy import score_domain_boost

    assert score_domain_boost(country_group, preferred_url) > score_domain_boost(country_group, local_url)
    assert score_domain_boost(country_group, local_url) > score_domain_boost(country_group, trusted_url)


def test_curated_domains_outrank_suffix_only_matches_for_new_country() -> None:
    from application.policies.web_source_policy import get_source_domain_priority, score_domain_boost

    preferred_url = "https://www.nytimes.com/2026/04/10/us/politics/security-update.html"
    suffix_only_url = "https://www.diarioregional.us/politica/security-update"
    trusted_url = "https://www.reuters.com/world/us/security-2026-04-10/"

    assert get_source_domain_priority("united_states", preferred_url) == 0
    assert get_source_domain_priority("united_states", suffix_only_url) == 1
    assert get_source_domain_priority("united_states", trusted_url) == 2
    assert score_domain_boost("united_states", preferred_url) > score_domain_boost("united_states", suffix_only_url)
    assert score_domain_boost("united_states", suffix_only_url) > score_domain_boost("united_states", trusted_url)


def test_suffix_only_matches_work_for_new_world_countries() -> None:
    from application.policies.web_source_policy import get_source_domain_priority, score_domain_boost

    suffix_cases = [
        ("brazil", "https://www.diarioregional.br/politica/seguranca-no-brasil"),
        ("uruguay", "https://www.diarioregional.uy/politica/seguridad-en-uruguay"),
        ("colombia", "https://www.diarioregional.co/politica/seguridad-en-colombia"),
        ("paraguay", "https://www.diarioregional.py/politica/seguridad-en-paraguay"),
        ("bolivia", "https://www.diarioregional.bo/politica/seguridad-en-bolivia"),
        ("spain", "https://www.diarioregional.es/politica/seguridad-en-espana"),
        ("ireland", "https://www.diarioregional.ie/politica/security-in-ireland"),
        ("iceland", "https://www.diarioregional.is/politica/security-in-iceland"),
        ("slovenia", "https://www.diarioregional.si/politica/security-in-slovenia"),
        ("italy", "https://www.diarioregional.it/politica/security-in-italy"),
        ("ukraine", "https://www.diarioregional.ua/politica/security-in-ukraine"),
        ("russia", "https://www.diarioregional.ru/politica/security-in-russia"),
        ("china", "https://www.diarioregional.cn/politica/security-in-china"),
        ("north_korea", "https://www.diarioregional.kp/politica/security-in-north-korea"),
        ("new_zealand", "https://www.diarioregional.nz/politica/security-in-new-zealand"),
    ]

    trusted_url = "https://www.reuters.com/world/2026-04-10/security-roundup/"
    for country_group, suffix_url in suffix_cases:
        assert get_source_domain_priority(country_group, suffix_url) == 1
        assert score_domain_boost(country_group, suffix_url) > score_domain_boost(country_group, trusted_url)


def test_detect_recent_query_horizon_supports_month_queries() -> None:
    from application.policies.web_source_policy import detect_recent_query_horizon

    assert detect_recent_query_horizon("noticias de seguridad en italia de este mes") == "month"
    assert detect_recent_query_horizon("security news in italy last month") == "month"


def test_topic_landings_are_treated_as_hubs_for_recent_news() -> None:
    from features.web_scraping.application.flow import _is_hub_like_candidate, _is_invalid_news_candidate

    candidate = {
        "title": "Ultima Ora, notizie in tempo reale - Il Messaggero",
        "url": "https://www.ilmessaggero.it/t/ultima-ora",
        "snippet": "Aggiornamenti in tempo reale",
    }
    tangential_candidate = {
        "title": "SICUREZZA INFORMATICA - Il Messaggero",
        "url": "https://www.ilmessaggero.it/t/sicurezza-informatica",
        "snippet": "Notizie, foto e video su sicurezza informatica",
    }

    assert _is_hub_like_candidate(candidate) is True
    assert _is_invalid_news_candidate(tangential_candidate, "dame las ultimas noticias sobre seguridad en italia de esta semana") is True


@pytest.mark.asyncio
async def test_country_press_discovery_is_cached_per_country() -> None:
    from features.web_scraping import api as web_scraping_flow

    web_scraping_flow._COUNTRY_PRESS_CACHE.clear()
    lookup_calls: list[dict[str, object]] = []

    async def fake_fetch(url: str, prompt: str, use_dynamic: bool = True) -> str:
        return (
            "URL: https://periodicos.com.ar/periodicos/asia/japon/\n\n"
            "1. [NHK](https://www.nhk.or.jp/)\n"
            "2. [Japan Times](https://www.japantimes.co.jp/)\n"
        )

    def fake_invoke(**payload: object) -> str:
        lookup_calls.append(payload)
        return (
            "Web search results for query: \"periodicos japon noticias diarios\"\n\n"
            "1. [NHK](https://www.nhk.or.jp/)\n"
            "2. [Japan Times](https://www.japantimes.co.jp/)\n"
            "Sources:\n"
            "- [NHK](https://www.nhk.or.jp/)\n"
            "- [Japan Times](https://www.japantimes.co.jp/)\n"
        )

    with (
        patch("features.web_scraping.infrastructure.search_tools.search_web.func", side_effect=fake_invoke),
        patch("features.web_scraping.infrastructure.scraping_tools.fetch_web_page", new=AsyncMock(side_effect=fake_fetch)),
    ):
        first = await web_scraping_flow._discover_country_press_sources(
            "seguridad en japon",
            "japan",
            ["japon"],
        )
        second = await web_scraping_flow._discover_country_press_sources(
            "seguridad en japon otra vez",
            "japan",
            ["japon"],
        )

    assert first == second
    assert len(lookup_calls) == 1


@pytest.mark.asyncio
async def test_country_recent_news_strategy_prioritizes_section_hits() -> None:
    from features.web_scraping.api import CountryRecentNewsStrategy
    from features.web_scraping.infrastructure.runtime import WebFetchRuntime, WebSearchRuntime

    fetch_runtime = WebFetchRuntime()
    fetch_runtime.fetch = AsyncMock(return_value=type("FetchResponse", (), {
        "provider_name": "default",
        "url": "https://www.huffingtonpost.it/news/cronaca/",
        "content": (
            '- "Quattordiceenne accoltellato nei bagni della scuola a Napoli, due minori denunciati."\n'
            '- "Esplosione al parco degli Acquedotti, si segue la pista anarchica: due morti, preparavano un ordigno."'
        ),
        "fetch_kind": "static",
        "status": "ok",
    })())
    search_runtime = WebSearchRuntime()

    strategy = CountryRecentNewsStrategy(search_runtime=search_runtime, fetch_runtime=fetch_runtime)

    with (
        patch("application.services.press_discovery.discover_country_press_sources", new=AsyncMock(return_value=(["huffingtonpost.it"], ["HuffPost Italia"]))),
        patch("features.web_scraping.application.flow._country_press_source_cache_get", return_value=[{"title": "HuffPost Italia", "url": "https://www.huffingtonpost.it/"}]),
        patch("features.web_scraping.application.flow._country_press_strategy_cache_get", return_value="bootstrap"),
        patch("features.web_scraping.application.flow._is_press_source_relevant_for_query", return_value=True),
        patch("features.web_scraping.application.flow._build_country_press_section_targets", return_value=[("https://www.huffingtonpost.it/news/cronaca/", "cronaca")]),
        patch("features.web_scraping.application.flow._detect_news_topic", return_value=None),
        patch("features.web_scraping.application.flow.get_group_language", return_value="es"),
    ):
        result = await strategy.execute("dame las ultimas noticias sobre seguridad en italia de esta semana", {})

    assert result is not None
    assert result.get("pre_synthesized") is True
    assert "Quattordiceenne accoltellato" in str(result.get("summary") or "")
    assert fetch_runtime.fetch.await_count == 1


@pytest.mark.asyncio
async def test_country_press_discovery_falls_back_to_homepage_when_lookup_fails() -> None:
    from features.web_scraping import api as web_scraping_flow

    web_scraping_flow._COUNTRY_PRESS_CACHE.clear()

    def fake_invoke(**payload: object) -> str:
        return "Error en búsqueda: provider down"

    async def fake_fetch(url: str, prompt: str, use_dynamic: bool = True) -> str:
        if url == "https://periodicos.com.ar/":
            return (
                "URL: https://periodicos.com.ar/\n\n"
                "1. [ANSA](https://www.ansa.it/)\n"
                "2. [La Repubblica](https://www.repubblica.it/)\n"
                "3. [Corriere della Sera](https://www.corriere.it/)\n"
            )
        raise AssertionError(f"Unexpected URL: {url}")

    with (
        patch("features.web_scraping.infrastructure.search_tools.search_web.func", side_effect=fake_invoke),
        patch("features.web_scraping.infrastructure.scraping_tools.fetch_web_page", new=AsyncMock(side_effect=fake_fetch)),
    ):
        domains, titles = await web_scraping_flow._discover_country_press_sources(
            "dame las ultimas noticias sobre seguridad en italia de esta semana",
            "italy",
            ["italia"],
        )

    assert "ansa.it" in domains
    assert "repubblica.it" in domains
    assert "ANSA" in titles
    assert "La Repubblica" in titles


@pytest.mark.asyncio
async def test_country_press_search_candidates_falls_back_to_source_homepage_when_search_fails() -> None:
    from features.web_scraping.api import _run_country_press_search_candidates

    async def fake_discover(*args, **kwargs):
        return (["ansa.it"], ["ANSA"])

    def fake_invoke(**payload: object) -> str:
        return "Error en búsqueda: provider down"

    async def fake_fetch(url: str, prompt: str, use_dynamic: bool = True) -> str:
        if url == "https://www.ansa.it/":
            return (
                "URL: https://www.ansa.it/\n\n"
                "ANSA reporta novedades de seguridad en Italia esta semana\n"
                "Roma refuerza controles en zonas críticas\n\n"
                "Sources:\n- [ANSA](https://www.ansa.it/)"
            )
        raise AssertionError(f"Unexpected URL: {url}")

    with (
        patch("features.web_scraping.application.flow._discover_country_press_sources", new=AsyncMock(side_effect=fake_discover)),
        patch("features.web_scraping.infrastructure.search_tools.search_web.func", side_effect=fake_invoke),
        patch("features.web_scraping.infrastructure.scraping_tools.fetch_web_page", new=AsyncMock(side_effect=fake_fetch)),
    ):
        candidates, search_text = await _run_country_press_search_candidates(
            "dame las ultimas noticias sobre seguridad en italia de esta semana",
            14,
            ["italia", "seguridad"],
            "italy",
            ["italia"],
            query_horizon="week",
        )

    assert candidates
    assert "ANSA reporta novedades de seguridad en Italia esta semana" in search_text or search_text == "Error en búsqueda: provider down"


@pytest.mark.asyncio
async def test_country_press_search_candidates_queries_each_diary() -> None:
    from features.web_scraping.api import _run_country_press_search_candidates

    call_payloads: list[dict[str, object]] = []

    async def fake_discover(*args, **kwargs):
        return (["ansa.it", "repubblica.it"], ["ANSA", "La Repubblica"])

    def fake_invoke(**payload: object) -> str:
        call_payloads.append(payload)
        domain = (payload.get("allowed_domains") or ["unknown"])[0]
        query = str(payload.get("query") or "")
        assert f"site:{domain}" in query
        if domain == "ansa.it":
            return (
                "Web search results for query: \"dame las ultimas noticias sobre seguridad en italia de esta semana ANSA\"\n\n"
                "1. [ANSA seguridad Italia](https://www.ansa.it/italia/notizie/2026/04/10/seguridad.html)\n"
                "   ANSA reporta novedades de seguridad en Italia\n\n"
                "Sources:\n- [ANSA seguridad Italia](https://www.ansa.it/italia/notizie/2026/04/10/seguridad.html)"
            )
        return (
            "Web search results for query: \"dame las ultimas noticias sobre seguridad en italia de esta semana La Repubblica\"\n\n"
            "1. [Repubblica seguridad Italia](https://www.repubblica.it/cronaca/2026/04/10/seguridad.html)\n"
            "   Repubblica reporta novedades de seguridad en Italia\n\n"
            "Sources:\n- [Repubblica seguridad Italia](https://www.repubblica.it/cronaca/2026/04/10/seguridad.html)"
        )

    with (
        patch("features.web_scraping.application.flow._discover_country_press_sources", new=AsyncMock(side_effect=fake_discover)),
        patch("features.web_scraping.infrastructure.search_tools.search_web.func", side_effect=fake_invoke),
    ):
        candidates, search_text = await _run_country_press_search_candidates(
            "dame las ultimas noticias sobre seguridad en italia de esta semana",
            14,
            ["italia", "seguridad"],
            "italy",
            ["italia"],
            query_horizon="week",
        )

    assert len(call_payloads) == 2
    assert call_payloads[0]["allowed_domains"] == ["ansa.it"]
    assert call_payloads[1]["allowed_domains"] == ["repubblica.it"]
    assert candidates
    assert "ANSA seguridad Italia" in search_text
    assert "Repubblica seguridad Italia" in search_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "local_url", "trusted_url"),
    [
        (
            "seguridad en argentina",
            "https://www.diarioregional.com.ar/politica/seguridad-en-buenos-aires",
            "https://www.reuters.com/world/americas/argentina-security-2026-04-10/",
        ),
        (
            "seguridad en chile",
            "https://www.diarioregional.cl/politica/seguridad-en-santiago",
            "https://www.reuters.com/world/americas/chile-security-2026-04-10/",
        ),
        (
            "seguridad en méxico",
            "https://www.diarioregional.mx/politica/seguridad-en-mexico",
            "https://www.reuters.com/world/americas/mexico-security-2026-04-10/",
        ),
    ],
)
async def test_run_generic_web_search_fetch_prefers_local_country_sources_before_global(query: str, local_url: str, trusted_url: str) -> None:
    from features.web_scraping.api import _run_generic_web_search_fetch

    lookup_text = (
        f'Web search results for query: "periodicos {query}"\n\n'
        f'1. [Local outlet primary]({local_url})\n'
        "   Directorio del diario local del país solicitado.\n\n"
        "2. [Local outlet secondary](https://www.mediolocal.example/)\n"
        "   Otro medio local del mismo país.\n\n"
        "Sources:\n"
        f"- [Local outlet primary]({local_url})\n"
        "- [Local outlet secondary](https://www.mediolocal.example/)\n"
    )
    search_text = (
        f'Web search results for query: "{query}"\n\n'
        f'1. [Reuters coverage]({trusted_url})\n'
        "   Reuters covers the security situation with international context and official statements.\n\n"
        f'2. [Local outlet]({local_url})\n'
        "   La cobertura local describe un operativo de seguridad y detalla la reacción oficial en el país.\n"
        "   El artículo agrega contexto nacional, cifras concretas y el impacto político inmediato.\n"
    )
    search_text_alt = (
        f'Web search results for query: "{query} últimas noticias recientes"\n\n'
        f'1. [Local outlet alt]({local_url})\n'
        "   Cobertura local ampliada con contexto nacional y reacciones oficiales.\n\n"
        f'2. [Reuters coverage alt]({trusted_url})\n'
        "   Reuters provides broader context on the situation.\n"
    )

    fetched_order: list[str] = []

    async def fake_fetch(url: str, prompt: str, use_dynamic: bool = True) -> str:
        fetched_order.append(url)
        if url == local_url:
            return (
                f"URL: {url}\n\n"
                "La cobertura local describe un operativo de seguridad y detalla la reacción oficial en el país.\n"
                "El artículo agrega contexto nacional, cifras concretas y el impacto político inmediato.\n"
                "Las autoridades locales confirmaron nuevas medidas de control.\n"
            )
        return (
            f"URL: {url}\n\n"
            "Reuters covers the security situation with international context and official statements.\n"
            "Officials highlighted regional implications and broader policy responses.\n"
            "The report remains relevant but is not the local primary source.\n"
        )

    with (
        patch("features.web_scraping.infrastructure.search_tools.search_web.func", side_effect=[lookup_text, search_text, search_text_alt]),
        patch("features.web_scraping.infrastructure.scraping_tools.fetch_web_page", new=AsyncMock(side_effect=fake_fetch)),
    ):
        result = await _run_generic_web_search_fetch(query)

    assert result is not None
    assert fetched_order[0] == local_url
    assert local_url in result["summary"]
