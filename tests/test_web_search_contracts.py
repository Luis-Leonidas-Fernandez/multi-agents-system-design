import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_weekly_security_query_returns_four_distinct_paragraphs():
    from features.web_scraping.application.fetch_dispatch import _run_generic_web_search_fetch

    search_text = (
        'Web search results for query: "dame las ultimas noticias sobre seguridad de estados unidos esta semana"\n\n'
        '1. [Congress and DHS funding](https://www.reuters.com/world/us/congress-dhs-funding-2026-04-08/)\n'
        '   Congress debates funding for the Department of Homeland Security\n\n'
        '2. [FBI cyber warning](https://www.washingtonpost.com/politics/2026/04/08/fbi-cyber-warning/)\n'
        '   FBI warns about a new cyber threat targeting critical infrastructure\n\n'
        '3. [Pentagon readiness](https://www.defense.gov/news/2026-04-09/pentagon-readiness/)\n'
        '   The Pentagon announces new readiness measures this week\n\n'
        '4. [Border security update](https://www.nbcnews.com/politics/border-security-2026-04-09/)\n'
        '   Border security operations are expanded along the southern border\n\n'
        'Sources:\n'
        '- [Congress and DHS funding](https://www.reuters.com/world/us/congress-dhs-funding-2026-04-08/)\n'
        '- [FBI cyber warning](https://www.washingtonpost.com/politics/2026/04/08/fbi-cyber-warning/)\n'
        '- [Pentagon readiness](https://www.defense.gov/news/2026-04-09/pentagon-readiness/)\n'
        '- [Border security update](https://www.nbcnews.com/politics/border-security-2026-04-09/)'
    )

    with (
        patch("features.web_scraping.infrastructure.search_tools.search_web.func", return_value=search_text),
        patch("features.web_scraping.infrastructure.scraping_tools.fetch_web_page", AsyncMock(side_effect=[
            "URL: https://www.reuters.com/world/us/congress-dhs-funding-2026-04-08/\n\nCongress and DHS funding remain the dominant security issue this week for the United States. Lawmakers are split over immigration enforcement and homeland security resources.\n\nSources:\n- [Reuters](https://www.reuters.com/world/us/congress-dhs-funding-2026-04-08/)",
            "URL: https://www.washingtonpost.com/politics/2026/04/08/fbi-cyber-warning/\n\nThe FBI issued a new warning about cyber threats aimed at critical infrastructure. Officials asked private companies to harden defenses immediately.\n\nSources:\n- [Washington Post](https://www.washingtonpost.com/politics/2026/04/08/fbi-cyber-warning/)",
            "URL: https://www.defense.gov/news/2026-04-09/pentagon-readiness/\n\nThe Pentagon rolled out readiness measures focused on deterrence and response capability. The move reflects a broader security posture update this week.\n\nSources:\n- [Defense.gov](https://www.defense.gov/news/2026-04-09/pentagon-readiness/)",
            "URL: https://www.nbcnews.com/politics/border-security-2026-04-09/\n\nBorder security operations were expanded this week, keeping the topic firmly on U.S. security policy. The announcement added a fourth distinct news item without repeating the others.\n\nSources:\n- [NBC News](https://www.nbcnews.com/politics/border-security-2026-04-09/)",
        ])),
    ):
        result = await _run_generic_web_search_fetch("dame las ultimas noticias sobre seguridad de estados unidos esta semana")

    assert result is not None
    summary = result["summary"]
    body = summary.split("\n\nSources:", 1)[0]
    paragraphs = [part for part in body.split("\n\n") if part.strip()]

    assert len(paragraphs) == 4
    assert len(set(paragraphs)) == 4
    assert "security" in body.lower() or "seguridad" in body.lower()
    assert "Sources:" in summary
