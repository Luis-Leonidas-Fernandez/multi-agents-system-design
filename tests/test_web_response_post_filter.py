from application.services.web_response_post_filter import apply_web_response_post_filter


def test_security_post_filter_removes_irrelevant_body_line_and_trims_sources():
    summary = (
        '"Vetri delle auto frantumati a Villa De Sanctis, record di vetture vandalizzate in sei mesi: è allarme sicurezza."\n'
        '"Violenta lite tra due rider in centro a Milano, coltellate e un arresto per tentato omicidio."\n'
        '"Spariti gli alberi appena piantati, il mistero della Pineta Sacchetti."\n'
        "\n"
        "Sources:\n"
        "- [Roma](https://www.ilmessaggero.it/roma/)\n"
        "- [Milano](https://www.ilfattoquotidiano.it/)\n"
        "- [Pineta](https://example.com/pineta)"
    )
    sources = [
        {"title": "Roma", "url": "https://www.ilmessaggero.it/roma/"},
        {"title": "Milano", "url": "https://www.ilfattoquotidiano.it/"},
        {"title": "Pineta", "url": "https://example.com/pineta"},
    ]

    filtered_summary, filtered_sources = apply_web_response_post_filter(
        summary,
        "dame las ultimas noticias sobre seguridad en italia en esta semana",
        sources,
    )

    assert "Pineta Sacchetti" not in filtered_summary
    assert "allarme sicurezza" in filtered_summary
    assert "tentato omicidio" in filtered_summary
    assert filtered_sources is not None
    assert len(filtered_sources) == 2


def test_economy_post_filter_respects_query_theme():
    summary = (
        "Italia mejora su crecimiento esta semana con nuevas inversiones.\n"
        "Fuertes movimientos en el mercado laboral y presión sobre los salarios.\n"
        "El parlamento discute una crisis de gabinete sin impacto económico directo.\n\n"
        "Sources:\n"
        "- [Economia](https://example.com/economia)\n"
        "- [Trabajo](https://example.com/trabajo)\n"
        "- [Politica](https://example.com/politica)"
    )

    filtered_summary, filtered_sources = apply_web_response_post_filter(
        summary,
        "dame las ultimas noticias sobre economia en italia esta semana",
        [
            {"title": "Economia", "url": "https://example.com/economia"},
            {"title": "Trabajo", "url": "https://example.com/trabajo"},
            {"title": "Politica", "url": "https://example.com/politica"},
        ],
    )

    assert "crecimiento" in filtered_summary
    assert "mercado laboral" in filtered_summary
    assert "crisis de gabinete" not in filtered_summary
    assert filtered_sources is not None
    assert len(filtered_sources) == 2


def test_post_filter_leaves_unknown_topics_untouched():
    summary = (
        "Italia mejora su crecimiento esta semana.\n"
        "Fuertes movimientos en el mercado laboral.\n\n"
        "Sources:\n- [Economia](https://example.com/economia)"
    )

    filtered_summary, filtered_sources = apply_web_response_post_filter(
        summary,
        "que novedades hay en italia esta semana",
        [{"title": "Economia", "url": "https://example.com/economia"}],
    )

    assert filtered_summary == summary
    assert filtered_sources == [{"title": "Economia", "url": "https://example.com/economia"}]
