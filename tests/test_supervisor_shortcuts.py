"""Tests para atajos de routing del supervisor."""


def test_should_route_to_web_scraping_para_btc_y_web_info():
    from features.supervisor.api import should_route_to_web_scraping

    assert should_route_to_web_scraping("dame el precio actual de bitcoin") is True
    assert should_route_to_web_scraping("buscame esto en internet") is True
    assert should_route_to_web_scraping("dame noticias de economia argentina") is True
    assert should_route_to_web_scraping("explicame algebra lineal") is False
