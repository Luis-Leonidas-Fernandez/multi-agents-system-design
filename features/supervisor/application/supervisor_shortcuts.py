"""Atajos de routing del supervisor."""

from core.helpers.message_flow_helpers import is_btc_price_query, is_web_information_query


def should_route_to_web_scraping(last_message: str) -> bool:
    return is_btc_price_query(last_message) or is_web_information_query(last_message)


__all__ = ["should_route_to_web_scraping"]
