"""Tools especializadas del sistema multi-agentes."""

from tools.math_tools import calculate
from tools.data_tools import analyze_data
from tools.code_tools import write_code
from tools.crypto_price import get_crypto_price
from tools.web_tools import (
    extract_price_from_text,
    search_web,
    scrape_website_simple,
    scrape_website_dynamic,
    scrape_website_with_json_capture,
    web_fetch,
)

__all__ = [
    "calculate",
    "analyze_data",
    "write_code",
    "get_crypto_price",
    "extract_price_from_text",
    "search_web",
    "scrape_website_simple",
    "scrape_website_dynamic",
    "scrape_website_with_json_capture",
    "web_fetch",
]
