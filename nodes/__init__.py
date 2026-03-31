"""
Paquete nodes — factories para los nodos del grafo LangGraph.

Uso:
    from nodes import make_math_node, make_analysis_node, make_code_node, make_web_scraping_node
"""
from nodes.math_node import make_math_node
from nodes.analysis_node import make_analysis_node
from nodes.code_node import make_code_node
from nodes.web_scraping_node import make_web_scraping_node

__all__ = [
    "make_math_node",
    "make_analysis_node",
    "make_code_node",
    "make_web_scraping_node",
]
