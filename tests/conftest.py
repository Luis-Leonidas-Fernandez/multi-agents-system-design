import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Provide a dummy API key so application/composition/graph.py can instantiate agents at module scope
# without a real .env file. Tests that need LLM calls mock get_llm() directly.
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-unit-tests")


@pytest.fixture(autouse=True)
def reset_supervisor_chain():
    """
    Resetea el supervisor chain cacheado antes de cada test.

    application/composition/graph.py cachea _supervisor_chain a nivel de módulo para evitar
    reinstanciar el LLM en cada turno (mejora de performance). Los tests
    que parchean application.composition.graph.get_llm necesitan que el cache esté limpio para que
    el patch tome efecto en la próxima construcción del chain.
    """
    from application.composition import graph
    graph._supervisor_chain = None
    yield
    graph._supervisor_chain = None
