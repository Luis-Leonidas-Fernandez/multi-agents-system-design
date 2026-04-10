"""
Definición de agentes especializados para el sistema multi-agentes

Este módulo usa langgraph.prebuilt.create_react_agent para crear agentes
que pueden usar herramientas de forma autónoma siguiendo el patrón ReAct.

Nota de migración (LangGraph 1.x): el warning sugiere mover a
`langchain.agents.create_react_agent`, pero esa función no existe con la misma
firma en la versión actual de langchain. Se suprime el warning hasta que haya
un reemplazo drop-in verificado.
"""
import warnings
from typing import Any, Sequence

from langgraph.prebuilt import create_react_agent

# LangGraph 1.x emite LangGraphDeprecatedSinceV10 en cada llamada a create_react_agent
# sugiriendo migrar a langchain.agents.create_react_agent, que no existe aún con la
# misma firma. Se suprime hasta que haya reemplazo drop-in verificado.
warnings.filterwarnings("ignore", message=".*create_react_agent.*", category=Warning)

from application.helpers.config_flow_helpers import get_llm
from application.services.agent_registry import get_agent_temperature
from application.services.prompt_assembly import build_agent_prompt_assembly
from application.services.tool_registry import get_tools_for_agent


def _build_specialized_agent(agent_name: str, tools: Sequence[Any]):
    """Construye un agente ReAct especializado con prompt y temperatura centralizados."""
    llm = get_llm(temperature=get_agent_temperature(agent_name))
    assembly = build_agent_prompt_assembly(agent_name)
    return create_react_agent(
        model=llm,
        tools=list(tools),
        prompt=assembly.system_prompt,
        name=agent_name,
    )


# ==================== AGENTES ESPECIALIZADOS ====================
# Usando langgraph.prebuilt.create_react_agent para crear agentes
# que siguen el patrón ReAct (Reasoning + Acting)


def create_math_agent():
    """
    Crea un agente especializado en matemáticas.

    Usa create_react_agent que devuelve un CompiledStateGraph
    que puede ser invocado directamente con mensajes.
    """
    return _build_specialized_agent("math_agent", get_tools_for_agent("math_agent"))


def create_analysis_agent():
    """
    Crea un agente especializado en análisis de datos.

    Este agente puede analizar descripciones de datos y generar insights.
    """
    return _build_specialized_agent("analysis_agent", get_tools_for_agent("analysis_agent"))


def create_code_agent():
    """
    Crea un agente especializado en programación.

    Este agente puede escribir código en varios lenguajes.
    """
    return _build_specialized_agent("code_agent", get_tools_for_agent("code_agent"))


def create_web_scraping_agent():
    """
    Crea un agente especializado en web scraping.

    Este agente puede extraer información de páginas web y capturar endpoints JSON.
    """
    return _build_specialized_agent("web_scraping_agent", get_tools_for_agent("web_scraping_agent"))
