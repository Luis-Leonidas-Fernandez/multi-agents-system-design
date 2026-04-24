"""Construcción del chain estructurado del supervisor."""

from typing import Any, Callable

from langchain_core.prompts import ChatPromptTemplate

from application.services.supervisor_prompt import build_supervisor_prompt_assembly
from core.domain.models import RoutingDecision


def build_supervisor_chain(get_llm_fn: Callable[[], Any]):
    assembly = build_supervisor_prompt_assembly()
    prompt = ChatPromptTemplate.from_messages([
        ("system", assembly.system_prompt),
        ("user", "{input}"),
    ])
    return prompt | get_llm_fn().with_structured_output(RoutingDecision)


__all__ = ["build_supervisor_chain"]
