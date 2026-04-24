"""Feature-level application API for math calculations."""

from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field


@tool
def calculate(
    expression: Annotated[str, Field(description="Expresión matemática a evaluar, ej: '2 + 2', 'sqrt(16)', 'sin(pi/2)'")],
) -> str:
    """Evalúa una expresión matemática y retorna el resultado."""
    try:
        import math
        import simpleeval
    except ImportError as e:
        return f"Error al calcular: {str(e)}"

    try:
        result = simpleeval.simple_eval(
            expression,
            names={"pi": math.pi, "e": math.e},
            functions={
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "log": math.log,
                "exp": math.exp,
                "abs": abs,
                "round": round,
            },
        )
        return f"Resultado: {result}"
    except (simpleeval.FeatureNotAvailable, simpleeval.InvalidExpression) as e:
        return f"Expresión no permitida: {str(e)}"
    except Exception as e:
        return f"Error al calcular: {str(e)}"


__all__ = ["calculate"]
