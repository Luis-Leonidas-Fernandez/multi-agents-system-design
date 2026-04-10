"""Tools de generación de código del sistema."""

from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field


@tool
def write_code(
    task: Annotated[str, Field(description="Descripción clara de la funcionalidad a implementar")],
    language: Annotated[str, Field(description="Lenguaje de programación, ej: 'python', 'javascript', 'typescript'")] = "python",
) -> str:
    """
    Genera un esqueleto de código válido para la tarea indicada.
    Para Python: valida sintaxis con compile(). Retorna código listo para completar.
    """
    import re
    import textwrap

    slug   = re.sub(r"[^a-z0-9]+", "_", task.lower())[:40].strip("_") or "solution"

    if language.lower() == "python":
        code = textwrap.dedent(f"""
            def {slug}(*args, **kwargs):
                \"\"\"
                {task}

                Args:
                    Definir parámetros según los requerimientos.

                Returns:
                    Resultado de la implementación.

                Raises:
                    NotImplementedError: hasta que se complete la implementación.
                \"\"\"
                # TODO: implementar lógica de '{task}'
                raise NotImplementedError("Implementación pendiente")
        """).strip()

        try:
            compile(code, "<generated>", "exec")
            validation = "✅ Sintaxis Python validada."
        except SyntaxError as e:
            validation = f"⚠️ Error de sintaxis detectado: {e}"

        return f"```python\n{code}\n```\n\n{validation}"

    templates = {
        "javascript": f"function {slug}() {{\n  // TODO: {task}\n  throw new Error('Not implemented');\n}}",
        "typescript": f"function {slug}(): void {{\n  // TODO: {task}\n  throw new Error('Not implemented');\n}}",
        "java":       f"public static void {slug}() {{\n    // TODO: {task}\n    throw new UnsupportedOperationException();\n}}",
        "go":         f"func {slug}() {{\n\t// TODO: {task}\n\tpanic(\"not implemented\")\n}}",
    }
    template = templates.get(language.lower(), f"// {task}\n// Implementar en {language}")
    return f"```{language}\n{template}\n```"


__all__ = ["write_code"]
