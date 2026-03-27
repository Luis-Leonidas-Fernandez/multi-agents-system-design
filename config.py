"""
Configuración para el sistema multi-agentes

Proveedores soportados via LLM_PROVIDER:
  openai  → ChatOpenAI (default)
  azure   → AzureChatOpenAI
  ollama  → ChatOllama (modelos locales, sin API key)
"""
import os
from dotenv import load_dotenv

load_dotenv()

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# LangSmith se activa automáticamente cuando LANGCHAIN_TRACING_V2=true
# No requiere código adicional — LangChain detecta las env vars al importar
if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
    project = os.getenv("LANGCHAIN_PROJECT", "multi-agents")
    print(f"[observabilidad] LangSmith activo → proyecto: {project}")


def get_llm():
    """Crea y retorna una instancia del LLM según LLM_PROVIDER."""
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    if provider == "azure":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=TEMPERATURE,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama  # pip install langchain-ollama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            temperature=TEMPERATURE,
        )

    # default: openai
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY no encontrada. "
            "Configura tu API key en .env o cambia LLM_PROVIDER."
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=TEMPERATURE,
        api_key=api_key,
    )
