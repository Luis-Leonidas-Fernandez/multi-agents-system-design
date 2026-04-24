"""Helpers de configuración y selección de proveedor LLM."""
import logging
import os
import json
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Any, cast

from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "WARNING").upper(),
    format="%(levelname)s %(name)s %(message)s",
)

_log = logging.getLogger(__name__)
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
    project = os.getenv("LANGCHAIN_PROJECT", "multi-agents")
    _log.info("LangSmith activo → proyecto: %s", project)

_VALID_AGENTDOG_POLICIES = {"fail_open", "fail_closed", "fail_soft"}
_VALID_AGENTDOG_EVAL_MODES = {"all_nodes", "high_risk_only", "final_only"}


@dataclass(frozen=True)
class WebSearchRuntimeConfig:
    selected_provider: str = ""
    provider_configured: str = ""


@lru_cache(maxsize=8)
def _load_web_search_runtime_config(cache_key: str) -> WebSearchRuntimeConfig:
    """Internal loader keyed by config path + env snapshot."""
    _ = cache_key
    candidates = [
        os.getenv("WEB_SEARCH_CONFIG", ""),
        str(Path(__file__).resolve().parents[1] / "policies" / "web_search_runtime_config.json"),
    ]
    for path in candidates:
        if not path:
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                raw = json.load(handle)
            if isinstance(raw, dict):
                selected = str(raw.get("selected_provider") or raw.get("provider") or "").strip()
                configured = str(raw.get("provider_configured") or raw.get("configured_provider") or "").strip()
                return WebSearchRuntimeConfig(selected_provider=selected, provider_configured=configured)
        except FileNotFoundError:
            continue
        except Exception:
            continue

    return WebSearchRuntimeConfig(
        selected_provider=os.getenv("WEB_SEARCH_SELECTED_PROVIDER", "").strip(),
        provider_configured=os.getenv("WEB_SEARCH_PROVIDER", "").strip(),
    )


def get_web_search_runtime_config() -> WebSearchRuntimeConfig:
    """Carga la configuración runtime de web search desde config o env.

    Precedencia:
    1. WEB_SEARCH_CONFIG (path explícito a JSON)
    2. application/policies/web_search_runtime_config.json
    3. variables de entorno heredadas
    """
    candidates = [
        os.getenv("WEB_SEARCH_CONFIG", ""),
        str(Path(__file__).resolve().parents[1] / "policies" / "web_search_runtime_config.json"),
    ]
    selected_path = next((path for path in candidates if path), "")
    mtime = ""
    if selected_path:
        try:
            mtime = str(Path(selected_path).stat().st_mtime_ns)
        except Exception:
            mtime = "missing"
    cache_key = "|".join([
        selected_path,
        mtime,
        os.getenv("WEB_SEARCH_SELECTED_PROVIDER", ""),
        os.getenv("WEB_SEARCH_PROVIDER", ""),
    ])
    return _load_web_search_runtime_config(cache_key)


get_web_search_runtime_config.cache_clear = _load_web_search_runtime_config.cache_clear  # type: ignore[attr-defined]


def validate_env() -> None:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider not in ("openai", "azure", "ollama"):
        _log.warning("LLM_PROVIDER='%s' no reconocido. Valores válidos: openai, azure, ollama. Se usará openai.", provider)

    policy = os.getenv("AGENTDOG_POLICY", "fail_open").strip().lower()
    if policy not in _VALID_AGENTDOG_POLICIES:
        _log.warning("AGENTDOG_POLICY='%s' no reconocido. Valores válidos: %s. Se usará fail_open.", policy, _VALID_AGENTDOG_POLICIES)

    eval_mode = os.getenv("AGENTDOG_EVAL_MODE", "high_risk_only").strip().lower()
    if eval_mode not in _VALID_AGENTDOG_EVAL_MODES:
        _log.warning("AGENTDOG_EVAL_MODE='%s' no reconocido. Valores válidos: %s. Se usará high_risk_only.", eval_mode, _VALID_AGENTDOG_EVAL_MODES)

    if not os.getenv("AGENTDOG_AUDIT_LOG"):
        _log.info("AGENTDOG_AUDIT_LOG no configurado — audit log irá a stdout.")


def _build_azure_llm(resolved_temperature: float):
    from langchain_openai import AzureChatOpenAI
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=cast(Any, SecretStr(azure_api_key)) if azure_api_key else None,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature=resolved_temperature,
    )


def _build_ollama_llm(resolved_temperature: float):
    from langchain_ollama import ChatOllama  # pyright: ignore[reportMissingImports]
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3"),
        temperature=resolved_temperature,
    )


def _build_openai_llm(resolved_temperature: float):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY no encontrada. "
            "Configura tu API key en .env o cambia LLM_PROVIDER."
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=resolved_temperature,
        api_key=cast(Any, SecretStr(api_key)),
    )


def get_llm(temperature: Optional[float] = None):
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    resolved_temperature = TEMPERATURE if temperature is None else temperature

    if provider == "azure":
        return _build_azure_llm(resolved_temperature)
    if provider == "ollama":
        return _build_ollama_llm(resolved_temperature)
    return _build_openai_llm(resolved_temperature)


__all__ = [
    "TEMPERATURE",
    "WebSearchRuntimeConfig",
    "get_web_search_runtime_config",
    "validate_env",
    "get_llm",
]
