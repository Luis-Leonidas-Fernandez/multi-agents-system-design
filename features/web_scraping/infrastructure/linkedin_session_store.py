"""Persistencia privada y atómica del estado autenticado de LinkedIn."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
import tempfile
from datetime import datetime, timezone
from typing import Any


_DEFAULT_STORAGE_STATE = "data/private/linkedin/default/storage_state.json"
_DEFAULT_PROFILE_DIRECTORY = "browser-profile"
_BROWSER_METADATA_SCHEMA_VERSION = "2"
_PROFILE_MODE = "persistent"


def _has_managed_private_segments(path: Path) -> bool:
    parts = [part.lower() for part in path.parts]
    return any(
        parts[index : index + 3] == ["data", "private", "linkedin"]
        for index in range(max(0, len(parts) - 2))
    )


def _looks_like_personal_browser_profile(path: Path) -> bool:
    normalized = path.as_posix().lower()
    common_profile_roots = (
        "/library/application support/bravesoftware/brave-browser",
        "/library/application support/google/chrome",
        "/appdata/local/google/chrome/user data",
        "/appdata/local/bravesoftware/brave-browser/user data",
    )
    suspicious_name = path.name.lower() == "default" or bool(
        re.fullmatch(r"profile\s+\d+", path.name, flags=re.IGNORECASE)
    )
    return suspicious_name or any(
        marker in normalized for marker in common_profile_roots
    )


def get_linkedin_storage_state_path() -> Path:
    configured = (os.getenv("LINKEDIN_STORAGE_STATE_PATH") or "").strip()
    return Path(configured or _DEFAULT_STORAGE_STATE).expanduser().resolve()


class LinkedInSessionStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path).expanduser().resolve() if path else get_linkedin_storage_state_path()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def browser_metadata_path(self) -> Path:
        return self._path.with_name(f"{self._path.stem}.browser.json")

    @property
    def runtime_diagnostic_path(self) -> Path:
        return self._path.with_name(f"{self._path.stem}.runtime-diagnostic.json")

    @property
    def default_profile_path(self) -> Path:
        return self._path.parent / _DEFAULT_PROFILE_DIRECTORY

    def exists(self) -> bool:
        return self._path.is_file()

    def validate(self) -> None:
        if not self.exists():
            raise FileNotFoundError(
                "No hay una sesión LinkedIn inicializada. Ejecutá "
                "`python scripts/bootstrap_linkedin_session.py`."
            )
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("El estado de sesión LinkedIn está corrupto.") from exc
        if not isinstance(payload, dict):
            raise ValueError("El estado de sesión LinkedIn tiene un formato inválido.")
        if not isinstance(payload.get("cookies"), list) or not isinstance(payload.get("origins"), list):
            raise ValueError("El estado de sesión LinkedIn no contiene el contrato Playwright esperado.")
        if os.name != "nt" and stat.S_IMODE(self._path.stat().st_mode) != 0o600:
            raise ValueError("El estado de sesión LinkedIn tiene permisos inseguros.")

    def _prepare_private_parent(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self._path.parent, 0o700)

    def resolve_profile_path(
        self,
        *,
        profile_dir: str | Path | None = None,
        persisted_profile_path: str | Path | None = None,
        create: bool = False,
    ) -> Path:
        """Resuelve CLI > env no vacío > metadata > default y bloquea perfiles personales."""

        env_profile = (os.getenv("LINKEDIN_PROFILE_DIR") or "").strip()
        if profile_dir is not None:
            raw_path = str(profile_dir).strip()
        elif env_profile:
            raw_path = env_profile
        elif str(persisted_profile_path or "").strip():
            raw_path = str(persisted_profile_path).strip()
        else:
            raw_path = str(self.default_profile_path)
        if not raw_path:
            raise ValueError("La ruta del perfil dedicado de LinkedIn está vacía.")

        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate)
        candidate = candidate.resolve()
        if (
            _looks_like_personal_browser_profile(candidate)
            and not _has_managed_private_segments(candidate)
        ):
            raise ValueError(
                "LINKEDIN_PROFILE_DIR parece un perfil personal de Chrome/Brave. "
                "Usá únicamente el perfil dedicado bajo data/private/linkedin."
            )
        private_root = self._path.parent.resolve()
        try:
            relative = candidate.relative_to(private_root)
        except ValueError as exc:
            raise ValueError(
                "LINKEDIN_PROFILE_DIR debe estar dentro del directorio privado "
                f"administrado de LinkedIn: {private_root}."
            ) from exc
        if not relative.parts:
            raise ValueError(
                "LINKEDIN_PROFILE_DIR debe ser un subdirectorio dedicado, no el "
                "directorio privado raíz."
            )
        if candidate.exists() and not candidate.is_dir():
            raise ValueError(
                "LINKEDIN_PROFILE_DIR existe pero no es un directorio."
            )
        if create:
            self._prepare_private_parent()
            candidate.mkdir(parents=True, exist_ok=True)
            os.chmod(candidate, 0o700)
        elif not candidate.is_dir():
            raise FileNotFoundError(
                "No existe el perfil persistente de LinkedIn. Ejecutá "
                "`python scripts/bootstrap_linkedin_session.py`."
            )
        if os.name != "nt" and stat.S_IMODE(candidate.stat().st_mode) != 0o700:
            raise ValueError(
                "El perfil persistente de LinkedIn tiene permisos inseguros."
            )
        return candidate

    def _write_private_json(self, destination: Path, payload: dict[str, Any]) -> None:
        self._prepare_private_parent()
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(self._path.parent),
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, destination)
            os.chmod(destination, 0o600)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def save_from_context(
        self,
        context,
        *,
        launch_config: object | None = None,
        profile_path: str | Path | None = None,
    ) -> None:
        self._prepare_private_parent()
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self._path.name}.",
            suffix=".tmp",
            dir=str(self._path.parent),
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            context.storage_state(path=str(tmp_path))
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, self._path)
            os.chmod(self._path, 0o600)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        if launch_config is not None:
            if profile_path is None:
                raise ValueError(
                    "Se requiere profile_path para registrar el contexto persistente."
                )
            resolved_profile = self.resolve_profile_path(
                profile_dir=profile_path,
                create=False,
            )
            executable_path = getattr(launch_config, "executable_path", None)
            self._write_private_json(
                self.browser_metadata_path,
                {
                    "schema_version": _BROWSER_METADATA_SCHEMA_VERSION,
                    "profile_mode": _PROFILE_MODE,
                    "profile_path": str(resolved_profile),
                    "browser": str(getattr(launch_config, "browser", "")).strip(),
                    "executable_path": (
                        str(executable_path) if executable_path is not None else ""
                    ),
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        if self.runtime_diagnostic_path.exists():
            self.runtime_diagnostic_path.unlink()

    def load_browser_metadata(self) -> dict[str, str] | None:
        path = self.browser_metadata_path
        if not path.exists():
            return None
        if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise ValueError(
                "La metadata del navegador LinkedIn tiene permisos inseguros."
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "La metadata del navegador LinkedIn está corrupta."
            ) from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != _BROWSER_METADATA_SCHEMA_VERSION
            or not isinstance(payload.get("browser"), str)
            or not payload.get("browser", "").strip()
            or not isinstance(payload.get("executable_path"), str)
            or payload.get("profile_mode") != _PROFILE_MODE
            or not isinstance(payload.get("profile_path"), str)
            or not payload.get("profile_path", "").strip()
        ):
            raise ValueError(
                "La metadata del navegador LinkedIn no describe un perfil persistente "
                "válido. Ejecutá nuevamente el bootstrap."
            )
        profile_path = self.resolve_profile_path(
            persisted_profile_path=payload["profile_path"],
            create=False,
        )
        return {
            "browser": payload["browser"].strip(),
            "executable_path": payload["executable_path"].strip(),
            "profile_mode": _PROFILE_MODE,
            "profile_path": str(profile_path),
        }

    def record_runtime_failure(
        self,
        reason: str,
        *,
        browser: str = "",
        headless: bool | None = None,
        profile_path: str | Path | None = None,
        diagnostic: dict[str, Any] | None = None,
    ) -> None:
        """Conserva sesión/cookies y escribe sólo diagnóstico no secreto."""

        allowed_diagnostic_keys = {
            "final_path",
            "linkedin_host",
            "is_auth_checkpoint",
            "has_li_at",
            "has_global_nav",
            "has_login_form",
        }
        safe_diagnostic = {
            key: value
            for key, value in (diagnostic or {}).items()
            if key in allowed_diagnostic_keys
            and isinstance(value, (str, bool, int, float, type(None)))
        }
        self._write_private_json(
            self.runtime_diagnostic_path,
            {
                "schema_version": "2",
                "reason": (reason or "unknown")[:200],
                "browser": browser[:80],
                "headless": headless,
                "profile_mode": _PROFILE_MODE,
                "profile_path": str(profile_path or "")[:500],
                "diagnostic": safe_diagnostic,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "storage_state_retained": self.exists(),
            },
        )

    def invalidate(self, reason: str = "", *, confirmed: bool = False) -> None:
        """Borrado destructivo reservado a corrupción/permisos confirmados."""

        safe_reasons = {"corrupt_state", "unsafe_permissions"}
        if reason not in safe_reasons and not confirmed:
            raise ValueError(
                "La sesión LinkedIn sólo puede invalidarse por corrupción/permisos "
                "o con confirmación explícita."
            )
        if not self.exists() and not self.browser_metadata_path.exists():
            return
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        if self._path.exists():
            self._path.unlink()
        if self.browser_metadata_path.exists():
            self.browser_metadata_path.unlink()
        if reason:
            reason_path = self._path.with_name(
                f"{self._path.stem}.invalid-{timestamp}.reason"
            )
            reason_path.write_text(reason[:200], encoding="utf-8")
            os.chmod(reason_path, 0o600)


__all__ = ["LinkedInSessionStore", "get_linkedin_storage_state_path"]
