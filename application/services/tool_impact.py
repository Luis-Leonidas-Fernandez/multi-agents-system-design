"""Vista previa de impacto para tools de código y web.

El objetivo no es adivinar el diff exacto, sino mostrar una estimación
honesta de archivos afectados, alcance y riesgo antes de ejecutar.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import re
import time
from pathlib import Path
from typing import Any, Mapping

from application.services.tool_registry import ToolSpec, get_tool_spec


@dataclass(frozen=True)
class ToolImpactPreview:
    agent_name: str
    tool_name: str
    category: str
    scope: str
    confidence: str
    estimated_diff_lines: str
    affected_files: tuple[str, ...]
    side_effects: str
    risk_notes: tuple[str, ...]
    generated_at_ms: int
    repo_matches: tuple[str, ...] = ()
    matched_symbols: tuple[str, ...] = ()
    discovery_basis: str = "heuristic"


class ToolImpactService:
    def build_preview(
        self,
        *,
        agent_name: str,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
        spec: ToolSpec | None = None,
        repo_root: Path | str | None = None,
    ) -> ToolImpactPreview:
        tool_spec = spec or get_tool_spec(tool_name)
        args = dict(arguments or {})
        repo_root_path = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
        if tool_spec.category == "code":
            return self._build_code_preview(agent_name, tool_spec, args, repo_root_path)
        if tool_spec.category == "web":
            return self._build_web_preview(agent_name, tool_spec, args, repo_root_path)
        return self._build_generic_preview(agent_name, tool_spec, args)

    def render_lines(self, preview: ToolImpactPreview | Mapping[str, Any]) -> list[str]:
        data = preview if isinstance(preview, Mapping) else asdict(preview)
        lines = [
            f"[impact] tool={data.get('tool_name', '?')} agent={data.get('agent_name', '?')} category={data.get('category', '?')} scope={data.get('scope', '?')} confidence={data.get('confidence', '?')}",
            f"  diff≈{data.get('estimated_diff_lines', '?')} side_effects={data.get('side_effects', '?')}",
        ]
        if data.get("affected_files"):
            lines.append(f"  files={', '.join(data.get('affected_files', []))}")
        if data.get("repo_matches"):
            lines.append(f"  repo_matches={', '.join(data.get('repo_matches', []))}")
        if data.get("matched_symbols"):
            lines.append(f"  symbols={', '.join(data.get('matched_symbols', []))}")
        if data.get("discovery_basis"):
            lines.append(f"  basis={data.get('discovery_basis')}")
        if data.get("risk_notes"):
            lines.append(f"  notes={'; '.join(data.get('risk_notes', []))}")
        return lines

    def _build_code_preview(self, agent_name: str, spec: ToolSpec, arguments: dict[str, Any], repo_root: Path) -> ToolImpactPreview:
        task = str(arguments.get("task", "")).strip()
        language = str(arguments.get("language", "python")).strip().lower()
        candidate_files = self._guess_code_files(task, language, arguments)
        repo_matches = self._discover_repo_files(repo_root, task, language, arguments)
        matched_symbols = self._discover_repo_symbols(repo_root, repo_matches, task, language, arguments)
        combined_files = self._merge_candidates(candidate_files, repo_matches)
        scope = "new-file" if any(path.endswith(('.py', '.ts', '.js', '.go', '.java')) for path in combined_files) else "module-change"
        diff_lines = self._estimate_code_diff(language, task, combined_files)
        risk_notes = (
            "Puede crear o modificar código sin conocer todo el contexto del repo",
            "Revisar imports, tests y nombres de archivos antes de aceptar",
        )
        confidence = "high" if repo_matches or matched_symbols else ("medium" if candidate_files else "low")
        basis = "repo-aware" if repo_matches else ("repo-aware+symbols" if matched_symbols else "heuristic")
        if repo_matches:
            risk_notes = risk_notes + ("Se priorizaron archivos reales del repo según la tarea",)
        if matched_symbols:
            risk_notes = risk_notes + ("Se detectaron símbolos reales relacionados con la tarea",)
        return ToolImpactPreview(
            agent_name=agent_name,
            tool_name=spec.name,
            category=spec.category,
            scope=scope,
            confidence=confidence,
            estimated_diff_lines=diff_lines,
            affected_files=tuple(combined_files) if combined_files else ("<archivo a determinar por la tarea>",),
            side_effects="cambios locales en el repositorio",
            risk_notes=risk_notes,
            generated_at_ms=int(time.time() * 1000),
            repo_matches=tuple(repo_matches),
            matched_symbols=tuple(matched_symbols),
            discovery_basis=basis,
        )

    def _build_web_preview(self, agent_name: str, spec: ToolSpec, arguments: dict[str, Any], _repo_root: Path) -> ToolImpactPreview:
        tool_name = spec.name
        url = str(arguments.get("url", "")).strip()
        candidate_files: list[str] = []
        side_effects = "sin cambios en el repositorio"
        notes = ["Requiere validación de dominio y URL"]

        if tool_name == "scrape_website_with_json_capture" and url:
            candidate_files.append(self._build_json_capture_path(url))
            side_effects = "escribe bundles JSON en data/web_scraping/data_trading/"
            notes.append("Puede persistir respuestas JSON y metadatos de captura")
        elif tool_name == "web_fetch":
            notes.append("Recupera contenido web y sintetiza respuesta sin tocar archivos locales")
        elif tool_name in {"scrape_website_simple", "scrape_website_dynamic"}:
            notes.append("La tool sólo lee la web; el riesgo es de red, no del árbol local")
        elif tool_name == "search_web":
            notes.append("Sólo consulta internet; no produce archivos locales")
        else:
            notes.append("Impacto externo limitado al dominio/endpoint consultado")

        return ToolImpactPreview(
            agent_name=agent_name,
            tool_name=tool_name,
            category=spec.category,
            scope="network" if not candidate_files else "artifact-write",
            confidence="high" if candidate_files else "medium",
            estimated_diff_lines="0 líneas en el repo" if not candidate_files else "0 líneas de código, 1 artefacto de datos",
            affected_files=tuple(candidate_files) if candidate_files else tuple(),
            side_effects=side_effects,
            risk_notes=tuple(notes),
            generated_at_ms=int(time.time() * 1000),
            repo_matches=tuple(candidate_files),
            matched_symbols=tuple(),
            discovery_basis="repo-aware" if candidate_files else "heuristic",
        )

    def _build_generic_preview(self, agent_name: str, spec: ToolSpec, arguments: dict[str, Any]) -> ToolImpactPreview:
        return ToolImpactPreview(
            agent_name=agent_name,
            tool_name=spec.name,
            category=spec.category,
            scope="no-op",
            confidence="low",
            estimated_diff_lines="sin cambios locales estimables",
            affected_files=tuple(),
            side_effects="sin impacto local conocido",
            risk_notes=("Impacto no especializado; revisar manualmente los argumentos",),
            generated_at_ms=int(time.time() * 1000),
        )

    def _discover_repo_files(self, repo_root: Path, task: str, language: str, arguments: dict[str, Any]) -> list[str]:
        if not repo_root.exists():
            return []
        keywords = self._extract_keywords(task, language, arguments)
        if not keywords:
            return []
        allowed_suffixes = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".java", ".md"}
        ignored_dirs = {".git", "node_modules", "dist", "build", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache"}
        scored: list[tuple[int, str]] = []

        for path in repo_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
                continue
            if any(part in ignored_dirs for part in path.parts):
                continue
            rel = path.relative_to(repo_root).as_posix()
            rel_lower = rel.lower()
            stem_lower = path.stem.lower()
            score = 0
            for keyword in keywords:
                if keyword in rel_lower:
                    score += 4
                if keyword in stem_lower:
                    score += 5
            if score == 0:
                continue
            try:
                snippet = path.read_text(encoding="utf-8", errors="ignore")[:3000].lower()
                score += sum(1 for keyword in keywords if keyword in snippet)
            except Exception:
                pass
            scored.append((score, rel))

        scored.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
        matches = [rel for _, rel in scored[:5]]
        matches.extend(self._augment_with_tests(repo_root, matches))
        return self._unique(matches)

    def _discover_repo_symbols(self, repo_root: Path, repo_matches: list[str], task: str, language: str, arguments: dict[str, Any]) -> list[str]:
        if not repo_matches:
            return []
        candidates = self._symbol_candidates(task, language, arguments)
        if not candidates:
            return []
        matches: list[str] = []
        for rel in repo_matches:
            path = repo_root / rel
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            file_symbols = self._extract_file_symbols(content)
            for symbol in file_symbols:
                normalized = self._normalize_symbol(symbol)
                if normalized in candidates and symbol not in matches:
                    matches.append(symbol)
        return matches

    def _augment_with_tests(self, repo_root: Path, matches: list[str]) -> list[str]:
        extras: list[str] = []
        for rel in matches:
            path = repo_root / rel
            if path.suffix.lower() != ".py":
                continue
            test_path = repo_root / "tests" / f"test_{path.stem}.py"
            if test_path.exists():
                extras.append(test_path.relative_to(repo_root).as_posix())
        return extras

    def _extract_keywords(self, task: str, language: str, arguments: dict[str, Any]) -> list[str]:
        tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", f"{task} {language} {' '.join(map(str, arguments.values()))}".lower()))
        stopwords = {"the", "and", "for", "with", "from", "task", "todo", "add", "new", "build", "make", "create", "update", "fix", "feature", "agent", "tool", "code", "write", "using"}
        return sorted(token for token in tokens if token not in stopwords)

    def _symbol_candidates(self, task: str, language: str, arguments: dict[str, Any]) -> set[str]:
        keywords = self._extract_keywords(task, language, arguments)
        candidates: set[str] = set()
        for keyword in keywords:
            base = keyword[:-1] if keyword.endswith("s") and len(keyword) > 3 else keyword
            variants = {
                keyword,
                base,
                f"session{keyword}",
                f"session{base}",
                f"session{keyword}service",
                f"session{base}service",
                f"session{keyword}store",
                f"session{base}store",
                f"{keyword}service",
                f"{base}service",
                f"{keyword}store",
                f"{base}store",
                f"{keyword}manager",
                f"{base}manager",
                f"build{keyword}",
                f"build{base}",
                f"{keyword}_service",
                f"{base}_service",
                f"{keyword}_store",
                f"{base}_store",
                f"{keyword}_manager",
                f"{base}_manager",
            }
            candidates.update(self._normalize_symbol(variant) for variant in variants)
        return candidates

    def _extract_file_symbols(self, content: str) -> list[str]:
        symbols: list[str] = []
        patterns = [
            r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"^\s*from\s+[.\w]+\s+import\s+([A-Za-z0-9_,\s()]+)",
            r"^\s*import\s+([A-Za-z0-9_,\s.]+)",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, content, flags=re.MULTILINE):
                if isinstance(match, tuple):
                    match = " ".join(part for part in match if part)
                for chunk in re.split(r"[,\s()]+", str(match)):
                    chunk = chunk.strip()
                    if chunk and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", chunk):
                        symbols.append(chunk)
        return self._unique(symbols)

    def _normalize_symbol(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    def _merge_candidates(self, primary: list[str], secondary: list[str]) -> list[str]:
        merged: list[str] = []
        for rel in [*secondary, *primary]:
            if rel not in merged:
                merged.append(rel)
        return merged

    def _unique(self, values: list[str]) -> list[str]:
        unique: list[str] = []
        for value in values:
            if value not in unique:
                unique.append(value)
        return unique

    def _guess_code_files(self, task: str, language: str, arguments: dict[str, Any]) -> list[str]:
        files: list[str] = []
        explicit_keys = ("path", "file", "filename", "target_path", "output", "files")
        for key in explicit_keys:
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                files.append(value.strip())
            elif isinstance(value, list):
                files.extend(str(item).strip() for item in value if str(item).strip())

        if files:
            return files

        slug = re.sub(r"[^a-z0-9]+", "_", task.lower()).strip("_") or "feature"
        if language in {"python", "py"}:
            return [f"application/services/{slug}.py", f"tests/test_{slug}.py"]
        if language in {"typescript", "ts"}:
            return [f"src/{slug}.ts", f"tests/{slug}.test.ts"]
        if language in {"javascript", "js"}:
            return [f"src/{slug}.js", f"tests/{slug}.test.js"]
        if language == "go":
            return [f"internal/{slug}/{slug}.go", f"internal/{slug}/{slug}_test.go"]
        if language == "java":
            return [f"src/main/java/{slug}.java", f"src/test/java/{slug}Test.java"]
        return [f"<archivo de {language} a definir>"]

    def _estimate_code_diff(self, language: str, task: str, files: list[str]) -> str:
        base = 18 if len(files) > 1 else 12
        if language in {"typescript", "ts", "javascript", "js"}:
            base = 14
        elif language == "go":
            base = 20
        elif language == "java":
            base = 28
        if "test" in task.lower():
            base = max(8, base - 4)
        high = base + 18
        return f"~{base}-{high} líneas"

    def _build_json_capture_path(self, url: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", url.lower()).strip("_")[:40] or "capture"
        sha = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
        return f"data/web_scraping/data_trading/{slug}_{sha}_<unix_ts>.json"


tool_impact_service = ToolImpactService()


__all__ = ["ToolImpactPreview", "ToolImpactService", "tool_impact_service"]
