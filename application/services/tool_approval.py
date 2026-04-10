"""Vista previa de aprobación para tools con foco en HITL.

Este servicio convierte el contrato de permisos/riesgo de una tool en una
previsualización humana legible para CLI y prompts de confirmación.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping

from application.policies.tool_permissions import ToolPermissionDecision, decide_tool_permission
from application.services.tool_impact import ToolImpactPreview, tool_impact_service
from application.services.tool_registry import ToolSpec, build_tool_catalog_lines, get_tool_spec


@dataclass(frozen=True)
class ToolApprovalPreview:
    agent_name: str
    tool_name: str
    description: str
    risk_level: str
    permission_mode: str
    allowed: bool
    requires_confirmation: bool
    reason: str
    arguments_preview: str
    confirmation_prompt: str
    impact_preview: ToolImpactPreview


class ToolApprovalService:
    def build_preview(
        self,
        *,
        agent_name: str,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
        decision: ToolPermissionDecision | None = None,
        spec: ToolSpec | None = None,
    ) -> ToolApprovalPreview:
        tool_spec = spec or get_tool_spec(tool_name)
        tool_decision = decision or decide_tool_permission(
            risk_level=tool_spec.risk_level,
            permission_mode=tool_spec.permission_mode,
        )
        args_preview = self._preview_arguments(arguments or {})
        impact_preview = tool_impact_service.build_preview(agent_name=agent_name, tool_name=tool_name, arguments=arguments or {}, spec=tool_spec)
        confirmation_prompt = self._build_confirmation_prompt(agent_name, tool_spec, tool_decision, args_preview, impact_preview)
        return ToolApprovalPreview(
            agent_name=agent_name,
            tool_name=tool_name,
            description=tool_spec.description,
            risk_level=tool_spec.risk_level,
            permission_mode=tool_decision.mode,
            allowed=tool_decision.allowed,
            requires_confirmation=tool_decision.requires_confirmation,
            reason=tool_decision.reason,
            arguments_preview=args_preview,
            confirmation_prompt=confirmation_prompt,
            impact_preview=impact_preview,
        )

    def render_preview_lines(self, preview: ToolApprovalPreview | Mapping[str, Any]) -> list[str]:
        data = preview if isinstance(preview, Mapping) else asdict(preview)
        lines = [
            f"[tool] {data.get('tool_name', '?')} agent={data.get('agent_name', '?')} risk={data.get('risk_level', '?')} mode={data.get('permission_mode', '?')}",
            f"  allowed={'yes' if data.get('allowed') else 'no'} confirm={'yes' if data.get('requires_confirmation') else 'no'} reason={data.get('reason', '?')}",
        ]
        if data.get("description"):
            lines.append(f"  desc={data.get('description')}")
        if data.get("arguments_preview"):
            lines.append(f"  args={data.get('arguments_preview')}")
        impact_preview = data.get("impact_preview")
        if impact_preview:
            lines.extend(tool_impact_service.render_lines(impact_preview))
        if data.get("confirmation_prompt"):
            lines.append(f"  prompt={data.get('confirmation_prompt')}")
        return lines

    def render_catalog_lines(self) -> list[str]:
        return build_tool_catalog_lines().splitlines()

    def _preview_arguments(self, arguments: Mapping[str, Any]) -> str:
        if not arguments:
            return "{}"
        parts: list[str] = []
        for key, value in arguments.items():
            value_repr = repr(value)
            if len(value_repr) > 80:
                value_repr = value_repr[:77] + "..."
            parts.append(f"{key}={value_repr}")
        return "{" + ", ".join(parts) + "}"

    def _build_confirmation_prompt(
        self,
        agent_name: str,
        spec: ToolSpec,
        decision: ToolPermissionDecision,
        args_preview: str,
        impact_preview: ToolImpactPreview,
    ) -> str:
        if not decision.requires_confirmation:
            return ""
        impact_line = tool_impact_service.render_lines(impact_preview)[0]
        return (
            f"[APPROVAL] {agent_name} quiere usar {spec.name} ({spec.risk_level}/{decision.mode})\n"
            f"  motivo={decision.reason}\n"
            f"  args={args_preview}\n"
            f"  impacto={impact_line}\n"
            f"  ¿Confirmar? [s/n]: "
        )


tool_approval_service = ToolApprovalService()


__all__ = ["ToolApprovalPreview", "ToolApprovalService", "tool_approval_service"]
