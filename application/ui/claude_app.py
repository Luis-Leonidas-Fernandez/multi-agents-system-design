"""TUI estilo Claude para el sistema multi-agentes."""
from __future__ import annotations

import asyncio
import os
from io import StringIO
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Input, RichLog, Static

from application.services.agent_registry import get_agent_specs
from application.services.cli_dispatch import dispatch_inspection_command
from application.services.session_inspection import format_cli_status


TURN_TIMEOUT_SECONDS = float(os.getenv("TURN_TIMEOUT_SECONDS", "60"))


def _message_text(title: str, content: str, color: str) -> Text:
    text = Text()
    text.append(f"{title}", style=f"bold {color}")
    lines = content.splitlines() or [content]
    for i, line in enumerate(lines):
        if i == 0:
            text.append(" ")
            text.append(line)
        else:
            text.append("\n  ")
            text.append(line)
    return text


def _render_markdown(content: str, width: int = 90) -> Text:
    """Pre-render Markdown to Rich Text via Console so RichLog displays it correctly."""
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, width=width, color_system="truecolor")
    console.print(Markdown(content))
    return Text.from_ansi(buffer.getvalue())


def _write_user_message(log, content: str) -> None:
    label = Text()
    label.append("  you  ", style="bold black on #d946ef")
    label.append("  " + content, style="#f1f5f9")
    log.write(label)


def _write_assistant_message(log, content: str) -> None:
    log.write(_render_markdown(content))
    log.write(Rule(style="#243145"))


def _agents_text() -> str:
    specs = list(get_agent_specs())
    return " · ".join(f"{spec.name}[{spec.risk_level}]" for spec in specs)


class ClaudeApp(App):
    CSS = """
    Screen {
        background: #0b0f14;
        color: #e5e7eb;
    }

    #shell {
        height: 100%;
        padding: 1 2;
    }

    #chrome {
        height: 4;
        border: round #243145;
        background: #0f172a;
        padding: 0 1;
        margin-bottom: 1;
    }

    #title {
        color: #7dd3fc;
        text-style: bold;
    }

    #status {
        color: #cbd5e1;
    }

    #agents {
        color: #64748b;
    }

    #conversation-label {
        color: #94a3b8;
        text-style: bold;
    }

    #transcript {
        border: round #243145;
        background: #0b1220;
        height: 1fr;
        padding: 1 1;
    }

    #prompt {
        border: round #243145;
        background: #0f172a;
        height: 3;
    }

    #footer {
        color: #94a3b8;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("ctrl+l", "refresh", "Refresh"),
        ("ctrl+c", "quit", "Quit"),
        ("escape", "focus_prompt", "Focus prompt"),
    ]

    def __init__(self, runtime, lifecycle) -> None:
        super().__init__()
        self._runtime = runtime
        self._lifecycle = lifecycle
        self._closed = False
        self._active_turn_tasks: set[asyncio.Task[None]] = set()

    def compose(self) -> ComposeResult:
        with Container(id="shell"):
            with Container(id="chrome"):
                yield Static("Multi-Agentes", id="title")
                yield Static("", id="status")
                yield Static(_agents_text(), id="agents")
                yield Static("Ctrl+L refresca · Esc enfoca prompt · Ctrl+C sale", id="footer")
            yield Static("CONVERSATION", id="conversation-label")
            yield RichLog(id="transcript", wrap=True, highlight=False)
            yield Input(placeholder="[session] › ", id="prompt")

    async def on_mount(self) -> None:
        self._refresh_chrome()
        self._seed_transcript()
        self.query_one("#prompt", Input).focus()

    def _refresh_chrome(self) -> None:
        session_view = self._lifecycle.view()
        status = format_cli_status(session_view.snapshot)
        if self._active_turn_tasks:
            status += " · procesando…"
        self.query_one("#status", Static).update(status)
        self.query_one("#prompt", Input).placeholder = f"[{session_view.snapshot.session_id}] › "
        self.query_one("#agents", Static).update(_agents_text())

    def _seed_transcript(self) -> None:
        transcript = self._runtime.build_session_artifact(self._lifecycle.session_id).transcript
        transcript_log = self.query_one("#transcript", RichLog)
        transcript_log.clear()
        if not transcript:
            transcript_log.write(Text("todavía no hay mensajes en esta sesión", style="#64748b"))
            return
        for item in transcript:
            role = str(item.get("role", "message")).lower()
            content = str(item.get("content", ""))
            if role in {"human", "user"}:
                _write_user_message(transcript_log, content)
            elif role in {"ai", "assistant"}:
                _write_assistant_message(transcript_log, content)
            else:
                transcript_log.write(_message_text(role, content, "#38bdf8"))

    def _append_block(self, title: str, content: str, border_style: str) -> None:
        self.query_one("#transcript", RichLog).write(_message_text(title.lower(), content, border_style))

    def _track_turn_task(self, task: asyncio.Task[None]) -> None:
        self._active_turn_tasks.add(task)
        self._refresh_chrome()

        def _done(finished: asyncio.Task[None]) -> None:
            self._active_turn_tasks.discard(finished)
            if self._closed:
                return
            self._refresh_chrome()
            self.query_one("#prompt", Input).focus()

        task.add_done_callback(_done)

    async def _cancel_active_turns(self) -> None:
        tasks = [task for task in self._active_turn_tasks if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active_turn_tasks.difference_update(tasks)

    async def _process_turn(self, text: str) -> None:
        try:
            session = self._lifecycle.resolve(text)
            if session.turn_context is None:
                return
            turn = await asyncio.wait_for(self._runtime.execute_turn(session.turn_context), timeout=TURN_TIMEOUT_SECONDS)
            if self._closed:
                return
            if turn.response:
                _write_assistant_message(self.query_one("#transcript", RichLog), turn.response)
            else:
                self._append_block("SYSTEM", "El turno terminó sin respuesta.", "#64748b")
        except asyncio.TimeoutError:
            if not self._closed:
                self._append_block("ERROR", f"El turno superó {int(TURN_TIMEOUT_SECONDS)}s. Revisá proveedor/red/modelo.", "#ef4444")
        except asyncio.CancelledError:
            return
        except Exception as exc:
            if not self._closed:
                self._append_block("ERROR", str(exc), "#ef4444")
        finally:
            if not self._closed:
                self._refresh_chrome()

    async def _close_session(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._cancel_active_turns()
        closure = await self._lifecycle.close()
        await self._runtime.shutdown()
        if closure.memory_written:
            self._append_block("SYSTEM", "Memoria destilada y persistida.", "#a78bfa")
        self._append_block(
            "SYSTEM",
            f"Sesión guardada: {closure.after.message_count} mensajes previos{', con memoria' if closure.after.has_memory else ''}.",
            "#94a3b8",
        )

    @on(Input.Submitted, "#prompt")
    async def handle_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return

        if text.lower() in {"salir", "exit", "quit"}:
            await self._close_session()
            self.exit()
            return

        if text.startswith("/"):
            result = dispatch_inspection_command(text, self._lifecycle, self._runtime)
            if result.handled:
                self._append_block("COMMAND", text, "#38bdf8")
                for line in result.lines:
                    self._append_block("SYSTEM", line, "#64748b")
                self._refresh_chrome()
                return

        _write_user_message(self.query_one("#transcript", RichLog), text)
        task = asyncio.create_task(self._process_turn(text))
        self._track_turn_task(task)
        self.query_one("#prompt", Input).focus()

    def action_refresh(self) -> None:
        self._refresh_chrome()
        self._seed_transcript()

    def action_focus_prompt(self) -> None:
        self.query_one("#prompt", Input).focus()

    async def on_shutdown_request(self) -> None:
        await self._close_session()


__all__ = ["ClaudeApp"]
