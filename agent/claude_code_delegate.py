"""Parsing and validation for the ``/cc-delegate`` operator command.

Pure, platform-agnostic logic for the explicit "delegate a task to an
authenticated Claude Code tmux worker" command. Kept separate from
``gateway/slash_commands.py`` so it can be unit tested without a
``GatewayRunner``, and separate from ``agent/claude_code_tmux_bridge.py``
so the bridge stays a thin tmux transport with no command-parsing or
allowlist-policy concerns of its own.

This module does not talk to tmux directly -- it only decides *whether*
a request is allowed to reach ``agent.claude_code_tmux_bridge.submit_prompt``
and formats the bridge's result for an operator to read.
"""

from __future__ import annotations

import dataclasses
import shlex
from pathlib import Path
from typing import Optional

from agent.claude_code_tmux_bridge import BridgeResult, BridgeStatus, WorkerConfig, get_worker

USAGE = "Usage: /cc-delegate <worker> [--path <dir>] <prompt>"

# Flags accepted before the prompt text. Both spellings are accepted since
# operators reach for either "path" or "repo" naturally.
_PATH_FLAGS = ("--path", "--repo")


class DelegateValidationError(Exception):
    """Raised when a /cc-delegate invocation fails parsing or validation.

    The message is operator-facing and safe to return verbatim in chat.
    """


@dataclasses.dataclass(frozen=True)
class DelegateRequest:
    """A parsed, worker-resolved, path-validated delegation request."""

    worker_name: str
    worker: WorkerConfig
    workspace: str
    requested_subpath: Optional[str]
    prompt: str

    @property
    def bridge_prompt(self) -> str:
        """Prompt text to send to the bridge, scoped to the workspace.

        When the operator did not supply an explicit path, the prompt is
        sent unmodified -- the worker's own tmux pane already starts in
        its configured workspace. When a subpath was supplied, a short
        advisory prefix is added so Claude Code scopes its work there;
        this is context for the model, not a shell command, so it cannot
        be used to inject a command into the pane.
        """
        if self.requested_subpath is None:
            return self.prompt
        return f"(Work only within {self.workspace}.) {self.prompt}"


def parse_delegate_args(raw_args: str) -> tuple[str, Optional[str], str]:
    """Split raw command text into ``(worker, path, prompt)``.

    Recognizes an optional ``--path <dir>`` (alias ``--repo``) flag
    anywhere after the worker name; everything else becomes the prompt.
    Values may be quoted to include spaces.

    Raises:
        DelegateValidationError: with an operator-facing message when the
            input can't be parsed, no worker is given, or the prompt is
            empty.
    """
    try:
        tokens = shlex.split(raw_args or "")
    except ValueError as exc:
        raise DelegateValidationError(f"Could not parse arguments ({exc}). {USAGE}") from exc

    if not tokens:
        raise DelegateValidationError(USAGE)

    worker_name = tokens[0]
    path: Optional[str] = None
    prompt_tokens: list[str] = []

    i = 1
    while i < len(tokens):
        token = tokens[i]
        if token in _PATH_FLAGS:
            if i + 1 >= len(tokens):
                raise DelegateValidationError(f"{token} requires a directory argument. {USAGE}")
            path = tokens[i + 1]
            i += 2
            continue
        prompt_tokens.append(token)
        i += 1

    prompt = " ".join(prompt_tokens).strip()
    if not prompt:
        raise DelegateValidationError(f"Prompt cannot be empty. {USAGE}")

    return worker_name, path, prompt


def resolve_worker(worker_name: str) -> WorkerConfig:
    """Return the allowlisted ``WorkerConfig`` for *worker_name*.

    Raises:
        DelegateValidationError: if *worker_name* is not registered via
            ``agent.claude_code_tmux_bridge.register_worker``. There is no
            implicit fallback -- an unrecognized worker is refused, never
            guessed at.
    """
    worker = get_worker(worker_name)
    if worker is None:
        raise DelegateValidationError(
            f"Worker {worker_name!r} is not allowlisted for delegation."
        )
    return worker


def resolve_workspace(worker: WorkerConfig, requested_path: Optional[str]) -> str:
    """Confine an optional operator-supplied path beneath the worker's root.

    Returns the resolved, absolute workspace path as a string -- either
    the worker's configured root (no path supplied) or the validated
    subpath. Symlinks and ``..`` segments are resolved before the
    containment check so neither can be used to escape the root.

    Raises:
        DelegateValidationError: if the resolved path is not the root
            itself or a descendant of it.
    """
    root = Path(worker.workspace).resolve()
    if requested_path is None:
        return str(root)

    candidate = Path(requested_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()

    if resolved != root and not resolved.is_relative_to(root):
        raise DelegateValidationError(
            f"Path {requested_path!r} is outside the allowed workspace {worker.workspace!r}."
        )
    return str(resolved)


def build_request(worker_name: str, requested_path: Optional[str], prompt: str) -> DelegateRequest:
    """Resolve and validate a parsed invocation into a ``DelegateRequest``.

    Combines ``resolve_worker`` and ``resolve_workspace``; raises the same
    ``DelegateValidationError`` as either.
    """
    worker = resolve_worker(worker_name)
    workspace = resolve_workspace(worker, requested_path)
    return DelegateRequest(
        worker_name=worker_name,
        worker=worker,
        workspace=workspace,
        requested_subpath=requested_path,
        prompt=prompt,
    )


# ── Result formatting ────────────────────────────────────────────────────────

_STATUS_TEMPLATES: dict[BridgeStatus, str] = {
    BridgeStatus.TIMEOUT: "⏱️ `{worker}` timed out before responding: {error}",
    BridgeStatus.BUSY: "🔒 `{worker}` is already handling another prompt. Try again shortly.",
    # The bridge's error text already names the session and, when safely
    # extractable, a sanitised preview of what's pending -- shown verbatim,
    # not automatically approved.
    BridgeStatus.APPROVAL_REQUIRED: "⏸️ {error}",
    BridgeStatus.AUTH_FAILURE: "🔑 `{worker}` needs to re-authenticate (Claude Code login/OAuth).",
    BridgeStatus.SESSION_MISSING: "❌ `{worker}` has no running tmux session.",
    BridgeStatus.SESSION_DEAD: "💀 `{worker}` session exists but has no live pane.",
    BridgeStatus.FAILURE: "⚠️ Delegation to `{worker}` failed: {error}",
}


def format_ack(request: DelegateRequest) -> str:
    """Operator-facing acknowledgement sent as soon as delegation starts."""
    preview = request.prompt[:80] + ("..." if len(request.prompt) > 80 else "")
    return (
        f"🚀 Delegating to `{request.worker_name}`\n"
        f"Workspace: `{request.workspace}`\n"
        f"Prompt: {preview}"
    )


def format_result(worker_name: str, result: BridgeResult) -> str:
    """Operator-facing status for a completed bridge call.

    For SUCCESS this carries only the scoped Claude response the bridge
    already extracted and sanitised -- no pane history, prompts, tokens,
    or terminal control sequences are included (the bridge strips those
    before the response ever reaches this module).
    """
    if result.status is BridgeStatus.SUCCESS:
        return f"✅ `{worker_name}`:\n{result.response}"
    template = _STATUS_TEMPLATES.get(
        result.status, "⚠️ Delegation to `{worker}` failed: {error}"
    )
    return template.format(worker=worker_name, error=result.error or "unknown error")
