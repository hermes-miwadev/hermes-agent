"""Configurable automatic Claude Code routing for Discord channels.

Lets an operator mark a specific Discord channel so that ordinary user
messages there are delegated straight to an authenticated Claude Code tmux
worker (``agent/claude_code_tmux_bridge.py`` via
``agent/claude_code_delegate.py``) instead of going through Hermes' own
agent loop. This module only decides *whether* a given message should be
auto-routed and, if so, builds the prompt and resolves the worker/workspace
-- it never talks to tmux or Discord directly, and it reuses the existing
delegation layer's validation (worker allowlisting, workspace containment)
rather than duplicating it. See ``gateway/claude_auto_routing.py`` for the
gateway-side call site that wires this into message dispatch.

Config (``config.yaml``)::

    claude_routing:
      - platform: discord
        channel_id: "123456789012345678"
        enabled: true
        worker: claude-momentum
        workspace: /home/michael/code/momentum-studio
        mode: claude_default
        standard_instructions: true   # or a custom string, or false to omit

``mode: claude_default`` is the only routing mode implemented so far:
deterministic, unconditional delegation of every ordinary (non-command,
non-bypass, non-bot) message in the configured channel.

Bypass syntax: a message beginning with ``chat:`` (after Hermes strips the
``@mention``) is never auto-routed, regardless of channel config -- e.g.
``@hermes chat: help me think through the strategy before changing code``.
Explicit slash commands (anything starting with ``/``) are never auto-routed
either; that decision is made by the caller before reaching this module.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# The only routing mode implemented so far. Named so config reads like
# documentation: "claude_default" for this channel means "hand ordinary
# messages straight to Claude Code, deterministically, every time."
CLAUDE_DEFAULT_MODE = "claude_default"
_SUPPORTED_MODES = frozenset({CLAUDE_DEFAULT_MODE})

DEFAULT_STANDARD_INSTRUCTIONS = (
    "Standard implementation requirements for this task: inspect the current "
    "repository state first; use a new branch or an isolated worktree; do not "
    "make unrelated changes; run relevant tests and checks; commit "
    "intentionally; push the branch; create or update a pull request; and "
    "report the branch, commit, PR link, files changed, and tests run."
)

# "chat:" (case-insensitive, optional surrounding space) right at the start
# of the already-mention-stripped message text opts a single message out of
# auto-routing, keeping it with Hermes instead.
_BYPASS_RE = re.compile(r"^\s*chat\s*:\s*", re.IGNORECASE)


class RoutingConfigError(Exception):
    """Raised for a structurally invalid ``claude_routing`` config entry.

    Config mistakes are reported (logged and, at the dispatch site, shown
    to the operator) rather than silently guessed at or ignored.
    """


@dataclasses.dataclass(frozen=True)
class RoutingMapping:
    """One resolved, validated ``claude_routing`` config entry."""

    platform: str
    channel_id: str
    enabled: bool
    worker: str
    workspace: str
    mode: str
    standard_instructions: Optional[str]  # None => omit the standard block


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_standard_instructions(raw: Any) -> Optional[str]:
    """Normalise the ``standard_instructions`` field to text or None.

    Unset or ``true`` -> the built-in default block. ``false`` -> omit the
    block entirely. Any other value -> used verbatim as custom text.
    """
    if raw is None or raw is True:
        return DEFAULT_STANDARD_INSTRUCTIONS
    if raw is False:
        return None
    text = str(raw).strip()
    return text or None


def parse_routing_mapping(entry: Any) -> RoutingMapping:
    """Validate and normalise one ``claude_routing`` config entry.

    Raises:
        RoutingConfigError: with an operator-facing message on any
            structural problem -- missing ``channel_id``/``worker``/
            ``workspace``, or an unsupported ``mode``.
    """
    if not isinstance(entry, dict):
        raise RoutingConfigError(
            f"claude_routing entry must be a mapping, got {type(entry).__name__}"
        )

    channel_id = str(entry.get("channel_id") or "").strip()
    if not channel_id:
        raise RoutingConfigError("claude_routing entry is missing 'channel_id'")

    worker = str(entry.get("worker") or "").strip()
    if not worker:
        raise RoutingConfigError(
            f"claude_routing entry for channel {channel_id!r} is missing 'worker'"
        )

    workspace = str(entry.get("workspace") or "").strip()
    if not workspace:
        raise RoutingConfigError(
            f"claude_routing entry for channel {channel_id!r} is missing 'workspace'"
        )

    mode = str(entry.get("mode") or CLAUDE_DEFAULT_MODE).strip()
    if mode not in _SUPPORTED_MODES:
        raise RoutingConfigError(
            f"claude_routing entry for channel {channel_id!r} has unsupported "
            f"mode {mode!r} (supported: {', '.join(sorted(_SUPPORTED_MODES))})"
        )

    return RoutingMapping(
        platform=str(entry.get("platform") or "discord").strip().lower(),
        channel_id=channel_id,
        enabled=_coerce_bool(entry.get("enabled"), default=True),
        worker=worker,
        workspace=workspace,
        mode=mode,
        standard_instructions=_resolve_standard_instructions(
            entry.get("standard_instructions")
        ),
    )


def load_routing_mappings(config: Any) -> list[RoutingMapping]:
    """Load and validate every ``claude_routing`` entry from gateway config.

    Accepts either a dict-like or attribute-having config object, mirroring
    the dual dict/object handling ``gateway/run.py`` already uses for
    ``quick_commands``. Invalid entries are logged and skipped rather than
    breaking dispatch for every message.
    """
    if isinstance(config, dict):
        raw_entries = config.get("claude_routing") or []
    else:
        raw_entries = getattr(config, "claude_routing", None) or []

    if not isinstance(raw_entries, list):
        logger.warning(
            "claude_routing config must be a list, got %s -- ignoring",
            type(raw_entries).__name__,
        )
        return []

    mappings: list[RoutingMapping] = []
    for entry in raw_entries:
        try:
            mappings.append(parse_routing_mapping(entry))
        except RoutingConfigError as exc:
            logger.warning("Skipping invalid claude_routing entry: %s", exc)
    return mappings


def find_routing_mapping(
    mappings: list[RoutingMapping],
    *,
    platform: str,
    channel_id: Optional[str],
    parent_channel_id: Optional[str] = None,
) -> Optional[RoutingMapping]:
    """Return the enabled mapping matching *channel_id*, or its parent.

    Checking the parent channel too means a thread under a configured
    channel still resolves -- mirroring how other Discord channel-scoped
    config (``free_response_channels`` etc.) matches a channel and its
    threads together. A *disabled* matching entry returns None rather than
    falling through to some other entry -- disabling a mapping always wins.
    """
    candidates = {c for c in (channel_id, parent_channel_id) if c}
    if not candidates:
        return None
    platform = (platform or "").strip().lower()
    for mapping in mappings:
        if mapping.platform != platform or mapping.channel_id not in candidates:
            continue
        return mapping if mapping.enabled else None
    return None


def strip_bypass_prefix(text: str) -> Optional[str]:
    """Return the message with a leading ``chat:`` bypass prefix removed.

    Returns None when no bypass prefix is present, meaning the caller
    should proceed with auto-routing. A non-None (possibly empty) string
    means the operator explicitly asked to keep this one message with
    Hermes instead of Claude Code.
    """
    match = _BYPASS_RE.match(text or "")
    if match is None:
        return None
    return text[match.end():]


def build_auto_routed_prompt(mapping: RoutingMapping, user_text: str) -> str:
    """Compose the prompt sent to Claude Code for an auto-routed message.

    The task leads (so the operator-facing ack preview -- the first ~80
    chars of this string, per ``agent.claude_code_delegate.format_ack`` --
    shows the actual task, not boilerplate) followed by the mapping's
    standard implementation instructions (or the built-in default) unless
    the mapping has them turned off (``standard_instructions: false``).
    These are advisory framing text, not a rule enforced in code -- Claude
    Code, as an agent in its own right, reconciles them against anything
    the user's own message explicitly says.
    """
    if mapping.standard_instructions:
        return f"Task: {user_text}\n\n{mapping.standard_instructions}"
    return user_text


def format_auto_routing_ack(mapping: RoutingMapping, request: Any) -> str:
    """Operator-facing acknowledgement sent as soon as auto-routing fires.

    Reuses ``agent.claude_code_delegate.format_ack`` for the worker/
    workspace/prompt-preview body (identical to a manual ``/cc-delegate``
    ack) and adds the framing that distinguishes an automatic delegation
    from one an operator typed explicitly. The eventual completion/
    approval-required/busy/auth-failure/timeout/failure status is
    delivered separately, through the same bridge-status formatting
    ``/cc-delegate`` uses (see ``agent.claude_code_delegate.format_result``).
    """
    from agent.claude_code_delegate import format_ack

    return (
        f"🤖 Automatic routing activated (mode: `{mapping.mode}`)\n"
        f"{format_ack(request)}"
    )
