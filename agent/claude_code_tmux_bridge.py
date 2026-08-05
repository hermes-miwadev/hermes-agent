"""Claude Code tmux bridge.

Submits prompts to a running Claude Code session in a named tmux pane,
waits for a unique completion marker, and returns the sanitised response.

Usage::

    from agent.claude_code_tmux_bridge import submit_prompt, BridgeStatus

    result = submit_prompt("claude-momentum", "Refactor utils.py to use dataclasses")
    if result.status == BridgeStatus.SUCCESS:
        print(result.response)

One pre-configured worker is registered at module level:
  session ``claude-momentum``, workspace ``/home/michael/code``.

Additional workers can be registered via ``register_worker()``.

Notes
-----
* Newlines in prompts are collapsed to spaces before sending; tmux
  ``send-keys`` treats each newline as an Enter keystroke.  Single-paragraph
  prompts work best.
* Auth failure is detected only at timeout (not during polling) to avoid
  false positives from unrelated pane history.
* An interactive approval prompt ("Do you want to proceed?") is detected on
  every poll (not just at timeout) so a stuck approval doesn't hold the
  per-session lock for the full timeout window. This phase only detects and
  reports the state -- it never approves anything automatically.
* The completion marker must appear alone on its own line to count as done.
  A substring match would also fire on the *echo* of the just-submitted
  prompt (which necessarily contains the marker text, since we asked Claude
  to repeat it back), producing a false empty "success" before Claude has
  done anything.
* The response boundary is anchored on that same echoed prompt (found by
  content, since the marker is a fresh random value each call and cannot
  coincidentally appear elsewhere) rather than a pre-send line count. The
  echo search tolerates the marker landing across a terminal soft-wrap
  boundary -- a long real prompt is one collapsed logical line, and the
  pane can wrap it across many physical rows, splitting the marker itself
  so no single row contains it -- by searching the wrapped rows joined
  back together (see ``_find_echo_line_idx``). Only when no echo can be
  found at all does extraction fall back to the pre-send line count, and
  even then only if that count still falls strictly before the marker
  line: Claude Code's TUI repaints its whole viewport per frame and can
  settle to a different total line count than the pre-send snapshot (e.g.
  a prior turn's tool output collapsing once it's no longer active), and a
  stale count past the marker line would otherwise produce a
  guaranteed-empty, detected-but-empty "success".
* No tmux session is created automatically; the session must already exist.
"""

from __future__ import annotations

import bisect
import dataclasses
import enum
import logging
import re
import subprocess
import threading
import time
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

# ── ANSI / credential / auth patterns ────────────────────────────────────────

_ANSI_RE = re.compile(
    r"\x1b\[[0-9;]*[mGKHFABCDEJ]"
    r"|\x1b\][\S\s]*?\x07"
    r"|\x1b[()][AB012]"
    r"|\x1b[=>]"
    r"|\r",
)

# Redact token-shaped values that might appear in captured pane output.
_CREDENTIAL_RE = re.compile(
    r"(?:Bearer\s+|(?:api[-_]?key|token|password|oauth|secret)[\s=:]+)"
    r"[A-Za-z0-9_\-\.]{16,}",
    re.IGNORECASE,
)

_AUTH_FAILURE_RE = re.compile(
    r"(?:"
    r"authentication\s+(?:required|failed|expired|error)"
    r"|oauth\s+(?:\w+\s+)*(?:error|expired|invalid|required|revoked)"
    r"|sign[\s-]?in\s+(?:required|to|with)"
    r"|login\s+(?:required|to|with)"
    r"|please\s+(?:authenticate|log\s*in|sign\s*in)"
    r"|unauthorized"
    r"|session\s+expired"
    r"|credentials?\s+(?:expired|invalid|required)"
    r")",
    re.IGNORECASE,
)

# Claude Code's interactive tool-approval UI (Bash/Edit/Write/etc. all use
# the same "Do you want to <verb>...?" + numbered-option shape). Requires
# both the question *and* a following "1. Yes" option within a short window
# so we don't fire on Claude's own prose merely discussing permission.
_APPROVAL_PROMPT_RE = re.compile(
    r"do\s+you\s+want\s+to\s+[^\n?]{0,80}\?"
    r"[\s\S]{0,200}?"
    r"❯?\s*1\.\s*yes\b",
    re.IGNORECASE,
)

# Box-drawing / bullet chrome Claude Code's ink-based UI wraps prompts in.
# Stripped from extracted preview text -- it's rendering noise, not content.
_BOX_CHARS_RE = re.compile(r"[─-╿]")
_LEADING_MARKER_RE = re.compile(r"^[\s❯>*\-]+")

_APPROVAL_PREVIEW_MAX_CHARS = 160

# ── Response-extraction chrome filters ────────────────────────────────────────
#
# Claude Code's CLI renders each turn as: echoed prompt -> zero or more tool
# calls ("⏺ Bash(...)" / "⎿ <result>") -> the model's final prose -> our
# completion marker. Only the final prose is "the response"; tool-call lines
# and the persistent status footer are UI chrome that must not leak into it.

# Tool-invocation and tool-result bullets ("⏺ Bash(git status)", "⎿  (25 lines)").
_TOOL_CALL_PREFIX_RE = re.compile(r"^\s*[⏺⎿]")
# The "thinking"/working spinner glyph Claude Code prefixes its status line
# with while a turn is in progress (frames vary; the glyph itself doesn't).
_SPINNER_PREFIX_RE = re.compile(r"^\s*[✻✽✢✶∴⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]")
# The persistent bottom status/recap line(s): token/cost/context recap and
# the "esc to interrupt" hint shown while a tool call or turn is running.
_STATUS_FOOTER_RE = re.compile(
    r"esc\s+to\s+interrupt"
    r"|[·•]\s*\d+(?:\.\d+)?s\b"
    r"|\btokens?\b\s*[·•]"
    r"|\bcontext\s+(?:left|used|remaining)\b"
    r"|\btotal\s+cost\b"
    r"|\d+\s+tool\s+uses?\b",
    re.IGNORECASE,
)


# ── Public types ──────────────────────────────────────────────────────────────

class BridgeStatus(enum.Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    BUSY = "busy"
    APPROVAL_REQUIRED = "approval_required"
    EXTRACTION_FAILURE = "extraction_failure"
    AUTH_FAILURE = "auth_failure"
    SESSION_MISSING = "session_missing"
    SESSION_DEAD = "session_dead"
    FAILURE = "failure"


@dataclasses.dataclass
class BridgeResult:
    status: BridgeStatus
    response: str = ""
    error: str = ""


@dataclasses.dataclass
class WorkerConfig:
    session: str
    workspace: str
    timeout: float = 300.0
    poll_interval: float = 2.0


# ── Worker registry ───────────────────────────────────────────────────────────

_WORKERS: dict[str, WorkerConfig] = {}


def register_worker(config: WorkerConfig) -> None:
    """Register a named tmux worker configuration."""
    _WORKERS[config.session] = config


def get_worker(session: str) -> Optional[WorkerConfig]:
    """Return the WorkerConfig for *session*, or None."""
    return _WORKERS.get(session)


# Pre-configured worker for Momentum Studio
register_worker(WorkerConfig(session="claude-momentum", workspace="/home/michael/code"))


# ── Per-session locks ─────────────────────────────────────────────────────────

_session_locks: dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()


def _get_session_lock(session: str) -> threading.Lock:
    with _registry_lock:
        if session not in _session_locks:
            _session_locks[session] = threading.Lock()
        return _session_locks[session]


# ── Low-level tmux helpers ────────────────────────────────────────────────────

def _run_tmux(*args: str, timeout: float = 10.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["tmux", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )


def _session_exists(session: str) -> bool:
    result = _run_tmux("has-session", "-t", session)
    return result.returncode == 0


def _pane_alive(session: str) -> bool:
    result = _run_tmux("list-panes", "-t", session, "-F", "#{pane_id}")
    return result.returncode == 0 and bool(result.stdout.strip())


def _send_keys(session: str, text: str) -> bool:
    """Send *text* literally to *session* then press Enter."""
    r = _run_tmux("send-keys", "-t", session, "-l", text)
    if r.returncode != 0:
        return False
    return _run_tmux("send-keys", "-t", session, "Enter").returncode == 0


def _capture_pane(session: str, history_lines: int = 5000) -> str:
    result = _run_tmux(
        "capture-pane", "-p",
        "-t", session,
        "-S", f"-{history_lines}",
        timeout=15.0,
    )
    return result.stdout if result.returncode == 0 else ""


# ── Output sanitisation ───────────────────────────────────────────────────────

def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _redact_credentials(text: str) -> str:
    return _CREDENTIAL_RE.sub("[REDACTED]", text)


def _is_auth_failure(text: str) -> bool:
    return bool(_AUTH_FAILURE_RE.search(_strip_ansi(text)))


def _is_approval_required(text: str) -> bool:
    return bool(_APPROVAL_PROMPT_RE.search(text))


def _extract_approval_preview(new_content: str) -> Optional[str]:
    """Best-effort, sanitised preview of what Claude Code is asking to run.

    Only looks at *new_content* -- pane output produced since this specific
    prompt was submitted -- and only at the lines immediately above the
    approval question, never at unrelated pane history. The scan stops the
    instant it reaches the echoed, just-submitted prompt (identified by our
    own instrumentation text or a leading "> ") so the preview can never
    include the prompt we sent or its embedded completion marker -- only
    genuine approval-box content. Returns None when nothing meaningful can
    be extracted (e.g. the box only contains decoration).
    """
    match = _APPROVAL_PROMPT_RE.search(new_content)
    if match is None:
        return None

    lines = new_content.splitlines()
    question_line_idx = new_content[: match.start()].count("\n")

    candidates: list[str] = []
    floor = max(-1, question_line_idx - 8)
    for idx in range(question_line_idx - 1, floor, -1):
        raw_line = lines[idx]
        if "output exactly on its own line" in raw_line.lower() or raw_line.lstrip().startswith(">"):
            break
        cleaned = _BOX_CHARS_RE.sub("", raw_line)
        cleaned = _LEADING_MARKER_RE.sub("", cleaned).strip()
        if not cleaned or "do you want to" in cleaned.lower():
            continue
        candidates.append(cleaned)
    candidates.reverse()

    if not candidates:
        return None

    preview = _redact_credentials(" ".join(candidates))
    if len(preview) > _APPROVAL_PREVIEW_MAX_CHARS:
        preview = preview[: _APPROVAL_PREVIEW_MAX_CHARS - 3] + "..."
    return preview


def _is_chrome_line(line: str) -> bool:
    """True if *line* is Claude Code UI chrome, not response content.

    Covers tool-call bullets, the working spinner, and the persistent
    status footer (tokens/cost/context/"esc to interrupt"). Blank lines are
    handled separately by the caller so interior blank lines (paragraph
    breaks) can be preserved.
    """
    stripped = line.strip()
    if not stripped:
        return False
    if _TOOL_CALL_PREFIX_RE.match(line) or _SPINNER_PREFIX_RE.match(line):
        return True
    if _STATUS_FOOTER_RE.search(stripped):
        return True
    # A line made entirely of box-drawing/rule characters is decoration
    # (e.g. a horizontal divider), not content.
    if not _BOX_CHARS_RE.sub("", stripped).strip("-—_= "):
        return True
    return False


def _clean_response_lines(lines: list[str]) -> str:
    """Drop chrome lines, then trim leading/trailing blanks.

    Interior blank lines are preserved so multi-paragraph and bulleted
    responses keep their formatting -- only chrome lines are removed, never
    collapsed/re-flowed.
    """
    kept = [line for line in lines if not _is_chrome_line(line)]
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def _find_echo_line_idx(
    clean_lines: list[str], done_idx: int, done_marker: str
) -> Optional[int]:
    """Find the line index of the echoed, just-submitted prompt's
    completion-marker instruction, searching clean_lines[:done_idx].

    A real prompt is one long logical line (newlines are collapsed to
    spaces before sending -- see _run_prompt), and Claude Code's terminal
    UI soft-wraps it across as many physical pane rows as its length
    requires. For a long prompt (the common case for real delegated
    tasks, as opposed to short test fixtures) the wrap boundary can fall
    *inside* the marker text itself, splitting it across two physical
    rows so that no single row contains it as a contiguous substring --
    a naive per-line scan then finds no echo at all, which is exactly the
    reported "completion marker seen but no response extracted" bug:
    with no echo found, extraction fell back to the pre-send line count,
    which can be stale enough (e.g. earlier tool output in the same
    session having since collapsed) to land at or past the marker line
    and produce a guaranteed-empty slice.

    Soft-wrap only breaks the character stream; it never inserts or drops
    characters. So concatenating consecutive rows with no separator
    reconstructs the original unwrapped text, and searching *that* for
    done_marker -- then mapping the match back to whichever row its last
    character landed on -- finds the echo regardless of where it happened
    to wrap. (This does mean two logically-unrelated adjacent rows could
    in principle be concatenated into a false match; the marker is a
    32-hex-char random UUID suffix, so this is not a practical risk.)

    Skips over a line that is *itself* an exact, standalone completion
    marker (rather than treating it as the echo) the same way the
    original per-line scan did -- an earlier duplicate "done" line is the
    marker being echoed back as itself, not the prompt that asked for it.
    """
    search_lines = clean_lines[:done_idx]
    if not search_lines:
        return None

    offsets: list[int] = []
    pos = 0
    for line in search_lines:
        offsets.append(pos)
        pos += len(line)
    joined = "".join(search_lines)

    search_end = len(joined)
    while True:
        match_start = joined.rfind(done_marker, 0, search_end)
        if match_start == -1:
            return None
        match_end = match_start + len(done_marker) - 1
        line_idx = bisect.bisect_right(offsets, match_end) - 1
        if clean_lines[line_idx].strip() != done_marker:
            return line_idx
        search_end = match_start


def _extract_response(pre_line_count: int, post_raw: str, done_marker: str) -> Optional[str]:
    """Extract Claude's response immediately preceding the completion marker.

    Returns None if the marker hasn't appeared yet (keep polling). Once
    found, always returns a string -- possibly empty if nothing extractable
    precedes the marker; the caller maps that to EXTRACTION_FAILURE rather
    than reporting a silent empty success.

    The marker must appear *alone* on its line (after stripping leading/
    trailing whitespace) to count as the completion line. A plain substring
    match would also match the echoed, just-submitted prompt -- which
    necessarily contains the marker text, since the prompt asks Claude to
    repeat it back -- causing a false "done" the moment the prompt is
    echoed, before Claude has produced any real output.

    The start of the response is anchored on that same echoed-prompt line,
    found via _find_echo_line_idx -- which tolerates the marker being
    split across a terminal soft-wrap boundary, see its docstring --
    rather than *pre_line_count* alone. Claude Code's TUI repaints its
    whole viewport per frame and can settle to a different total line
    count than the pre-send snapshot (e.g. a prior turn's tool output
    collapsing once no longer active) -- a stale count-based boundary can
    then land at or past the marker line itself, producing a
    guaranteed-empty (start >= stop) slice: a detected-but-empty "success"
    for a turn that actually has real content.

    When no echo line is found at all (e.g. Claude Code's TUI collapsed a
    very long pasted prompt to a placeholder that never renders the
    marker text anywhere, or simple/synthetic pane content with no
    embedded marker substring), *pre_line_count* is used as a fallback
    floor -- but only when it doesn't exceed the marker line itself
    (pre_line_count == done_idx is a legitimate "nothing new was added"
    case, not staleness). A pre_line_count *past* done_idx is provably
    stale (no legitimate content can start after the marker that
    terminates it), so it is discarded in favour of 0 (the start of the
    captured pane) rather than guaranteeing an empty result.
    """
    clean_lines = _strip_ansi(post_raw).splitlines()

    done_idx = None
    for i, line in enumerate(clean_lines):
        if line.strip() == done_marker:
            done_idx = i
            break

    if done_idx is None:
        return None

    echo_idx = _find_echo_line_idx(clean_lines, done_idx, done_marker)

    if echo_idx is not None:
        start_idx = echo_idx + 1
    else:
        start_idx = pre_line_count if pre_line_count <= done_idx else 0

    response = _clean_response_lines(clean_lines[start_idx:done_idx])
    return _redact_credentials(response)


# ── Public API ────────────────────────────────────────────────────────────────

def submit_prompt(
    session: str,
    prompt: str,
    *,
    timeout: Optional[float] = None,
    poll_interval: Optional[float] = None,
) -> BridgeResult:
    """Submit *prompt* to the Claude Code session *session*.

    Parameters
    ----------
    session:
        Name of the tmux session running Claude Code.
    prompt:
        Prompt text.  Newlines are collapsed to spaces before sending.
    timeout:
        Seconds to wait for a response.  Falls back to the worker's
        configured timeout (default 300 s).
    poll_interval:
        Seconds between pane captures.  Falls back to the worker's
        configured poll interval (default 2 s).

    Returns
    -------
    BridgeResult with status one of:

    SUCCESS           Response captured successfully.
    TIMEOUT           Completion marker not seen before deadline.
    BUSY              Another call already holds the per-session lock.
    APPROVAL_REQUIRED Claude Code is blocked on an interactive approval
                      prompt ("Do you want to proceed?"). Detected as soon
                      as it appears, not automatically approved.
    EXTRACTION_FAILURE The completion marker was seen, but no response text
                      could be extracted from the pane -- never reported as
                      a silent empty SUCCESS.
    AUTH_FAILURE      Pane shows authentication or OAuth error at timeout.
    SESSION_MISSING   Named tmux session does not exist.
    SESSION_DEAD      Session exists but has no responsive pane.
    FAILURE           Subprocess error or unexpected condition.
    """
    worker = _WORKERS.get(session) or WorkerConfig(session=session, workspace="")
    eff_timeout = timeout if timeout is not None else worker.timeout
    eff_poll = poll_interval if poll_interval is not None else worker.poll_interval

    # Verify session exists
    try:
        if not _session_exists(session):
            return BridgeResult(
                status=BridgeStatus.SESSION_MISSING,
                error=f"tmux session {session!r} not found",
            )
    except subprocess.TimeoutExpired:
        return BridgeResult(status=BridgeStatus.FAILURE, error="has-session timed out")
    except FileNotFoundError:
        return BridgeResult(status=BridgeStatus.FAILURE, error="tmux not found on PATH")
    except OSError as exc:
        return BridgeResult(status=BridgeStatus.FAILURE, error=str(exc))

    # Verify pane is alive
    try:
        if not _pane_alive(session):
            return BridgeResult(
                status=BridgeStatus.SESSION_DEAD,
                error=f"tmux session {session!r} has no live pane",
            )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return BridgeResult(status=BridgeStatus.FAILURE, error=str(exc))

    # Acquire per-session lock (non-blocking — BUSY if already held)
    lock = _get_session_lock(session)
    if not lock.acquire(blocking=False):
        return BridgeResult(
            status=BridgeStatus.BUSY,
            error=f"Session {session!r} is already handling a prompt",
        )
    try:
        return _run_prompt(
            session=session,
            prompt=prompt,
            timeout=eff_timeout,
            poll_interval=eff_poll,
        )
    finally:
        lock.release()


def _run_prompt(
    *,
    session: str,
    prompt: str,
    timeout: float,
    poll_interval: float,
) -> BridgeResult:
    run_id = uuid.uuid4().hex
    done_marker = f"HERMES_BRIDGE_DONE_{run_id}"

    # Collapse newlines — tmux send-keys treats each \n as Enter
    flat_prompt = " ".join(prompt.splitlines())
    instrumented = (
        f"{flat_prompt} "
        f"(When your response is complete, output exactly on its own line: {done_marker})"
    )

    # Snapshot pane before sending so we can slice out only new content
    try:
        pre_raw = _capture_pane(session)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return BridgeResult(status=BridgeStatus.FAILURE, error=str(exc))
    pre_line_count = len(_strip_ansi(pre_raw).splitlines())

    logger.debug(
        "bridge: sending to %r run_id=%s pre_lines=%d",
        session, run_id, pre_line_count,
    )

    try:
        if not _send_keys(session, instrumented):
            return BridgeResult(status=BridgeStatus.FAILURE, error="tmux send-keys failed")
    except (subprocess.TimeoutExpired, OSError) as exc:
        return BridgeResult(status=BridgeStatus.FAILURE, error=str(exc))

    # Poll for completion marker
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(poll_interval)
        try:
            post_raw = _capture_pane(session)
        except (subprocess.TimeoutExpired, OSError) as exc:
            return BridgeResult(status=BridgeStatus.FAILURE, error=str(exc))

        response = _extract_response(pre_line_count, post_raw, done_marker)
        if response is not None:
            if not response.strip():
                logger.warning(
                    "bridge: completion marker seen but no response extracted run_id=%s",
                    run_id,
                )
                return BridgeResult(
                    status=BridgeStatus.EXTRACTION_FAILURE,
                    error=(
                        f"Claude Code finished in tmux session {session!r}, but no "
                        "response text could be extracted from the pane."
                    ),
                )
            logger.debug("bridge: done run_id=%s chars=%d", run_id, len(response))
            return BridgeResult(status=BridgeStatus.SUCCESS, response=response)

        # No completion marker yet -- check whether Claude Code is blocked
        # on an interactive approval prompt. Checked every poll (not just
        # at timeout) so a prompt nobody is watching doesn't hold the
        # per-session lock for the full timeout. Only ever looks at content
        # produced since this submission (post pre_line_count), same as the
        # auth-failure check below, so unrelated pane history is never
        # inspected or returned.
        new_content = "\n".join(_strip_ansi(post_raw).splitlines()[pre_line_count:])
        if _is_approval_required(new_content):
            preview = _extract_approval_preview(new_content)
            message = f"Claude Code requires approval in tmux session {session!r}."
            if preview:
                message += f" Pending: {preview}"
            logger.debug("bridge: approval required run_id=%s", run_id)
            return BridgeResult(status=BridgeStatus.APPROVAL_REQUIRED, error=message)

    # Timeout — check whether pane shows an auth error
    try:
        final_raw = _capture_pane(session)
    except (subprocess.TimeoutExpired, OSError):
        final_raw = ""

    new_content = "\n".join(_strip_ansi(final_raw).splitlines()[pre_line_count:])
    if _is_auth_failure(new_content):
        return BridgeResult(
            status=BridgeStatus.AUTH_FAILURE,
            error="Authentication or OAuth failure detected in pane output",
        )

    return BridgeResult(
        status=BridgeStatus.TIMEOUT,
        error=f"Completion marker not seen within {timeout:.0f}s",
    )
