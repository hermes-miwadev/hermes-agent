"""Tests for agent.claude_code_tmux_bridge.

All tmux subprocess calls are mocked; no real tmux session is required.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

import agent.claude_code_tmux_bridge as bridge
from agent.claude_code_tmux_bridge import (
    BridgeResult,
    BridgeStatus,
    WorkerConfig,
    _clean_response_lines,
    _extract_approval_preview,
    _extract_response,
    _is_approval_required,
    _is_auth_failure,
    _is_chrome_line,
    _redact_credentials,
    _strip_ansi,
)

FAKE_RUN_ID = "cafebabe" * 4  # 32 hex chars


@pytest.fixture(autouse=True)
def _clear_session_locks():
    bridge._session_locks.clear()
    yield
    bridge._session_locks.clear()


# ── Pure helper unit tests ────────────────────────────────────────────────────

class TestStripAnsi:
    def test_removes_sgr_codes(self):
        assert _strip_ansi("\x1b[32mgreen\x1b[0m") == "green"

    def test_removes_osc_sequences(self):
        assert _strip_ansi("\x1b]0;title\x07text") == "text"

    def test_removes_carriage_return(self):
        assert _strip_ansi("a\rb") == "ab"

    def test_passthrough_plain_text(self):
        assert _strip_ansi("hello world") == "hello world"

    def test_empty_string(self):
        assert _strip_ansi("") == ""


class TestRedactCredentials:
    def test_redacts_bearer_token(self):
        text = "Authorization: Bearer sk-abcdef1234567890abcdef1234567890"
        out = _redact_credentials(text)
        assert "[REDACTED]" in out
        assert "sk-abcdef" not in out

    def test_redacts_api_key_value(self):
        text = "api_key=supersecretlongvalue12345"
        out = _redact_credentials(text)
        assert "[REDACTED]" in out

    def test_leaves_short_values_alone(self):
        # Values shorter than 16 chars are not credential-shaped
        assert _redact_credentials("token: abc") == "token: abc"

    def test_leaves_unrelated_text_alone(self):
        text = "The answer is 42."
        assert _redact_credentials(text) == text


class TestIsAuthFailure:
    @pytest.mark.parametrize("text", [
        "Authentication required",
        "OAuth token expired",
        "Please sign in to continue",
        "Unauthorized",
        "Session expired",
        "Login required",
        "credentials expired",
        "Please authenticate",
    ])
    def test_detects_known_patterns(self, text):
        assert _is_auth_failure(text)

    def test_ignores_plain_text(self):
        assert not _is_auth_failure("The function returns a boolean value.")

    def test_strips_ansi_before_matching(self):
        assert _is_auth_failure("\x1b[31mAuthentication required\x1b[0m")

    def test_case_insensitive(self):
        assert _is_auth_failure("AUTHENTICATION REQUIRED")


class TestExtractResponse:
    def test_extracts_lines_after_pre_count_up_to_marker(self):
        done = f"HERMES_BRIDGE_DONE_{FAKE_RUN_ID}"
        pane = "\n".join([
            "old line 1",
            "old line 2",
            "The answer is 42.",
            done,
            "stuff after",
        ])
        result = _extract_response(pre_line_count=2, post_raw=pane, done_marker=done)
        assert result is not None
        assert "The answer is 42." in result
        assert "old line" not in result
        assert done not in result

    def test_returns_none_when_marker_absent(self):
        assert _extract_response(0, "line 1\nline 2", "NO_MARKER") is None

    def test_strips_ansi_from_extracted_text(self):
        done = f"HERMES_BRIDGE_DONE_{FAKE_RUN_ID}"
        pane = f"preamble\n\x1b[32mgreen text\x1b[0m\n{done}"
        result = _extract_response(1, pane, done)
        assert result is not None
        assert "\x1b" not in result
        assert "green text" in result

    def test_returns_empty_string_when_nothing_between(self):
        done = f"HERMES_BRIDGE_DONE_{FAKE_RUN_ID}"
        pane = f"pre\n{done}"
        result = _extract_response(1, pane, done)
        assert result == ""

    def test_redacts_credential_in_response(self):
        done = f"HERMES_BRIDGE_DONE_{FAKE_RUN_ID}"
        pane = f"pre\nBearer sk-supersecretlongtokenvalue1234567\n{done}"
        result = _extract_response(1, pane, done)
        assert result is not None
        assert "[REDACTED]" in result
        assert "sk-supersecret" not in result

    # ── Regression: the marker must appear ALONE on its line ─────────────────
    #
    # A plain substring match also matches the echoed, just-submitted prompt
    # (which necessarily contains the marker text, since the instrumented
    # prompt asks Claude to repeat it back). That produced a false empty
    # "success" the instant the prompt was echoed, before Claude had done
    # anything -- including before it could even reach an approval prompt.

    def test_ignores_marker_embedded_in_longer_echoed_prompt_line(self):
        done = f"HERMES_BRIDGE_DONE_{FAKE_RUN_ID}"
        pane = (
            "prior history\n"
            f"> delete the tmp dir (When your response is complete, output exactly on its own line: {done})\n"
        )
        # No standalone marker line yet -- Claude hasn't finished (or has
        # paused at an approval prompt). Must NOT be treated as done.
        assert _extract_response(pre_line_count=1, post_raw=pane, done_marker=done) is None

    def test_matches_exact_marker_even_with_surrounding_whitespace(self):
        done = f"HERMES_BRIDGE_DONE_{FAKE_RUN_ID}"
        pane = f"pre\nThe answer is 42.\n   {done}   \n"
        result = _extract_response(1, pane, done)
        assert result is not None
        assert "The answer is 42." in result

    def test_finds_real_standalone_marker_not_earlier_embedded_occurrence(self):
        done = f"HERMES_BRIDGE_DONE_{FAKE_RUN_ID}"
        pane = "\n".join([
            "prior history",
            f"> do the thing (When your response is complete, output exactly on its own line: {done})",
            "The answer is 42.",
            done,
        ])
        result = _extract_response(pre_line_count=1, post_raw=pane, done_marker=done)
        assert result is not None
        # With a naive substring match, done_idx would land on the echoed
        # prompt line (it embeds the marker text too) and the response
        # would be truncated to "" before ever reaching the real answer.
        # Finding the real standalone marker instead means the response is
        # correctly anchored to start right after the echoed prompt.
        assert result == "The answer is 42."


# ── Chrome-line detection ──────────────────────────────────────────────────────

class TestIsChromeLine:
    def test_tool_call_bullet_is_chrome(self):
        assert _is_chrome_line("⏺ Bash(git status)") is True

    def test_tool_result_bullet_is_chrome(self):
        assert _is_chrome_line("  ⎿  feat/virtual-services-astro-migration") is True

    def test_spinner_line_is_chrome(self):
        assert _is_chrome_line("✻ Herding cats… (esc to interrupt · 12s)") is True

    def test_status_footer_tokens_is_chrome(self):
        assert _is_chrome_line("2 tool uses · 15.2s · ↑ 1.2k tokens") is True

    def test_esc_to_interrupt_is_chrome(self):
        assert _is_chrome_line("  (esc to interrupt)") is True

    def test_box_border_only_line_is_chrome(self):
        assert _is_chrome_line("╭──────────────────────────────╮") is True
        assert _is_chrome_line("──────────────────────") is True

    def test_blank_line_is_not_chrome(self):
        assert _is_chrome_line("") is False
        assert _is_chrome_line("   ") is False

    def test_prose_with_markdown_bullet_is_not_chrome(self):
        assert _is_chrome_line("- Branch: main") is False
        assert _is_chrome_line("1. First step") is False

    def test_plain_prose_mentioning_tools_is_not_chrome(self):
        assert _is_chrome_line("I used a couple of tools to check this.") is False


class TestCleanResponseLines:
    def test_removes_tool_call_chrome_keeps_prose(self):
        lines = [
            "⏺ Bash(git branch --show-current)",
            "  ⎿  feat/virtual-services-astro-migration",
            "",
            "⏺ Bash(git status)",
            "  ⎿  (25 lines)",
            "",
            "Current branch: feat/virtual-services-astro-migration.",
        ]
        assert _clean_response_lines(lines) == "Current branch: feat/virtual-services-astro-migration."

    def test_preserves_interior_blank_lines_and_bullets(self):
        lines = [
            "Here's what I found:",
            "",
            "- Branch: main",
            "- Status: clean",
            "",
            "Let me know if you need more detail.",
        ]
        assert _clean_response_lines(lines) == "\n".join(lines)

    def test_trims_leading_and_trailing_blank_lines(self):
        lines = ["", "", "The answer is 42.", "", ""]
        assert _clean_response_lines(lines) == "The answer is 42."

    def test_empty_input_returns_empty_string(self):
        assert _clean_response_lines([]) == ""

    def test_all_chrome_returns_empty_string(self):
        lines = ["⏺ Bash(rm -rf /tmp/foo)", "  ⎿  (no output)", "╰──────────────────╯"]
        assert _clean_response_lines(lines) == ""


# ── Realistic Claude Code transcript scenarios ────────────────────────────────
#
# Regression coverage for the reported bug: a real multi-tool-call turn was
# detected as complete (marker found) but Discord received an empty
# response. Root cause: the old pre_line_count-slice boundary went stale
# once Claude Code's TUI repainted the pane (a prior turn's tool output can
# collapse to a shorter summary once settled), landing the slice's start
# past the marker entirely.

def _realistic_transcript(done_marker: str, *, prior_line_count: int = 20) -> tuple[str, int]:
    """Build a (post_pane, pre_line_count) pair modelling a real Claude Code
    turn: a prior turn's tool output that has since collapsed to a shorter
    summary, an echoed prompt embedding *done_marker*, two tool calls, a
    final multi-line prose answer, and the standalone marker line."""
    post_pane = "\n".join([
        "old expanded tool output (collapsed summary)",
        "",
        f"> (Work only within /repo.) Report status. "
        f"(When your response is complete, output exactly on its own line: {done_marker})",
        "",
        "⏺ Bash(git branch --show-current)",
        "  ⎿  feat/virtual-services-astro-migration",
        "",
        "⏺ Bash(git status)",
        "  ⎿  (25 lines)",
        "",
        "Current branch: feat/virtual-services-astro-migration.",
        "",
        "- Working tree is otherwise clean",
        "- 5 untracked files present",
        "",
        "No files were modified.",
        "",
        done_marker,
    ])
    # A stale pre-send snapshot with MORE lines than the settled/repainted
    # pane ends up with before the marker -- this is what desyncs a naive
    # count-based boundary.
    return post_pane, prior_line_count


class TestRealisticResponseExtraction:
    def test_reproduces_and_fixes_reported_empty_response_bug(self):
        done = f"HERMES_BRIDGE_DONE_{FAKE_RUN_ID}"
        post_pane, pre_line_count = _realistic_transcript(done)

        result = _extract_response(pre_line_count, post_pane, done)

        assert result is not None
        assert result != ""
        assert "Current branch: feat/virtual-services-astro-migration." in result

    def test_multiline_response_with_bullets_preserved(self):
        done = f"HERMES_BRIDGE_DONE_{FAKE_RUN_ID}"
        post_pane, pre_line_count = _realistic_transcript(done)

        result = _extract_response(pre_line_count, post_pane, done)

        assert result == (
            "Current branch: feat/virtual-services-astro-migration.\n"
            "\n"
            "- Working tree is otherwise clean\n"
            "- 5 untracked files present\n"
            "\n"
            "No files were modified."
        )

    def test_echoed_prompt_excluded_even_with_tool_calls_between(self):
        done = f"HERMES_BRIDGE_DONE_{FAKE_RUN_ID}"
        post_pane, pre_line_count = _realistic_transcript(done)

        result = _extract_response(pre_line_count, post_pane, done)

        assert "Report status" not in result
        assert "output exactly on its own line" not in result
        assert done not in result

    def test_tool_progress_lines_excluded(self):
        done = f"HERMES_BRIDGE_DONE_{FAKE_RUN_ID}"
        post_pane, pre_line_count = _realistic_transcript(done)

        result = _extract_response(pre_line_count, post_pane, done)

        assert "⏺" not in result
        assert "⎿" not in result
        assert "git branch --show-current" not in result
        assert "git status" not in result

    def test_marker_alone_on_own_line_still_required(self):
        # The echo embeds the marker as a substring; that alone must not
        # be mistaken for completion.
        done = f"HERMES_BRIDGE_DONE_{FAKE_RUN_ID}"
        pane = (
            "prior history\n"
            f"> do the thing (When your response is complete, output exactly on its own line: {done})\n"
            "⏺ Bash(sleep 1)\n"
        )
        assert _extract_response(pre_line_count=1, post_raw=pane, done_marker=done) is None

    def test_prior_pane_history_excluded_before_echoed_prompt(self):
        done = f"HERMES_BRIDGE_DONE_{FAKE_RUN_ID}"
        post_pane, pre_line_count = _realistic_transcript(done)

        result = _extract_response(pre_line_count, post_pane, done)

        assert "old expanded tool output" not in result

    def test_no_valid_response_before_marker_yields_empty_for_caller(self):
        """Only chrome precedes the marker -- extraction must not fabricate
        content; the empty result signals the caller to report
        EXTRACTION_FAILURE rather than a silent empty success."""
        done = f"HERMES_BRIDGE_DONE_{FAKE_RUN_ID}"
        pane = "\n".join([
            f"> do the thing (When your response is complete, output exactly on its own line: {done})",
            "⏺ Bash(git commit -am wip)",
            "  ⎿  (no output)",
            "╰──────────────────────────────╯",
            done,
        ])
        result = _extract_response(pre_line_count=0, post_raw=pane, done_marker=done)
        assert result == ""

    def test_terminal_control_sequences_stripped_from_realistic_transcript(self):
        done = f"HERMES_BRIDGE_DONE_{FAKE_RUN_ID}"
        post_pane, pre_line_count = _realistic_transcript(done)
        # Interleave ANSI SGR colour codes, an OSC window-title sequence,
        # and bare carriage returns throughout, as a real terminal-emulated
        # pane capture would contain.
        noisy = (
            "\x1b]0;claude\x07"
            + post_pane.replace(
                "Current branch:", "\x1b[32mCurrent branch:\x1b[0m"
            ).replace("\n", "\r\n")
        )

        result = _extract_response(pre_line_count, noisy, done)

        assert result is not None
        assert "\x1b" not in result
        assert "\r" not in result
        assert "Current branch: feat/virtual-services-astro-migration." in result


class TestExtractionFailureIntegration:
    def test_returns_extraction_failure_when_marker_found_but_nothing_extractable(self, monkeypatch):
        run_id = FAKE_RUN_ID
        done = f"HERMES_BRIDGE_DONE_{run_id}"
        pre = ""
        post = "\n".join([
            f"> do the thing (When your response is complete, output exactly on its own line: {done})",
            "⏺ Bash(git commit -am wip)",
            "  ⎿  (no output)",
            done,
        ])

        monkeypatch.setattr(bridge.uuid, "uuid4", lambda: MagicMock(hex=run_id))
        _setup_session(monkeypatch)
        _setup_io(monkeypatch, pre_pane=pre, post_pane=post)
        monkeypatch.setattr(bridge.time, "sleep", lambda x: None)

        result = bridge.submit_prompt("claude-momentum", "do the thing", timeout=10.0)

        assert result.status == BridgeStatus.EXTRACTION_FAILURE
        assert result.status != BridgeStatus.SUCCESS
        assert result.response == ""
        assert "claude-momentum" in result.error

    def test_reproduces_reported_bug_end_to_end_via_submit_prompt(self, monkeypatch):
        run_id = FAKE_RUN_ID
        done = f"HERMES_BRIDGE_DONE_{run_id}"
        post_pane, pre_line_count = _realistic_transcript(done)
        pre_pane = "\n".join([f"old expanded tool output line {i}" for i in range(pre_line_count)]) + "\n"

        monkeypatch.setattr(bridge.uuid, "uuid4", lambda: MagicMock(hex=run_id))
        _setup_session(monkeypatch)
        _setup_io(monkeypatch, pre_pane=pre_pane, post_pane=post_pane)
        monkeypatch.setattr(bridge.time, "sleep", lambda x: None)

        result = bridge.submit_prompt("claude-momentum", "Report status.", timeout=10.0)

        assert result.status == BridgeStatus.SUCCESS
        assert "Current branch: feat/virtual-services-astro-migration." in result.response

    def test_lock_released_after_extraction_failure(self, monkeypatch):
        run_id = FAKE_RUN_ID
        done = f"HERMES_BRIDGE_DONE_{run_id}"
        pre = ""
        post = f"⏺ Bash(noop)\n  ⎿  (no output)\n{done}\n"

        monkeypatch.setattr(bridge.uuid, "uuid4", lambda: MagicMock(hex=run_id))
        _setup_session(monkeypatch)
        _setup_io(monkeypatch, pre_pane=pre, post_pane=post)
        monkeypatch.setattr(bridge.time, "sleep", lambda x: None)

        result = bridge.submit_prompt("claude-momentum", "noop", timeout=10.0)
        assert result.status == BridgeStatus.EXTRACTION_FAILURE

        lock = bridge._get_session_lock("claude-momentum")
        assert lock.acquire(blocking=False)
        lock.release()


# ── Interactive approval-prompt detection ─────────────────────────────────────

class TestApprovalRequiredDetection:
    def test_detects_do_you_want_to_proceed_prompt(self):
        text = "Do you want to proceed?\n1. Yes\n2. No\n"
        assert _is_approval_required(text) is True

    def test_detects_prompt_with_ink_ui_chevron_and_box(self):
        text = (
            "╭──────────────────────╮\n"
            "│ Bash command          │\n"
            "│   rm -rf /tmp/foo     │\n"
            "│ Do you want to proceed? │\n"
            "│ ❯ 1. Yes              │\n"
            "│   2. No               │\n"
            "╰──────────────────────╯\n"
        )
        assert _is_approval_required(text) is True

    def test_detects_edit_approval_wording_variant(self):
        text = "Do you want to make this edit to config.py?\n❯ 1. Yes\n  2. No, and tell Claude what to do differently\n"
        assert _is_approval_required(text) is True

    def test_no_false_positive_on_unrelated_text_mentioning_yes(self):
        text = "The test passed. 1. Yes it did. 2. No issues found.\n"
        assert _is_approval_required(text) is False

    def test_no_false_positive_on_question_without_options(self):
        text = "Do you want to proceed? I think we should discuss this further first.\n"
        assert _is_approval_required(text) is False

    def test_no_false_positive_on_plain_response_text(self):
        text = "The answer is 42.\nHERMES_BRIDGE_DONE_abc123\n"
        assert _is_approval_required(text) is False


class TestApprovalPreviewExtraction:
    def _approval_pane(self, echoed_prompt_line: str) -> str:
        return (
            "prior history\n"
            f"{echoed_prompt_line}\n"
            "╭──────────────────────────────╮\n"
            "│ Bash command                  │\n"
            "│                                │\n"
            "│   rm -rf /tmp/foo              │\n"
            "│                                │\n"
            "│ Do you want to proceed?       │\n"
            "│ ❯ 1. Yes                      │\n"
            "│   2. No                       │\n"
            "╰──────────────────────────────╯\n"
        )

    def test_extracts_command_preview_from_approval_box(self):
        done = f"HERMES_BRIDGE_DONE_{FAKE_RUN_ID}"
        pane = self._approval_pane(
            f"> delete the tmp dir (When your response is complete, output exactly on its own line: {done})"
        )
        new_content = "\n".join(_strip_ansi(pane).splitlines()[1:])
        preview = _extract_approval_preview(new_content)
        assert preview is not None
        assert "rm -rf /tmp/foo" in preview

    def test_preview_never_includes_echoed_prompt_or_marker(self):
        done = f"HERMES_BRIDGE_DONE_{FAKE_RUN_ID}"
        pane = self._approval_pane(
            f"> delete the tmp dir (When your response is complete, output exactly on its own line: {done})"
        )
        new_content = "\n".join(_strip_ansi(pane).splitlines()[1:])
        preview = _extract_approval_preview(new_content)
        assert preview is not None
        assert done not in preview
        assert "own line" not in preview
        assert "delete the tmp dir" not in preview

    def test_preview_redacts_credentials(self):
        pane = self._approval_pane("> rotate the key").replace(
            "rm -rf /tmp/foo", "curl -H 'Authorization: Bearer sk-supersecretlongtokenvalue1234567'"
        )
        new_content = "\n".join(_strip_ansi(pane).splitlines()[1:])
        preview = _extract_approval_preview(new_content)
        assert preview is not None
        assert "[REDACTED]" in preview
        assert "sk-supersecret" not in preview

    def test_returns_none_when_no_approval_prompt_present(self):
        assert _extract_approval_preview("just a normal response\n") is None


# ── Integration tests with mocked tmux helpers ────────────────────────────────

def _setup_session(monkeypatch, *, exists: bool = True, alive: bool = True) -> None:
    monkeypatch.setattr(bridge, "_session_exists", lambda s: exists)
    monkeypatch.setattr(bridge, "_pane_alive", lambda s: alive)


def _setup_io(
    monkeypatch,
    *,
    pre_pane: str = "",
    post_pane: str = "",
    send_ok: bool = True,
) -> None:
    """Mock _capture_pane to return pre_pane on first call, post_pane thereafter."""
    call_count = [0]

    def _mock_capture(s, **kw):
        call_count[0] += 1
        return pre_pane if call_count[0] == 1 else post_pane

    monkeypatch.setattr(bridge, "_capture_pane", _mock_capture)
    monkeypatch.setattr(bridge, "_send_keys", lambda s, t: send_ok)


class TestSessionVerification:
    def test_missing_session_returns_session_missing(self, monkeypatch):
        _setup_session(monkeypatch, exists=False)
        result = bridge.submit_prompt("claude-momentum", "hello")
        assert result.status == BridgeStatus.SESSION_MISSING
        assert "claude-momentum" in result.error

    def test_dead_session_returns_session_dead(self, monkeypatch):
        _setup_session(monkeypatch, exists=True, alive=False)
        result = bridge.submit_prompt("claude-momentum", "hello")
        assert result.status == BridgeStatus.SESSION_DEAD

    def test_tmux_not_found_returns_failure(self, monkeypatch):
        def _raise(s):
            raise FileNotFoundError("tmux")
        monkeypatch.setattr(bridge, "_session_exists", _raise)
        result = bridge.submit_prompt("claude-momentum", "hello")
        assert result.status == BridgeStatus.FAILURE
        assert "not found" in result.error.lower()

    def test_os_error_returns_failure(self, monkeypatch):
        def _raise(s):
            raise OSError("permission denied")
        monkeypatch.setattr(bridge, "_session_exists", _raise)
        result = bridge.submit_prompt("claude-momentum", "hello")
        assert result.status == BridgeStatus.FAILURE

    def test_unknown_session_uses_default_config(self, monkeypatch):
        """Ad-hoc sessions not in _WORKERS are accepted with defaults."""
        _setup_session(monkeypatch, exists=False)
        result = bridge.submit_prompt("some-other-session", "hello")
        assert result.status == BridgeStatus.SESSION_MISSING


class TestLocking:
    def test_busy_when_lock_already_held(self, monkeypatch):
        _setup_session(monkeypatch)
        lock = bridge._get_session_lock("claude-momentum")
        lock.acquire()
        try:
            result = bridge.submit_prompt("claude-momentum", "hello")
            assert result.status == BridgeStatus.BUSY
            assert "claude-momentum" in result.error
        finally:
            lock.release()

    def test_lock_released_after_successful_call(self, monkeypatch):
        run_id = FAKE_RUN_ID
        done = f"HERMES_BRIDGE_DONE_{run_id}"
        pre = "history\n"
        post = f"history\nresponse line\n{done}\n"

        monkeypatch.setattr(bridge.uuid, "uuid4", lambda: MagicMock(hex=run_id))
        _setup_session(monkeypatch)
        _setup_io(monkeypatch, pre_pane=pre, post_pane=post)
        monkeypatch.setattr(bridge.time, "sleep", lambda x: None)

        result = bridge.submit_prompt("claude-momentum", "hi", timeout=30.0)
        assert result.status == BridgeStatus.SUCCESS

        # Lock must be free after the call
        lock = bridge._get_session_lock("claude-momentum")
        assert lock.acquire(blocking=False)
        lock.release()

    def test_lock_released_after_send_failure(self, monkeypatch):
        _setup_session(monkeypatch)
        monkeypatch.setattr(bridge, "_capture_pane", lambda s, **kw: "")
        monkeypatch.setattr(bridge, "_send_keys", lambda s, t: False)

        result = bridge.submit_prompt("claude-momentum", "hello", timeout=5.0)
        assert result.status == BridgeStatus.FAILURE

        lock = bridge._get_session_lock("claude-momentum")
        assert lock.acquire(blocking=False)
        lock.release()

    def test_separate_sessions_use_separate_locks(self, monkeypatch):
        lock_a = bridge._get_session_lock("session-a")
        lock_b = bridge._get_session_lock("session-b")
        assert lock_a is not lock_b

        lock_a.acquire()
        try:
            # session-b lock is independent
            assert lock_b.acquire(blocking=False)
            lock_b.release()
        finally:
            lock_a.release()


class TestSuccessPath:
    def test_returns_success_with_response(self, monkeypatch):
        run_id = FAKE_RUN_ID
        done = f"HERMES_BRIDGE_DONE_{run_id}"
        pre = "prior history\n"
        post = f"prior history\nThe answer is 42.\n{done}\n"

        monkeypatch.setattr(bridge.uuid, "uuid4", lambda: MagicMock(hex=run_id))
        _setup_session(monkeypatch)
        _setup_io(monkeypatch, pre_pane=pre, post_pane=post)
        monkeypatch.setattr(bridge.time, "sleep", lambda x: None)

        result = bridge.submit_prompt("claude-momentum", "What is 6×7?", timeout=30.0)
        assert result.status == BridgeStatus.SUCCESS
        assert "The answer is 42." in result.response

    def test_excludes_prior_pane_history(self, monkeypatch):
        run_id = FAKE_RUN_ID
        done = f"HERMES_BRIDGE_DONE_{run_id}"
        pre = "old output line 1\nold output line 2\n"
        post = f"old output line 1\nold output line 2\nnew response\n{done}\n"

        monkeypatch.setattr(bridge.uuid, "uuid4", lambda: MagicMock(hex=run_id))
        _setup_session(monkeypatch)
        _setup_io(monkeypatch, pre_pane=pre, post_pane=post)
        monkeypatch.setattr(bridge.time, "sleep", lambda x: None)

        result = bridge.submit_prompt("claude-momentum", "hello", timeout=30.0)
        assert result.status == BridgeStatus.SUCCESS
        assert "old output" not in result.response
        assert "new response" in result.response

    def test_ansi_codes_stripped_from_response(self, monkeypatch):
        run_id = FAKE_RUN_ID
        done = f"HERMES_BRIDGE_DONE_{run_id}"
        post = f"\x1b[32mColoured output\x1b[0m\n{done}\n"

        monkeypatch.setattr(bridge.uuid, "uuid4", lambda: MagicMock(hex=run_id))
        _setup_session(monkeypatch)
        _setup_io(monkeypatch, pre_pane="", post_pane=post)
        monkeypatch.setattr(bridge.time, "sleep", lambda x: None)

        result = bridge.submit_prompt("claude-momentum", "color?", timeout=30.0)
        assert result.status == BridgeStatus.SUCCESS
        assert "\x1b" not in result.response
        assert "Coloured output" in result.response


class TestTimeoutAndAuthFailure:
    def _fast_monotonic(self):
        ticks = [0.0]
        def _advance():
            ticks[0] += 10.0
            return ticks[0]
        return _advance

    def test_returns_timeout_when_marker_never_appears(self, monkeypatch):
        _setup_session(monkeypatch)
        monkeypatch.setattr(bridge, "_capture_pane", lambda s, **kw: "no marker here")
        monkeypatch.setattr(bridge, "_send_keys", lambda s, t: True)
        monkeypatch.setattr(bridge.time, "monotonic", self._fast_monotonic())

        result = bridge.submit_prompt("claude-momentum", "hello", timeout=5.0)
        assert result.status == BridgeStatus.TIMEOUT
        assert "5s" in result.error

    def test_returns_auth_failure_when_detected_at_timeout(self, monkeypatch):
        _setup_session(monkeypatch)
        monkeypatch.setattr(bridge, "_send_keys", lambda s, t: True)
        monkeypatch.setattr(bridge.time, "monotonic", self._fast_monotonic())

        call_count = [0]
        def _mock_capture(s, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return ""  # pre-send snapshot
            return "Claude: Authentication required. Please sign in."

        monkeypatch.setattr(bridge, "_capture_pane", _mock_capture)

        result = bridge.submit_prompt("claude-momentum", "hello", timeout=5.0)
        assert result.status == BridgeStatus.AUTH_FAILURE

    def test_no_false_auth_failure_on_success(self, monkeypatch):
        """Old pane history with auth text must not prevent SUCCESS."""
        run_id = FAKE_RUN_ID
        done = f"HERMES_BRIDGE_DONE_{run_id}"
        # Pre-existing history contains an old auth message
        pre = "Authentication required (old history)\n"
        post = f"Authentication required (old history)\nNew response\n{done}\n"

        monkeypatch.setattr(bridge.uuid, "uuid4", lambda: MagicMock(hex=run_id))
        _setup_session(monkeypatch)
        _setup_io(monkeypatch, pre_pane=pre, post_pane=post)
        monkeypatch.setattr(bridge.time, "sleep", lambda x: None)

        result = bridge.submit_prompt("claude-momentum", "hi", timeout=30.0)
        assert result.status == BridgeStatus.SUCCESS


class TestApprovalRequiredIntegration:
    """End-to-end submit_prompt() coverage for the approval-required path."""

    def test_returns_approval_required_status_with_preview(self, monkeypatch):
        run_id = FAKE_RUN_ID
        pre = "prior history\n"
        post = (
            "prior history\n"
            "╭──────────────────────────────╮\n"
            "│ Bash command                  │\n"
            "│   rm -rf /tmp/foo              │\n"
            "│ Do you want to proceed?       │\n"
            "│ ❯ 1. Yes                      │\n"
            "│   2. No                       │\n"
            "╰──────────────────────────────╯\n"
        )
        monkeypatch.setattr(bridge.uuid, "uuid4", lambda: MagicMock(hex=run_id))
        _setup_session(monkeypatch)
        _setup_io(monkeypatch, pre_pane=pre, post_pane=post)
        monkeypatch.setattr(bridge.time, "sleep", lambda x: None)

        result = bridge.submit_prompt("claude-momentum", "delete the tmp dir", timeout=30.0)
        assert result.status == BridgeStatus.APPROVAL_REQUIRED
        assert "claude-momentum" in result.error
        assert "rm -rf /tmp/foo" in result.error
        assert result.response == ""

    def test_no_empty_success_when_approval_prompt_present(self, monkeypatch):
        """Regression for the reported bug: an approval prompt sitting right
        after the echoed prompt must never be reported as an empty SUCCESS."""
        run_id = FAKE_RUN_ID
        done = f"HERMES_BRIDGE_DONE_{run_id}"
        pre = ""
        post = (
            f"> delete the tmp dir (When your response is complete, output exactly on its own line: {done})\n"
            "Do you want to proceed?\n"
            "1. Yes\n"
            "2. No\n"
        )
        monkeypatch.setattr(bridge.uuid, "uuid4", lambda: MagicMock(hex=run_id))
        _setup_session(monkeypatch)
        _setup_io(monkeypatch, pre_pane=pre, post_pane=post)
        monkeypatch.setattr(bridge.time, "sleep", lambda x: None)

        result = bridge.submit_prompt("claude-momentum", "delete the tmp dir", timeout=30.0)
        assert result.status != BridgeStatus.SUCCESS
        assert result.status == BridgeStatus.APPROVAL_REQUIRED
        assert result.response == ""

    def test_completion_marker_still_wins_when_actually_present(self, monkeypatch):
        """Sanity check: a real completed turn is unaffected by the fix."""
        run_id = FAKE_RUN_ID
        done = f"HERMES_BRIDGE_DONE_{run_id}"
        pre = ""
        post = f"> refactor utils.py (When your response is complete, output exactly on its own line: {done})\nDone.\n{done}\n"

        monkeypatch.setattr(bridge.uuid, "uuid4", lambda: MagicMock(hex=run_id))
        _setup_session(monkeypatch)
        _setup_io(monkeypatch, pre_pane=pre, post_pane=post)
        monkeypatch.setattr(bridge.time, "sleep", lambda x: None)

        result = bridge.submit_prompt("claude-momentum", "refactor utils.py", timeout=30.0)
        assert result.status == BridgeStatus.SUCCESS
        assert "Done." in result.response

    def test_lock_released_promptly_after_approval_required(self, monkeypatch):
        """A stuck approval prompt must not hold the per-session lock for the
        full timeout -- it's detected (and the lock released) on the first
        poll, not only at timeout."""
        run_id = FAKE_RUN_ID
        pre = ""
        post = "Do you want to proceed?\n1. Yes\n2. No\n"

        monkeypatch.setattr(bridge.uuid, "uuid4", lambda: MagicMock(hex=run_id))
        _setup_session(monkeypatch)
        _setup_io(monkeypatch, pre_pane=pre, post_pane=post)
        monkeypatch.setattr(bridge.time, "sleep", lambda x: None)

        result = bridge.submit_prompt("claude-momentum", "hello", timeout=300.0)
        assert result.status == BridgeStatus.APPROVAL_REQUIRED

        lock = bridge._get_session_lock("claude-momentum")
        assert lock.acquire(blocking=False), "lock must be free immediately, not held for the full timeout"
        lock.release()

    def test_busy_returned_while_worker_stuck_at_approval(self, monkeypatch):
        """A second delegation to the same worker while the first call is
        still active gets BUSY, not a silent hang or a second concurrent
        prompt submitted into the same pane."""
        _setup_session(monkeypatch)
        lock = bridge._get_session_lock("claude-momentum")
        lock.acquire()
        try:
            result = bridge.submit_prompt("claude-momentum", "another task")
            assert result.status == BridgeStatus.BUSY
        finally:
            lock.release()


class TestSendFailure:
    def test_failure_when_send_keys_returns_false(self, monkeypatch):
        _setup_session(monkeypatch)
        monkeypatch.setattr(bridge, "_capture_pane", lambda s, **kw: "")
        monkeypatch.setattr(bridge, "_send_keys", lambda s, t: False)

        result = bridge.submit_prompt("claude-momentum", "hello", timeout=5.0)
        assert result.status == BridgeStatus.FAILURE
        assert "send-keys" in result.error


class TestWorkerRegistry:
    def test_default_worker_registered(self):
        w = bridge.get_worker("claude-momentum")
        assert w is not None
        assert w.workspace == "/home/michael/code"

    def test_register_worker_roundtrip(self):
        cfg = WorkerConfig(session="test-session", workspace="/tmp/test", timeout=60.0)
        bridge.register_worker(cfg)
        assert bridge.get_worker("test-session") is cfg

    def test_get_worker_returns_none_for_unknown(self):
        assert bridge.get_worker("nonexistent-xyz") is None
