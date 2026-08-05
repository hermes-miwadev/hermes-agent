"""Tests for agent.claude_code_delegate.

Pure unit tests for the /cc-delegate command's parsing, worker
allowlisting, workspace path containment, and result formatting. No
tmux, gateway, or event loop involved.
"""

from __future__ import annotations

import pytest

import agent.claude_code_delegate as delegate
from agent.claude_code_delegate import (
    DelegateValidationError,
    build_request,
    format_ack,
    format_result,
    parse_delegate_args,
    resolve_worker,
    resolve_workspace,
)
from agent.claude_code_tmux_bridge import (
    BridgeResult,
    BridgeStatus,
    WorkerConfig,
    _WORKERS,
)


@pytest.fixture(autouse=True)
def _restore_worker_registry():
    """Snapshot/restore the module-level worker registry around each test.

    Tests register throwaway workers (or rely on the pre-configured
    ``claude-momentum`` one) without leaking state between tests.
    """
    snapshot = dict(_WORKERS)
    yield
    _WORKERS.clear()
    _WORKERS.update(snapshot)


# ── parse_delegate_args ───────────────────────────────────────────────────────

class TestParseDelegateArgs:
    def test_worker_and_prompt(self):
        worker, path, prompt = parse_delegate_args("claude-momentum fix the bug")
        assert worker == "claude-momentum"
        assert path is None
        assert prompt == "fix the bug"

    def test_worker_with_path_flag(self):
        worker, path, prompt = parse_delegate_args(
            "claude-momentum --path subdir/app do the thing"
        )
        assert worker == "claude-momentum"
        assert path == "subdir/app"
        assert prompt == "do the thing"

    def test_repo_alias_flag(self):
        worker, path, prompt = parse_delegate_args(
            "claude-momentum --repo subdir/app do the thing"
        )
        assert path == "subdir/app"
        assert prompt == "do the thing"

    def test_path_flag_can_appear_after_prompt_start(self):
        # Flags are recognized anywhere after the worker name.
        worker, path, prompt = parse_delegate_args(
            "claude-momentum do the thing --path subdir"
        )
        assert path == "subdir"
        assert prompt == "do the thing"

    def test_quoted_prompt_preserves_spacing_semantics(self):
        worker, path, prompt = parse_delegate_args(
            'claude-momentum "refactor the auth module please"'
        )
        assert prompt == "refactor the auth module please"

    def test_quoted_path_with_spaces(self):
        worker, path, prompt = parse_delegate_args(
            'claude-momentum --path "my dir" go'
        )
        assert path == "my dir"
        assert prompt == "go"

    def test_empty_args_raises(self):
        with pytest.raises(DelegateValidationError, match="Usage:"):
            parse_delegate_args("")

    def test_whitespace_only_args_raises(self):
        with pytest.raises(DelegateValidationError, match="Usage:"):
            parse_delegate_args("   ")

    def test_worker_with_no_prompt_raises(self):
        with pytest.raises(DelegateValidationError, match="Prompt cannot be empty"):
            parse_delegate_args("claude-momentum")

    def test_worker_with_only_path_flag_raises_empty_prompt(self):
        with pytest.raises(DelegateValidationError, match="Prompt cannot be empty"):
            parse_delegate_args("claude-momentum --path subdir")

    def test_path_flag_missing_value_raises(self):
        with pytest.raises(DelegateValidationError, match="requires a directory"):
            parse_delegate_args("claude-momentum --path")

    def test_unbalanced_quotes_raise(self):
        with pytest.raises(DelegateValidationError, match="Could not parse"):
            parse_delegate_args('claude-momentum "unterminated')


# ── resolve_worker (allowlisting) ─────────────────────────────────────────────

class TestResolveWorker:
    def test_preconfigured_worker_resolves(self):
        worker = resolve_worker("claude-momentum")
        assert worker.session == "claude-momentum"
        assert worker.workspace == "/home/michael/code"

    def test_unknown_worker_rejected(self):
        with pytest.raises(DelegateValidationError, match="not allowlisted"):
            resolve_worker("some-random-session")

    def test_case_sensitive_worker_name(self):
        # tmux session names are case-sensitive; an unregistered variant
        # must not silently match the registered one.
        with pytest.raises(DelegateValidationError, match="not allowlisted"):
            resolve_worker("Claude-Momentum")


# ── resolve_workspace (path containment) ──────────────────────────────────────

class TestResolveWorkspace:
    def setup_method(self):
        self.worker = WorkerConfig(session="w", workspace="/home/michael/code")

    def test_no_path_returns_root(self):
        assert resolve_workspace(self.worker, None) == "/home/michael/code"

    def test_relative_subpath_contained(self):
        assert resolve_workspace(self.worker, "hermes-agent") == "/home/michael/code/hermes-agent"

    def test_nested_relative_subpath_contained(self):
        assert (
            resolve_workspace(self.worker, "hermes-agent/gateway")
            == "/home/michael/code/hermes-agent/gateway"
        )

    def test_absolute_path_inside_root_allowed(self):
        assert (
            resolve_workspace(self.worker, "/home/michael/code/hermes-agent")
            == "/home/michael/code/hermes-agent"
        )

    @pytest.mark.parametrize("bad_path", [
        "../etc",
        "../../etc/passwd",
        "/etc/passwd",
        "/home/michael",
        "/home/michael/codebase",  # sibling with root as a string-prefix, not a path-prefix
        "subdir/../../etc",
    ])
    def test_path_outside_workspace_rejected(self, bad_path):
        with pytest.raises(DelegateValidationError, match="outside the allowed workspace"):
            resolve_workspace(self.worker, bad_path)

    def test_dot_dot_that_stays_inside_is_allowed(self):
        # "a/../b" normalizes to "b", still under the root.
        assert (
            resolve_workspace(self.worker, "hermes-agent/../hermes-agent")
            == "/home/michael/code/hermes-agent"
        )


# ── build_request (combined resolve) ──────────────────────────────────────────

class TestBuildRequest:
    def test_success_produces_request(self):
        req = build_request("claude-momentum", None, "do the thing")
        assert req.worker_name == "claude-momentum"
        assert req.workspace == "/home/michael/code"
        assert req.prompt == "do the thing"
        assert req.bridge_prompt == "do the thing"

    def test_subpath_prefixes_bridge_prompt_with_scope(self):
        req = build_request("claude-momentum", "hermes-agent", "run tests")
        assert req.workspace == "/home/michael/code/hermes-agent"
        assert req.bridge_prompt == "(Work only within /home/michael/code/hermes-agent.) run tests"
        # The unmodified user prompt is preserved separately.
        assert req.prompt == "run tests"

    def test_unknown_worker_propagates_error(self):
        with pytest.raises(DelegateValidationError, match="not allowlisted"):
            build_request("nope", None, "do the thing")

    def test_escaping_path_propagates_error(self):
        with pytest.raises(DelegateValidationError, match="outside the allowed workspace"):
            build_request("claude-momentum", "/etc", "do the thing")


# ── format_ack / format_result ─────────────────────────────────────────────────

class TestFormatAck:
    def test_includes_worker_workspace_and_prompt(self):
        req = build_request("claude-momentum", None, "do the thing")
        ack = format_ack(req)
        assert "claude-momentum" in ack
        assert "/home/michael/code" in ack
        assert "do the thing" in ack

    def test_truncates_long_prompt_preview(self):
        long_prompt = "x" * 200
        req = build_request("claude-momentum", None, long_prompt)
        ack = format_ack(req)
        assert "..." in ack
        assert len(ack.splitlines()[-1]) < len(long_prompt)


class TestFormatResult:
    def test_success_returns_only_response(self):
        result = BridgeResult(status=BridgeStatus.SUCCESS, response="Here is the diff.")
        out = format_result("claude-momentum", result)
        assert "Here is the diff." in out
        assert "claude-momentum" in out

    def test_success_does_not_leak_error_field(self):
        result = BridgeResult(status=BridgeStatus.SUCCESS, response="ok", error="should not show")
        out = format_result("claude-momentum", result)
        assert "should not show" not in out

    @pytest.mark.parametrize("status,expect_snippets", [
        (BridgeStatus.TIMEOUT, ["timed out"]),
        (BridgeStatus.BUSY, ["already handling"]),
        (BridgeStatus.AUTH_FAILURE, ["re-authenticate"]),
        (BridgeStatus.SESSION_MISSING, ["no running tmux session"]),
        (BridgeStatus.SESSION_DEAD, ["no live pane"]),
        (BridgeStatus.FAILURE, ["failed"]),
    ])
    def test_error_statuses_map_to_operator_text(self, status, expect_snippets):
        result = BridgeResult(status=status, error="boom")
        out = format_result("claude-momentum", result)
        assert "claude-momentum" in out
        for snippet in expect_snippets:
            assert snippet in out

    def test_approval_required_shows_bridge_error_verbatim(self):
        # The bridge's own error text already names the session (and, when
        # extractable, a sanitised preview) -- format_result must not
        # editorialise or truncate it, just prefix it for operators.
        result = BridgeResult(
            status=BridgeStatus.APPROVAL_REQUIRED,
            error="Claude Code requires approval in tmux session 'claude-momentum'. "
                  "Pending: Bash command rm -rf /tmp/foo",
        )
        out = format_result("claude-momentum", result)
        assert "claude-momentum" in out
        assert "requires approval" in out
        assert "Pending: Bash command rm -rf /tmp/foo" in out

    def test_approval_required_never_reads_as_success(self):
        result = BridgeResult(status=BridgeStatus.APPROVAL_REQUIRED, error="needs approval")
        out = format_result("claude-momentum", result)
        assert not out.startswith("✅")

    def test_extraction_failure_shows_bridge_error_verbatim(self):
        result = BridgeResult(
            status=BridgeStatus.EXTRACTION_FAILURE,
            error="Claude Code finished in tmux session 'claude-momentum', but no "
                  "response text could be extracted from the pane.",
        )
        out = format_result("claude-momentum", result)
        assert "claude-momentum" in out
        assert "no response text could be extracted" in out

    def test_extraction_failure_never_reads_as_success(self):
        result = BridgeResult(status=BridgeStatus.EXTRACTION_FAILURE, error="nothing extracted")
        out = format_result("claude-momentum", result)
        assert not out.startswith("✅")

    def test_failure_includes_error_detail(self):
        result = BridgeResult(status=BridgeStatus.FAILURE, error="tmux not found on PATH")
        out = format_result("claude-momentum", result)
        assert "tmux not found on PATH" in out
