"""Tests for /cc-delegate gateway slash command.

Covers GatewayRunner._handle_cc_delegate_command (parsing/validation +
immediate ack) and GatewayRunner._run_cc_delegate_task (the deferred bridge
call and result delivery), matching the pattern used by
tests/gateway/test_background_command.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agent.claude_code_tmux_bridge import BridgeResult, BridgeStatus, _WORKERS
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text, platform=Platform.TELEGRAM, user_id="12345", chat_id="67890"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    """Create a bare GatewayRunner with minimal mocks (mirrors _make_runner
    in tests/gateway/test_background_command.py)."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._background_tasks = set()
    return runner


@pytest.fixture(autouse=True)
def _restore_worker_registry():
    snapshot = dict(_WORKERS)
    yield
    _WORKERS.clear()
    _WORKERS.update(snapshot)


# ---------------------------------------------------------------------------
# _handle_cc_delegate_command — parsing/validation + immediate ack
# ---------------------------------------------------------------------------


class TestHandleCcDelegateCommand:
    @pytest.mark.asyncio
    async def test_no_args_shows_usage(self):
        runner = _make_runner()
        event = _make_event("/cc-delegate")
        result = await runner._handle_cc_delegate_command(event)
        assert "Usage:" in result
        assert not runner._background_tasks

    @pytest.mark.asyncio
    async def test_unknown_worker_rejected(self):
        runner = _make_runner()
        event = _make_event("/cc-delegate not-a-real-worker do something")
        result = await runner._handle_cc_delegate_command(event)
        assert "not allowlisted" in result
        assert not runner._background_tasks

    @pytest.mark.asyncio
    async def test_empty_prompt_rejected(self):
        runner = _make_runner()
        event = _make_event("/cc-delegate claude-momentum")
        result = await runner._handle_cc_delegate_command(event)
        assert "Prompt cannot be empty" in result
        assert not runner._background_tasks

    @pytest.mark.asyncio
    async def test_path_outside_workspace_rejected(self):
        runner = _make_runner()
        event = _make_event("/cc-delegate claude-momentum --path /etc do something")
        result = await runner._handle_cc_delegate_command(event)
        assert "outside the allowed workspace" in result
        assert not runner._background_tasks

    @pytest.mark.asyncio
    async def test_valid_request_returns_ack_and_schedules_task(self):
        runner = _make_runner()
        event = _make_event("/cc-delegate claude-momentum fix the flaky test")
        with patch("gateway.slash_commands.GatewaySlashCommandsMixin._run_cc_delegate_task",
                   new_callable=AsyncMock) as mock_run:
            result = await runner._handle_cc_delegate_command(event)
        assert "Delegating to" in result
        assert "claude-momentum" in result
        assert "fix the flaky test" in result
        assert "/home/michael/code" in result
        # A background task was scheduled and tracked for cleanup.
        assert len(runner._background_tasks) == 1
        for task in list(runner._background_tasks):
            await task
        mock_run.assert_awaited_once()


# ---------------------------------------------------------------------------
# _run_cc_delegate_task — bridge call + result delivery
# ---------------------------------------------------------------------------


class TestRunCcDelegateTask:
    def _source(self):
        return SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            user_name="testuser",
        )

    @pytest.mark.asyncio
    async def test_success_sends_scoped_response(self):
        from agent.claude_code_delegate import build_request

        runner = _make_runner()
        mock_adapter = AsyncMock()
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        request = build_request("claude-momentum", None, "say hi")
        bridge_result = BridgeResult(status=BridgeStatus.SUCCESS, response="hello there")

        with patch("agent.claude_code_tmux_bridge.submit_prompt", return_value=bridge_result) as mock_submit:
            await runner._run_cc_delegate_task(request, self._source())

        mock_submit.assert_called_once_with("claude-momentum", "say hi")
        mock_adapter.send.assert_called_once()
        content = mock_adapter.send.call_args.kwargs["content"]
        assert "hello there" in content
        assert "claude-momentum" in content

    @pytest.mark.asyncio
    async def test_subpath_scopes_bridge_prompt(self):
        from agent.claude_code_delegate import build_request

        runner = _make_runner()
        mock_adapter = AsyncMock()
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        request = build_request("claude-momentum", "hermes-agent", "run the tests")
        bridge_result = BridgeResult(status=BridgeStatus.SUCCESS, response="tests pass")

        with patch("agent.claude_code_tmux_bridge.submit_prompt", return_value=bridge_result) as mock_submit:
            await runner._run_cc_delegate_task(request, self._source())

        sent_prompt = mock_submit.call_args.args[1]
        assert "/home/michael/code/hermes-agent" in sent_prompt
        assert "run the tests" in sent_prompt

    @pytest.mark.parametrize("status,expect_snippet", [
        (BridgeStatus.TIMEOUT, "timed out"),
        (BridgeStatus.BUSY, "already handling"),
        (BridgeStatus.APPROVAL_REQUIRED, "boom"),
        (BridgeStatus.EXTRACTION_FAILURE, "boom"),
        (BridgeStatus.AUTH_FAILURE, "re-authenticate"),
        (BridgeStatus.SESSION_MISSING, "no running tmux session"),
        (BridgeStatus.SESSION_DEAD, "no live pane"),
        (BridgeStatus.FAILURE, "failed"),
    ])
    @pytest.mark.asyncio
    async def test_error_statuses_delivered_to_operator(self, status, expect_snippet):
        from agent.claude_code_delegate import build_request

        runner = _make_runner()
        mock_adapter = AsyncMock()
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        request = build_request("claude-momentum", None, "do something")
        bridge_result = BridgeResult(status=status, error="boom")

        with patch("agent.claude_code_tmux_bridge.submit_prompt", return_value=bridge_result):
            await runner._run_cc_delegate_task(request, self._source())

        content = mock_adapter.send.call_args.kwargs["content"]
        assert expect_snippet in content

    @pytest.mark.asyncio
    async def test_unexpected_exception_reported_as_failure(self):
        from agent.claude_code_delegate import build_request

        runner = _make_runner()
        mock_adapter = AsyncMock()
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        request = build_request("claude-momentum", None, "do something")

        with patch("agent.claude_code_tmux_bridge.submit_prompt", side_effect=RuntimeError("tmux exploded")):
            await runner._run_cc_delegate_task(request, self._source())

        content = mock_adapter.send.call_args.kwargs["content"]
        assert "failed" in content.lower()
        assert "tmux exploded" in content

    @pytest.mark.asyncio
    async def test_missing_adapter_does_not_raise(self):
        from agent.claude_code_delegate import build_request

        runner = _make_runner()  # no adapters registered
        request = build_request("claude-momentum", None, "do something")

        with patch("agent.claude_code_tmux_bridge.submit_prompt") as mock_submit:
            await runner._run_cc_delegate_task(request, self._source())

        mock_submit.assert_not_called()
