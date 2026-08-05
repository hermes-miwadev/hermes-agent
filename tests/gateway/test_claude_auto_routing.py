"""Tests for GatewayClaudeAutoRoutingMixin (configurable Discord auto-routing).

Covers GatewayRunner._maybe_auto_route_to_claude (the routing decision) and
_dispatch_auto_routed_delegation (building + firing the delegation), reusing
the same _make_runner/_make_event conventions as
tests/gateway/test_cc_delegate_command.py and
tests/gateway/test_discord_thread_reliability.py.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.claude_code_tmux_bridge import BridgeResult, BridgeStatus, _WORKERS
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource

CHANNEL_ID = "111222333444555666"
THREAD_ID = "777888999000111222"


def _routing_config(**overrides):
    entry = {
        "platform": "discord",
        "channel_id": CHANNEL_ID,
        "enabled": True,
        "worker": "claude-momentum",
        "workspace": "/home/michael/code/momentum-studio",
        "mode": "claude_default",
    }
    entry.update(overrides)
    return {"claude_routing": [entry]}


def _make_event(
    text,
    *,
    platform=Platform.DISCORD,
    chat_id=CHANNEL_ID,
    parent_chat_id=None,
    is_bot=False,
    media_urls=None,
    message_id=None,
):
    source = SessionSource(
        platform=platform,
        user_id="42",
        chat_id=chat_id,
        parent_chat_id=parent_chat_id,
        is_bot=is_bot,
        user_name="Michael",
    )
    return MessageEvent(
        text=text, source=source, media_urls=list(media_urls or []), message_id=message_id
    )


def _make_runner(config=None):
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._background_tasks = set()
    runner.config = config if config is not None else {}
    return runner


@pytest.fixture(autouse=True)
def _restore_worker_registry():
    snapshot = dict(_WORKERS)
    yield
    _WORKERS.clear()
    _WORKERS.update(snapshot)


@pytest.fixture(autouse=True)
def _isolated_state_db(tmp_path, monkeypatch):
    """Isolated state.db per test for the Phase 4 task registry + delivery
    ledger, both exercised by _dispatch_auto_routed_delegation /
    _run_cc_delegate_task. Mirrors tests/gateway/test_delivery_ledger.py's
    explicit _db_path patch -- without this, duplicate-suppression tests
    could see stale rows a previous test in the same session left behind."""
    import gateway.claude_task_registry as task_registry
    import gateway.delivery_ledger as delivery_ledger

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(task_registry, "_db_path", lambda: home / "state.db")
    monkeypatch.setattr(delivery_ledger, "_db_path", lambda: home / "state.db")


# ---------------------------------------------------------------------------
# _maybe_auto_route_to_claude -- the routing decision
# ---------------------------------------------------------------------------


class TestMaybeAutoRouteToClaudeGating:
    @pytest.mark.asyncio
    async def test_unconfigured_channel_returns_none(self):
        runner = _make_runner(config={})
        event = _make_event("Ship the pricing page.")
        result = await runner._maybe_auto_route_to_claude(event, event.source)
        assert result is None
        assert not runner._background_tasks

    @pytest.mark.asyncio
    async def test_channel_configured_for_different_id_returns_none(self):
        runner = _make_runner(config=_routing_config())
        event = _make_event("Ship the pricing page.", chat_id="999999999999999999")
        result = await runner._maybe_auto_route_to_claude(event, event.source)
        assert result is None
        assert not runner._background_tasks

    @pytest.mark.asyncio
    async def test_disabled_mapping_returns_none(self):
        runner = _make_runner(config=_routing_config(enabled=False))
        event = _make_event("Ship the pricing page.")
        result = await runner._maybe_auto_route_to_claude(event, event.source)
        assert result is None
        assert not runner._background_tasks

    @pytest.mark.asyncio
    async def test_non_discord_platform_returns_none(self):
        runner = _make_runner(config=_routing_config())
        event = _make_event("Ship the pricing page.", platform=Platform.TELEGRAM)
        result = await runner._maybe_auto_route_to_claude(event, event.source)
        assert result is None
        assert not runner._background_tasks

    @pytest.mark.asyncio
    async def test_bot_message_not_recursively_delegated(self):
        runner = _make_runner(config=_routing_config())
        event = _make_event("Ship the pricing page.", is_bot=True)
        result = await runner._maybe_auto_route_to_claude(event, event.source)
        assert result is None
        assert not runner._background_tasks

    @pytest.mark.asyncio
    async def test_slash_command_not_intercepted(self):
        runner = _make_runner(config=_routing_config())
        event = _make_event("/status", chat_id=CHANNEL_ID)
        result = await runner._maybe_auto_route_to_claude(event, event.source)
        assert result is None
        assert not runner._background_tasks

    @pytest.mark.asyncio
    async def test_cc_delegate_command_not_intercepted(self):
        runner = _make_runner(config=_routing_config())
        event = _make_event("/cc-delegate claude-momentum do something", chat_id=CHANNEL_ID)
        result = await runner._maybe_auto_route_to_claude(event, event.source)
        assert result is None
        assert not runner._background_tasks

    @pytest.mark.asyncio
    async def test_empty_text_returns_none(self):
        runner = _make_runner(config=_routing_config())
        event = _make_event("   ")
        result = await runner._maybe_auto_route_to_claude(event, event.source)
        assert result is None

    @pytest.mark.asyncio
    async def test_media_attachment_not_auto_routed(self):
        runner = _make_runner(config=_routing_config())
        event = _make_event("check this out", media_urls=["/tmp/image.png"])
        result = await runner._maybe_auto_route_to_claude(event, event.source)
        assert result is None
        assert not runner._background_tasks

    @pytest.mark.asyncio
    async def test_thread_under_configured_channel_matches_via_parent(self):
        runner = _make_runner(config=_routing_config())
        event = _make_event(
            "Ship the pricing page.", chat_id=THREAD_ID, parent_chat_id=CHANNEL_ID
        )
        with patch("gateway.run.GatewayRunner._run_cc_delegate_task", new_callable=AsyncMock):
            result = await runner._maybe_auto_route_to_claude(event, event.source)
        assert result is not None
        assert "Automatic routing activated" in result


class TestMaybeAutoRouteToClaudeBypass:
    @pytest.mark.asyncio
    async def test_chat_bypass_returns_none_and_rewrites_text(self):
        runner = _make_runner(config=_routing_config())
        event = _make_event("chat: help me think through the strategy first")
        result = await runner._maybe_auto_route_to_claude(event, event.source)
        assert result is None
        assert event.text == "help me think through the strategy first"
        assert not runner._background_tasks

    @pytest.mark.asyncio
    async def test_no_bypass_does_not_alter_text(self):
        runner = _make_runner(config=_routing_config())
        event = _make_event("Ship the pricing page.")
        with patch("gateway.run.GatewayRunner._run_cc_delegate_task", new_callable=AsyncMock):
            await runner._maybe_auto_route_to_claude(event, event.source)
        assert event.text == "Ship the pricing page."


# ---------------------------------------------------------------------------
# Successful dispatch: reuses build_request + _run_cc_delegate_task verbatim
# ---------------------------------------------------------------------------


class TestSuccessfulAutoRoutedDispatch:
    @pytest.mark.asyncio
    async def test_returns_ack_and_schedules_delegation_task(self):
        runner = _make_runner(config=_routing_config())
        event = _make_event(
            "Continue getting the Momentum Studio website launch-ready. "
            "Choose the highest-priority remaining improvement, implement "
            "it, test it, push it, and open a PR."
        )
        with patch("gateway.run.GatewayRunner._run_cc_delegate_task", new_callable=AsyncMock) as mock_run:
            result = await runner._maybe_auto_route_to_claude(event, event.source)

        assert result is not None
        assert "Automatic routing activated" in result
        assert "claude_default" in result
        assert "claude-momentum" in result
        assert "/home/michael/code/momentum-studio" in result
        assert len(runner._background_tasks) == 1
        for task in list(runner._background_tasks):
            await task
        mock_run.assert_awaited_once()
        request = mock_run.await_args.args[0]
        assert request.worker_name == "claude-momentum"
        assert request.workspace == "/home/michael/code/momentum-studio"
        assert "Continue getting the Momentum Studio website launch-ready" in request.prompt
        # Standard implementation requirements are included by default.
        assert "pull request" in request.prompt

    @pytest.mark.asyncio
    async def test_standard_instructions_disabled_are_omitted(self):
        runner = _make_runner(config=_routing_config(standard_instructions=False))
        event = _make_event("Ship the pricing page.")
        with patch("gateway.run.GatewayRunner._run_cc_delegate_task", new_callable=AsyncMock) as mock_run:
            await runner._maybe_auto_route_to_claude(event, event.source)
        for task in list(runner._background_tasks):
            await task
        request = mock_run.await_args.args[0]
        assert request.prompt == "Ship the pricing page."

    @pytest.mark.asyncio
    async def test_result_delivery_reuses_existing_bridge_status_mapping(self):
        """End-to-end through the real _run_cc_delegate_task (not mocked) --
        proves auto-routing reuses the exact same bridge-status formatting
        /cc-delegate uses, rather than duplicating it."""
        runner = _make_runner(config=_routing_config())
        mock_adapter = AsyncMock()
        runner.adapters[Platform.DISCORD] = mock_adapter
        event = _make_event("Ship the pricing page.")

        bridge_result = BridgeResult(status=BridgeStatus.SUCCESS, response="Shipped it.")
        with patch("agent.claude_code_tmux_bridge.submit_prompt", return_value=bridge_result):
            await runner._maybe_auto_route_to_claude(event, event.source)
            for task in list(runner._background_tasks):
                await task

        mock_adapter.send.assert_called_once()
        content = mock_adapter.send.call_args.kwargs["content"]
        assert "Shipped it." in content
        assert "claude-momentum" in content

    @pytest.mark.parametrize("status,expect_snippet", [
        (BridgeStatus.TIMEOUT, "timed out"),
        (BridgeStatus.BUSY, "already handling"),
        (BridgeStatus.APPROVAL_REQUIRED, "boom"),
        (BridgeStatus.AUTH_FAILURE, "re-authenticate"),
        (BridgeStatus.SESSION_MISSING, "no running tmux session"),
        (BridgeStatus.SESSION_DEAD, "no live pane"),
        (BridgeStatus.EXTRACTION_FAILURE, "boom"),
        (BridgeStatus.FAILURE, "failed"),
    ])
    @pytest.mark.asyncio
    async def test_every_bridge_status_is_reported_not_swallowed(self, status, expect_snippet):
        runner = _make_runner(config=_routing_config())
        mock_adapter = AsyncMock()
        runner.adapters[Platform.DISCORD] = mock_adapter
        event = _make_event("Ship the pricing page.")

        bridge_result = BridgeResult(status=status, error="boom")
        with patch("agent.claude_code_tmux_bridge.submit_prompt", return_value=bridge_result):
            await runner._maybe_auto_route_to_claude(event, event.source)
            for task in list(runner._background_tasks):
                await task

        content = mock_adapter.send.call_args.kwargs["content"]
        assert expect_snippet in content
        # Never silently reads as a normal Hermes/OpenAI response -- it is
        # always attributable to the Claude Code worker's own outcome.
        assert content != ""


# ---------------------------------------------------------------------------
# Misconfiguration: never silently falls back to Hermes/OpenAI
# ---------------------------------------------------------------------------


class TestMisconfiguredMapping:
    @pytest.mark.asyncio
    async def test_unknown_worker_reports_clear_error_not_none(self):
        runner = _make_runner(config=_routing_config(worker="not-a-real-worker"))
        event = _make_event("Ship the pricing page.")
        result = await runner._maybe_auto_route_to_claude(event, event.source)
        assert result is not None
        assert "misconfigured" in result.lower()
        assert not runner._background_tasks

    @pytest.mark.asyncio
    async def test_workspace_outside_worker_root_reports_clear_error(self):
        """Workspace containment: a configured workspace outside the
        worker's allowed root is rejected, not silently widened."""
        runner = _make_runner(config=_routing_config(workspace="/etc"))
        event = _make_event("Ship the pricing page.")
        result = await runner._maybe_auto_route_to_claude(event, event.source)
        assert result is not None
        assert "misconfigured" in result.lower()
        assert not runner._background_tasks

    @pytest.mark.asyncio
    async def test_misconfiguration_never_returns_none(self):
        """A None return here means 'fall through to the agent loop' --
        misconfiguration must never produce that, or a routed channel would
        silently degrade into ordinary Hermes/OpenAI handling."""
        runner = _make_runner(config=_routing_config(worker="not-a-real-worker"))
        event = _make_event("Ship the pricing page.")
        result = await runner._maybe_auto_route_to_claude(event, event.source)
        assert result is not None


# ---------------------------------------------------------------------------
# Duplicate/racing delegation: the bridge's per-session lock, reused as-is
# ---------------------------------------------------------------------------


class TestDuplicateDelegationSuppression:
    @pytest.mark.asyncio
    async def test_second_concurrent_delegation_for_same_worker_gets_busy(self):
        """Two auto-routed messages hitting the same worker back-to-back
        (e.g. a duplicate event that slipped past upstream dedup) must not
        create two Claude tasks -- the bridge's existing per-session lock
        (unchanged, reused) rejects the second with BUSY."""
        from agent.claude_code_tmux_bridge import _get_session_lock

        runner = _make_runner(config=_routing_config())
        mock_adapter = AsyncMock()
        runner.adapters[Platform.DISCORD] = mock_adapter

        lock = _get_session_lock("claude-momentum")
        lock.acquire()
        try:
            event = _make_event("Ship the pricing page.")
            await runner._maybe_auto_route_to_claude(event, event.source)
            for task in list(runner._background_tasks):
                await task
        finally:
            lock.release()

        content = mock_adapter.send.call_args.kwargs["content"]
        assert "already handling" in content


# ---------------------------------------------------------------------------
# Real GatewayConfig + RelayRuntime unavailability
#
# Every test above builds `runner.config` from a bare dict. That's exactly
# how the bug shipped and went unnoticed: GatewayRunner.config in production
# is a `gateway.config.GatewayConfig` dataclass instance built by
# `load_gateway_config()`, not a dict, and (before the fix this module
# accompanies) that dataclass had no `claude_routing` field at all --
# `getattr(config, "claude_routing", None)` silently returned None no matter
# what was in config.yaml, so `load_routing_mappings` always saw an empty
# list and auto-routing could never fire for a real gateway. The bare-dict
# mocks above couldn't catch that because a dict always has the key once
# it's put there.
#
# The RelayRuntime/nemo_relay angle looked related only because of log
# proximity: when auto-routing silently no-ops, the message falls through to
# the normal Hermes/OpenAI agent loop, which is what actually triggers
# agent.relay_runtime's lazy, per-profile RelayRuntime init (and its
# graceful ModuleNotFoundError-to-NoopRelayRuntime fallback) on the next
# turn. Nothing in the auto-routing decision/dispatch path
# (gateway/claude_auto_routing.py, agent/claude_code_auto_routing.py,
# agent/claude_code_delegate.py, agent/claude_code_tmux_bridge.py) imports
# agent.relay_runtime or nemo_relay, so the tests below simulate nemo_relay
# being fully unimportable and confirm that has zero effect on auto-routing.
# ---------------------------------------------------------------------------


def _real_config(**overrides) -> GatewayConfig:
    """Build claude_routing through the actual production config type."""
    entry = {
        "platform": "discord",
        "channel_id": CHANNEL_ID,
        "enabled": True,
        "worker": "claude-momentum",
        "workspace": "/home/michael/code/momentum-studio",
        "mode": "claude_default",
    }
    entry.update(overrides)
    return GatewayConfig.from_dict({"claude_routing": [entry]})


class TestAutoRoutingAgainstRealGatewayConfig:
    @pytest.mark.asyncio
    async def test_fires_against_a_real_gatewayconfig_instance(self):
        """Regression for the actual root cause: GatewayRunner.config is a
        GatewayConfig dataclass in production, not a dict. Auto-routing must
        activate when the mapping comes from GatewayConfig.from_dict() (what
        load_gateway_config() produces), not just from a hand-built dict."""
        runner = _make_runner(config=_real_config())
        event = _make_event("Say exactly: automatic routing test successful")

        with patch(
            "gateway.run.GatewayRunner._run_cc_delegate_task", new_callable=AsyncMock
        ) as mock_run:
            result = await runner._maybe_auto_route_to_claude(event, event.source)

        assert result is not None
        assert "Automatic routing activated" in result
        assert len(runner._background_tasks) == 1
        for task in list(runner._background_tasks):
            await task
        mock_run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disabled_mapping_via_real_config_returns_none(self):
        runner = _make_runner(config=_real_config(enabled=False))
        event = _make_event("Ship the pricing page.")
        result = await runner._maybe_auto_route_to_claude(event, event.source)
        assert result is None
        assert not runner._background_tasks


class TestAutoRoutingWithoutNemoRelay:
    """Automatic Claude Code routing must not depend on RelayRuntime/nemo_relay."""

    @pytest.fixture(autouse=True)
    def _nemo_relay_unavailable(self):
        from agent import relay_runtime

        relay_runtime._reset_for_tests()

        def _raise_module_not_found(*_args, **_kwargs):
            raise ModuleNotFoundError("No module named 'nemo_relay'")

        with patch.object(
            relay_runtime, "_load_nemo_relay", side_effect=_raise_module_not_found
        ):
            yield
        relay_runtime._reset_for_tests()

    @pytest.mark.asyncio
    async def test_auto_routing_fires_when_nemo_relay_import_fails(self):
        from agent import relay_runtime

        runner = _make_runner(config=_real_config())
        event = _make_event("Say exactly: automatic routing test successful")

        with patch(
            "gateway.run.GatewayRunner._run_cc_delegate_task", new_callable=AsyncMock
        ) as mock_run:
            result = await runner._maybe_auto_route_to_claude(event, event.source)

        assert result is not None
        assert "Automatic routing activated" in result
        for task in list(runner._background_tasks):
            await task
        mock_run.assert_awaited_once()

        # Confirm the simulated unavailability was real (not silently
        # skipped): RelayRuntime construction actually failed and the
        # registry fell back to NoopRelayRuntime, exactly as production logs
        # show ("Hermes Relay runtime initialization failed").
        assert relay_runtime.get_runtime(create=True) is None

    @pytest.mark.asyncio
    async def test_end_to_end_delegation_result_delivered_without_nemo_relay(self):
        """Not just the routing decision -- the full dispatch through the
        real _run_cc_delegate_task/tmux bridge machinery must also complete
        and deliver a result when nemo_relay cannot be imported."""
        runner = _make_runner(config=_real_config())
        mock_adapter = AsyncMock()
        runner.adapters[Platform.DISCORD] = mock_adapter
        event = _make_event("Ship the pricing page.")

        bridge_result = BridgeResult(status=BridgeStatus.SUCCESS, response="Shipped it.")
        with patch("agent.claude_code_tmux_bridge.submit_prompt", return_value=bridge_result):
            await runner._maybe_auto_route_to_claude(event, event.source)
            for task in list(runner._background_tasks):
                await task

        mock_adapter.send.assert_called_once()
        content = mock_adapter.send.call_args.kwargs["content"]
        assert "Shipped it." in content


# ---------------------------------------------------------------------------
# Phase 4: asynchronous completion delivery
#
# Auto-routing already acknowledged immediately and ran the bridge call in
# a background asyncio task (Phase 3). Phase 4 adds: a stable task ID in
# the ack, a configurable background-worker timeout longer than the
# bridge's 300s default, duplicate-event suppression keyed by the
# triggering message, and delivery to the originating thread/channel with
# durable state surviving a restart -- all layered on top of the same
# unchanged bridge/delegation machinery.
# ---------------------------------------------------------------------------


class TestImmediateAcknowledgementAndTaskId:
    @pytest.mark.asyncio
    async def test_ack_returns_before_bridge_call_completes(self):
        """The ack is returned as soon as the background task is scheduled --
        it does not wait for submit_prompt, however long that takes."""
        runner = _make_runner(config=_routing_config())
        event = _make_event("Ship the pricing page.", message_id="msg-immediate")

        never_returns = asyncio.Event()

        def _blocking_submit(*args, **kwargs):
            # A real bridge call would block synchronously in a thread;
            # simulate "still running" by never returning during the test.
            never_returns.set()
            raise AssertionError("submit_prompt should not have been awaited yet")

        with patch("agent.claude_code_tmux_bridge.submit_prompt", side_effect=_blocking_submit):
            result = await asyncio.wait_for(
                runner._maybe_auto_route_to_claude(event, event.source), timeout=2.0
            )

        assert result is not None
        assert "Automatic routing activated" in result
        # The background task exists but we deliberately never awaited it,
        # proving the ack did not wait on the bridge call.
        assert len(runner._background_tasks) == 1
        for task in list(runner._background_tasks):
            task.cancel()

    @pytest.mark.asyncio
    async def test_ack_contains_worker_workspace_and_stable_task_id(self):
        runner = _make_runner(config=_routing_config())
        event = _make_event("Ship the pricing page.", message_id="msg-fields")

        with patch("gateway.run.GatewayRunner._run_cc_delegate_task", new_callable=AsyncMock):
            result = await runner._maybe_auto_route_to_claude(event, event.source)

        assert "claude-momentum" in result
        assert "/home/michael/code/momentum-studio" in result
        assert "Task ID: `cc-" in result
        assert "posted" in result.lower() or "post" in result.lower()

    @pytest.mark.asyncio
    async def test_task_id_is_stable_for_the_same_message(self):
        from gateway.claude_task_registry import compute_task_id

        runner = _make_runner(config=_routing_config())
        event = _make_event("Ship the pricing page.", message_id="msg-stable")
        session_key = runner._session_key_for_source(event.source)
        expected = compute_task_id(session_key, "msg-stable", "claude-momentum")

        with patch("gateway.run.GatewayRunner._run_cc_delegate_task", new_callable=AsyncMock):
            result = await runner._maybe_auto_route_to_claude(event, event.source)

        assert expected in result


class TestDelayedCompletionDelivery:
    @pytest.mark.asyncio
    async def test_result_delivered_after_ack_once_bridge_finishes(self):
        runner = _make_runner(config=_routing_config())
        mock_adapter = AsyncMock()
        runner.adapters[Platform.DISCORD] = mock_adapter
        event = _make_event("Ship the pricing page.", message_id="msg-delayed")

        bridge_result = BridgeResult(status=BridgeStatus.SUCCESS, response="Shipped it.")
        with patch("agent.claude_code_tmux_bridge.submit_prompt", return_value=bridge_result):
            ack = await runner._maybe_auto_route_to_claude(event, event.source)
            # At this point nothing has been sent to Discord yet.
            mock_adapter.send.assert_not_called()
            for task in list(runner._background_tasks):
                await task

        assert "Task ID:" in ack
        mock_adapter.send.assert_called_once()
        assert "Shipped it." in mock_adapter.send.call_args.kwargs["content"]

    @pytest.mark.asyncio
    async def test_task_longer_than_300_seconds_uses_background_timeout_not_bridge_default(self):
        """The bridge's own per-call default is 300s. Auto-routing must pass
        a configurable, longer background-worker timeout explicitly, so a
        task that legitimately runs past 300s is still waited for instead
        of being abandoned at the bridge's default ceiling."""
        runner = _make_runner(config=_routing_config(background_timeout_seconds=3600))
        mock_adapter = AsyncMock()
        runner.adapters[Platform.DISCORD] = mock_adapter
        event = _make_event("Do a very long task.", message_id="msg-long")

        bridge_result = BridgeResult(status=BridgeStatus.SUCCESS, response="Finally done.")
        with patch(
            "agent.claude_code_tmux_bridge.submit_prompt", return_value=bridge_result
        ) as mock_submit:
            await runner._maybe_auto_route_to_claude(event, event.source)
            for task in list(runner._background_tasks):
                await task

        assert mock_submit.call_args.kwargs["timeout"] == 3600
        assert mock_submit.call_args.kwargs["timeout"] > 300

    @pytest.mark.asyncio
    async def test_default_background_timeout_exceeds_300_seconds(self):
        from agent.claude_code_auto_routing import DEFAULT_BACKGROUND_TIMEOUT_SECONDS

        assert DEFAULT_BACKGROUND_TIMEOUT_SECONDS > 300

        runner = _make_runner(config=_routing_config())  # no override -> default
        mock_adapter = AsyncMock()
        runner.adapters[Platform.DISCORD] = mock_adapter
        event = _make_event("Ship the pricing page.", message_id="msg-default-timeout")

        bridge_result = BridgeResult(status=BridgeStatus.SUCCESS, response="done")
        with patch(
            "agent.claude_code_tmux_bridge.submit_prompt", return_value=bridge_result
        ) as mock_submit:
            await runner._maybe_auto_route_to_claude(event, event.source)
            for task in list(runner._background_tasks):
                await task

        assert mock_submit.call_args.kwargs["timeout"] == DEFAULT_BACKGROUND_TIMEOUT_SECONDS

    @pytest.mark.parametrize("status,expect_snippet", [
        (BridgeStatus.APPROVAL_REQUIRED, "boom"),
        (BridgeStatus.TIMEOUT, "timed out"),
        (BridgeStatus.BUSY, "already handling"),
        (BridgeStatus.AUTH_FAILURE, "re-authenticate"),
        (BridgeStatus.SESSION_MISSING, "no running tmux session"),
        (BridgeStatus.SESSION_DEAD, "no live pane"),
        (BridgeStatus.EXTRACTION_FAILURE, "boom"),
        (BridgeStatus.FAILURE, "failed"),
    ])
    @pytest.mark.asyncio
    async def test_every_bridge_outcome_is_delivered_asynchronously(self, status, expect_snippet):
        runner = _make_runner(config=_routing_config())
        mock_adapter = AsyncMock()
        runner.adapters[Platform.DISCORD] = mock_adapter
        event = _make_event("Ship the pricing page.", message_id=f"msg-{status.value}")

        bridge_result = BridgeResult(status=status, error="boom")
        with patch("agent.claude_code_tmux_bridge.submit_prompt", return_value=bridge_result):
            await runner._maybe_auto_route_to_claude(event, event.source)
            for task in list(runner._background_tasks):
                await task

        content = mock_adapter.send.call_args.kwargs["content"]
        assert expect_snippet in content


class TestDuplicateEventSuppression:
    @pytest.mark.asyncio
    async def test_same_message_id_dispatched_twice_only_runs_bridge_once(self):
        runner = _make_runner(config=_routing_config())
        mock_adapter = AsyncMock()
        runner.adapters[Platform.DISCORD] = mock_adapter

        bridge_result = BridgeResult(status=BridgeStatus.SUCCESS, response="Shipped it.")
        with patch(
            "agent.claude_code_tmux_bridge.submit_prompt", return_value=bridge_result
        ) as mock_submit:
            event1 = _make_event("Ship the pricing page.", message_id="msg-dup")
            first_ack = await runner._maybe_auto_route_to_claude(event1, event1.source)
            for task in list(runner._background_tasks):
                await task

            # A second, independent MessageEvent carrying the SAME Discord
            # message_id -- simulating a gateway RESUME replay, missed-
            # message backfill, or a retried dispatch of the same event.
            event2 = _make_event("Ship the pricing page.", message_id="msg-dup")
            second_ack = await runner._maybe_auto_route_to_claude(event2, event2.source)

        mock_submit.assert_called_once()
        mock_adapter.send.assert_called_once()
        assert "Task ID:" in first_ack
        assert "already dispatched" in second_ack
        assert "🔁" in second_ack

    @pytest.mark.asyncio
    async def test_duplicate_notice_reports_current_state(self):
        runner = _make_runner(config=_routing_config())
        event1 = _make_event("Ship the pricing page.", message_id="msg-dup-state")
        with patch("gateway.run.GatewayRunner._run_cc_delegate_task", new_callable=AsyncMock):
            await runner._maybe_auto_route_to_claude(event1, event1.source)

        event2 = _make_event("Ship the pricing page.", message_id="msg-dup-state")
        second_ack = await runner._maybe_auto_route_to_claude(event2, event2.source)

        assert "Execution: `running`" in second_ack
        assert "delivery: `pending`" in second_ack

    @pytest.mark.asyncio
    async def test_different_message_id_same_text_is_not_suppressed(self):
        """Two genuinely distinct messages with identical text are two
        separate requests -- only a literal replay of the SAME message_id
        is a duplicate."""
        runner = _make_runner(config=_routing_config())
        mock_adapter = AsyncMock()
        runner.adapters[Platform.DISCORD] = mock_adapter

        bridge_result = BridgeResult(status=BridgeStatus.SUCCESS, response="done")
        with patch("agent.claude_code_tmux_bridge.submit_prompt", return_value=bridge_result) as mock_submit:
            event1 = _make_event("Ship the pricing page.", message_id="msg-a")
            await runner._maybe_auto_route_to_claude(event1, event1.source)
            for task in list(runner._background_tasks):
                await task

            event2 = _make_event("Ship the pricing page.", message_id="msg-b")
            ack2 = await runner._maybe_auto_route_to_claude(event2, event2.source)
            for task in list(runner._background_tasks):
                await task

        assert mock_submit.call_count == 2
        assert "already dispatched" not in ack2


class TestThreadAndChannelFallback:
    @pytest.mark.asyncio
    async def test_delivers_into_the_originating_thread_when_present(self):
        runner = _make_runner(config=_routing_config())
        mock_adapter = AsyncMock()
        runner.adapters[Platform.DISCORD] = mock_adapter
        event = _make_event(
            "Ship the pricing page.", chat_id=THREAD_ID, parent_chat_id=CHANNEL_ID,
            message_id="msg-thread",
        )
        event.source.thread_id = THREAD_ID

        bridge_result = BridgeResult(status=BridgeStatus.SUCCESS, response="done")
        with patch("agent.claude_code_tmux_bridge.submit_prompt", return_value=bridge_result):
            await runner._maybe_auto_route_to_claude(event, event.source)
            for task in list(runner._background_tasks):
                await task

        _, kwargs = mock_adapter.send.call_args
        assert kwargs["chat_id"] == THREAD_ID

    @pytest.mark.asyncio
    async def test_falls_back_to_the_channel_when_no_thread(self):
        runner = _make_runner(config=_routing_config())
        mock_adapter = AsyncMock()
        runner.adapters[Platform.DISCORD] = mock_adapter
        event = _make_event("Ship the pricing page.", message_id="msg-no-thread")

        bridge_result = BridgeResult(status=BridgeStatus.SUCCESS, response="done")
        with patch("agent.claude_code_tmux_bridge.submit_prompt", return_value=bridge_result):
            await runner._maybe_auto_route_to_claude(event, event.source)
            for task in list(runner._background_tasks):
                await task

        _, kwargs = mock_adapter.send.call_args
        assert kwargs["chat_id"] == CHANNEL_ID


class TestNoFallbackToHermesOpenAI:
    @pytest.mark.asyncio
    async def test_long_running_task_never_invokes_the_normal_agent_loop(self):
        """Requirement: never report a timeout (or silently swap in Hermes/
        OpenAI implementation work) merely because nothing is still waiting
        on the originating request. The background task is independent of
        the dispatch coroutine returning."""
        runner = _make_runner(config=_routing_config())
        runner.handle_message = AsyncMock()  # stands in for the normal agent-loop entry point
        mock_adapter = AsyncMock()
        runner.adapters[Platform.DISCORD] = mock_adapter
        event = _make_event("Ship the pricing page.", message_id="msg-no-agent-loop")

        bridge_result = BridgeResult(status=BridgeStatus.SUCCESS, response="Shipped it.")
        with patch("agent.claude_code_tmux_bridge.submit_prompt", return_value=bridge_result):
            result = await runner._maybe_auto_route_to_claude(event, event.source)
            for task in list(runner._background_tasks):
                await task

        assert result is not None  # never None -> caller never falls through to the agent loop
        runner.handle_message.assert_not_awaited()
        content = mock_adapter.send.call_args.kwargs["content"]
        assert content == "✅ `claude-momentum`:\nShipped it."


class TestRestartRecovery:
    @pytest.mark.asyncio
    async def test_sweep_marks_still_running_task_interrupted(self):
        from gateway.claude_task_registry import (
            EXECUTION_INTERRUPTED,
            EXECUTION_RUNNING,
            find_task,
            record_task_started,
        )

        runner = _make_runner()
        task_id = "cc-restart-test"
        record_task_started(
            task_id,
            session_key="discord:1:2", platform="discord", chat_id="1", thread_id=None,
            worker="claude-momentum", workspace="/home/michael/code/momentum-studio",
            prompt_preview="do the thing",
        )
        assert find_task(task_id)["execution_state"] == EXECUTION_RUNNING

        # Simulate the owning process being long gone (a previous gateway boot).
        import sqlite3

        from gateway.claude_task_registry import _db_path

        with sqlite3.connect(_db_path()) as conn:
            conn.execute(
                "UPDATE claude_auto_tasks SET owner_pid=? WHERE task_id=?",
                (999999999, task_id),
            )

        count = await runner._sweep_interrupted_claude_tasks()

        assert count == 1
        assert find_task(task_id)["execution_state"] == EXECUTION_INTERRUPTED

    @pytest.mark.asyncio
    async def test_completed_but_undelivered_result_is_recoverable_via_delivery_ledger(self):
        """The 'supported' restart-recovery case: a bridge call that
        finished but crashed before a confirmed send is redelivered by the
        EXISTING gateway.delivery_ledger sweep (gateway/run.py's
        _redeliver_pending_obligations, unchanged), not a new mechanism."""
        from gateway.delivery_ledger import sweep_recoverable

        runner = _make_runner(config=_routing_config())
        mock_adapter = AsyncMock()
        runner.adapters[Platform.DISCORD] = mock_adapter
        event = _make_event("Ship the pricing page.", message_id="msg-crash-before-send")

        # Simulate a send that never got a chance to complete (process died
        # mid-send) by making adapter.send raise.
        mock_adapter.send.side_effect = RuntimeError("process died mid-send")
        bridge_result = BridgeResult(status=BridgeStatus.SUCCESS, response="Shipped it.")
        with patch("agent.claude_code_tmux_bridge.submit_prompt", return_value=bridge_result):
            await runner._maybe_auto_route_to_claude(event, event.source)
            for task in list(runner._background_tasks):
                await task

        # The obligation is recorded and NOT marked delivered -- claimable
        # by the next boot's sweep once this (test) process looks dead.
        claimed = sweep_recoverable(deliverable_platforms={"discord"})
        # This process is alive (it's running the test), so nothing is
        # claimed yet -- demonstrates the row exists and is still owned,
        # not lost. A real restart (dead owner pid) would let it through.
        assert claimed == []

        from gateway.delivery_ledger import _connect

        with _connect() as conn:
            row = conn.execute(
                "SELECT state, content FROM delivery_obligations WHERE chat_id=?",
                (CHANNEL_ID,),
            ).fetchone()
        assert row is not None
        assert row[0] in ("attempting", "failed")
        assert "Shipped it." in row[1]

    @pytest.mark.asyncio
    async def test_interrupted_task_does_not_block_a_genuinely_new_message(self):
        """A follow-up message (even identical text) has its own message_id
        and therefore its own task_id -- an earlier interrupted task must
        not block it."""
        from gateway.claude_task_registry import record_task_started

        runner = _make_runner(config=_routing_config())
        mock_adapter = AsyncMock()
        runner.adapters[Platform.DISCORD] = mock_adapter

        old_session_key = runner._session_key_for_source(
            _make_event("x", message_id="irrelevant").source
        )
        record_task_started(
            "cc-old-interrupted",
            session_key=old_session_key, platform="discord", chat_id=CHANNEL_ID, thread_id=None,
            worker="claude-momentum", workspace="/home/michael/code/momentum-studio",
            prompt_preview="an old, now-interrupted task",
        )

        bridge_result = BridgeResult(status=BridgeStatus.SUCCESS, response="Shipped it.")
        with patch("agent.claude_code_tmux_bridge.submit_prompt", return_value=bridge_result) as mock_submit:
            new_event = _make_event("Ship the pricing page.", message_id="msg-brand-new")
            result = await runner._maybe_auto_route_to_claude(new_event, new_event.source)
            for task in list(runner._background_tasks):
                await task

        assert "already dispatched" not in result
        mock_submit.assert_called_once()


class TestCcTasksCommand:
    @pytest.mark.asyncio
    async def test_no_tasks_yet(self):
        runner = _make_runner()
        event = _make_event("/cc-tasks")
        result = await runner._handle_cc_tasks_command(event)
        assert "No automatically-routed" in result

    @pytest.mark.asyncio
    async def test_lists_task_id_worker_workspace_execution_and_delivery_state(self):
        runner = _make_runner(config=_routing_config())
        mock_adapter = AsyncMock()
        runner.adapters[Platform.DISCORD] = mock_adapter
        event = _make_event("Ship the pricing page.", message_id="msg-for-listing")

        bridge_result = BridgeResult(status=BridgeStatus.SUCCESS, response="Shipped it.")
        with patch("agent.claude_code_tmux_bridge.submit_prompt", return_value=bridge_result):
            await runner._maybe_auto_route_to_claude(event, event.source)
            for task in list(runner._background_tasks):
                await task

        listing = await runner._handle_cc_tasks_command(_make_event("/cc-tasks"))
        assert "claude-momentum" in listing
        assert "/home/michael/code/momentum-studio" in listing
        assert "finished" in listing
        assert "delivered" in listing

    @pytest.mark.asyncio
    async def test_invalid_limit_shows_usage(self):
        runner = _make_runner()
        event = _make_event("/cc-tasks not-a-number")
        result = await runner._handle_cc_tasks_command(event)
        assert "Usage:" in result


# ---------------------------------------------------------------------------
# Production regression: a live auto-routed task reportedly still timed out
# at exactly the bridge's OLD 300s default ("Completion marker not seen
# within 300s"), with an old-format ack (no task ID / no later-delivery
# promise). Every test above proves the *argument-passing* is correct by
# mocking submit_prompt() itself -- which cannot catch a regression in
# submit_prompt/_run_prompt's own poll loop, only in what gets passed to it.
# These two tests drive the REAL bridge (agent.claude_code_tmux_bridge),
# mocking only the low-level tmux subprocess calls, through the REAL
# auto-routing dispatch path -- so they fail if the effective timeout used
# is ever the bridge's bare 300s default instead of the configured
# background-worker timeout, and they fail if the ack is missing a task ID
# or the "delivered later" framing.
# ---------------------------------------------------------------------------


class TestRealBridgePollLoopExceedsOldDefault:
    @pytest.mark.asyncio
    async def test_task_finishing_past_300s_succeeds_not_old_default_timeout(self, monkeypatch):
        """Drives the REAL submit_prompt()/_run_prompt() poll loop (only
        tmux subprocess calls mocked) through the real auto-routing dispatch
        path, using a controlled fake clock. The completion marker appears
        only after ~360 virtual seconds -- past the bridge's OLD 300s
        default. If auto-routing ever regresses to not passing
        background_timeout_seconds through (request.timeout ends up None,
        eff_timeout falls back to the bridge's bare 300s default), the real
        poll loop hits its deadline at 300s and this test fails with a
        TIMEOUT result and "300s" in the delivered content -- exactly the
        reported production symptom.
        """
        import agent.claude_code_tmux_bridge as bridge

        run_id = "f" * 32
        done_marker = f"HERMES_BRIDGE_DONE_{run_id}"

        monkeypatch.setattr(bridge, "_session_exists", lambda s: True)
        monkeypatch.setattr(bridge, "_pane_alive", lambda s: True)
        monkeypatch.setattr(bridge, "_send_keys", lambda s, t: True)
        monkeypatch.setattr(bridge.uuid, "uuid4", lambda: MagicMock(hex=run_id))

        # Fake clock: submit_prompt/_run_prompt only ever call time.monotonic()
        # and time.sleep() -- no real wall-clock waiting happens in this test.
        clock = [0.0]
        monkeypatch.setattr(bridge.time, "monotonic", lambda: clock[0])

        def _fake_sleep(seconds):
            clock[0] += 60.0  # each poll "costs" 60 virtual seconds

        monkeypatch.setattr(bridge.time, "sleep", _fake_sleep)

        # First capture is the pre-send snapshot (empty pane). Then "still
        # working" for several polls -- crossing the 300s mark -- before the
        # completion marker finally appears at the 6th poll (~360s virtual).
        call_count = [0]

        def _fake_capture(session, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return ""
            if call_count[0] < 7:
                return "still working...\n"
            return f"Finished after six minutes.\n{done_marker}\n"

        monkeypatch.setattr(bridge, "_capture_pane", _fake_capture)

        runner = _make_runner(config=_routing_config())  # default background_timeout_seconds (1800s)
        mock_adapter = AsyncMock()
        runner.adapters[Platform.DISCORD] = mock_adapter
        event = _make_event(
            "Wait six minutes without modifying any files. Then reply with "
            "exactly: Finished after six minutes.",
            message_id="msg-real-long-task",
        )

        ack = await runner._maybe_auto_route_to_claude(event, event.source)
        # The ack itself proves the new format shipped: stable task ID and
        # an explicit "posted later" promise, not the old bare ack.
        assert "Task ID: `cc-" in ack
        assert "post" in ack.lower()
        # Nothing has been delivered yet -- the poll loop hasn't run.
        mock_adapter.send.assert_not_called()

        for task in list(runner._background_tasks):
            await task

        assert clock[0] > 300, (
            "the real poll loop must genuinely advance past the bridge's old "
            "300s default for this test to be a meaningful regression check"
        )
        content = mock_adapter.send.call_args.kwargs["content"]
        assert "timed out" not in content.lower()
        assert "300s" not in content
        assert "Finished after six minutes." in content

    @pytest.mark.asyncio
    async def test_same_scenario_would_time_out_at_300s_under_the_old_bridge_default(self, monkeypatch):
        """Companion/control test: proves the previous test's virtual-clock
        scenario is genuinely sensitive to the timeout used -- calling the
        real bridge directly with the bare 300s default (as auto-routing
        would if it regressed to not passing a timeout) DOES time out
        before the marker appears, confirming the two tests exercise the
        exact reported failure mode and its fix."""
        import agent.claude_code_tmux_bridge as bridge

        bridge._session_locks.clear()
        run_id = "e" * 32
        done_marker = f"HERMES_BRIDGE_DONE_{run_id}"

        monkeypatch.setattr(bridge, "_session_exists", lambda s: True)
        monkeypatch.setattr(bridge, "_pane_alive", lambda s: True)
        monkeypatch.setattr(bridge, "_send_keys", lambda s, t: True)
        monkeypatch.setattr(bridge.uuid, "uuid4", lambda: MagicMock(hex=run_id))

        clock = [0.0]
        monkeypatch.setattr(bridge.time, "monotonic", lambda: clock[0])

        def _fake_sleep(seconds):
            clock[0] += 60.0

        monkeypatch.setattr(bridge.time, "sleep", _fake_sleep)

        call_count = [0]

        def _fake_capture(session, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return ""
            if call_count[0] < 7:
                return "still working...\n"
            return f"Finished after six minutes.\n{done_marker}\n"

        monkeypatch.setattr(bridge, "_capture_pane", _fake_capture)

        # No timeout override -- falls back to claude-momentum's registered
        # WorkerConfig.timeout (300.0), exactly like a regression that drops
        # request.timeout back to None.
        result = bridge.submit_prompt("claude-momentum", "wait six minutes")

        assert result.status == BridgeStatus.TIMEOUT
        assert "300s" in result.error
