"""Tests for GatewayClaudeAutoRoutingMixin (configurable Discord auto-routing).

Covers GatewayRunner._maybe_auto_route_to_claude (the routing decision) and
_dispatch_auto_routed_delegation (building + firing the delegation), reusing
the same _make_runner/_make_event conventions as
tests/gateway/test_cc_delegate_command.py and
tests/gateway/test_discord_thread_reliability.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

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
):
    source = SessionSource(
        platform=platform,
        user_id="42",
        chat_id=chat_id,
        parent_chat_id=parent_chat_id,
        is_bot=is_bot,
        user_name="Michael",
    )
    return MessageEvent(text=text, source=source, media_urls=list(media_urls or []))


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
