"""Automatic Claude Code routing for ``GatewayRunner`` (configurable, Discord).

Extracted as its own mixin (mirrors the god-file decomposition already
applied to ``gateway/authz_mixin.py`` / ``gateway/slash_commands.py``)
rather than folded into either of those: this isn't authorization and it
isn't a slash command -- it's a pre-agent-loop routing decision for
*ordinary* messages, gated entirely by ``agent/claude_code_auto_routing.py``
config. Kept intentionally thin: this module answers "does auto-routing
apply to this message, and if so, what request does it build", then hands
off to the exact same delegation machinery ``/cc-delegate`` uses
(``agent.claude_code_delegate.build_request`` +
``GatewaySlashCommandsMixin._run_cc_delegate_task``) -- no duplicated
validation, locking, or result formatting.

Call site: ``gateway/run.py``'s ``_handle_message``, immediately before it
would otherwise hand the message to the normal Hermes/OpenAI agent loop
(after command dispatch, quick/plugin/skill commands, and the drain gates
have all had their say -- so auto-routing only ever sees messages that
would otherwise become an ordinary agent turn).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource

if TYPE_CHECKING:
    from agent.claude_code_auto_routing import RoutingMapping

logger = logging.getLogger("gateway.run")


class GatewayClaudeAutoRoutingMixin:
    """Configurable automatic Claude Code delegation for ``GatewayRunner``."""

    async def _maybe_auto_route_to_claude(
        self, event: MessageEvent, source: SessionSource
    ) -> Optional[str]:
        """Delegate *event* to Claude Code if this channel is configured for it.

        Returns an immediate operator-facing ack string when auto-routing
        fires (the caller should treat this exactly like any other
        synchronous command reply -- return it and stop). Returns None
        when auto-routing does not apply, meaning the caller should fall
        through to the normal agent loop: unconfigured or disabled
        channel, non-Discord source, bot-authored message, empty text, or
        an explicit ``chat:`` bypass (which also rewrites ``event.text``
        to drop the bypass prefix before falling through).

        Never falls back to the agent loop on a *failed* delegation --
        once a channel is configured and the message isn't bypassed, the
        outcome (success, approval-required, busy, auth failure, timeout,
        extraction failure, or a misconfigured mapping) is always reported
        back to the chat, never silently swapped for substantial Hermes/
        OpenAI implementation work.
        """
        if source.platform != Platform.DISCORD or source.is_bot:
            return None

        # Explicit commands (built-in, quick, plugin, skill/bundle) are
        # never auto-routed -- the caller in gateway/run.py already
        # guarantees this by only reaching here when the message wasn't a
        # command, but this check makes that guarantee true of this
        # function in isolation too, not just by caller convention.
        if event.is_command():
            return None

        # Structured replies and attachment-bearing messages aren't plain
        # implementation requests; auto-routing only handles plain text.
        if event.prompt_response is not None or event.media_urls:
            return None

        from agent.claude_code_auto_routing import (
            find_routing_mapping,
            load_routing_mappings,
        )

        mappings = load_routing_mappings(getattr(self, "config", None))
        if not mappings:
            return None

        mapping = find_routing_mapping(
            mappings,
            platform="discord",
            channel_id=source.chat_id,
            parent_channel_id=source.parent_chat_id,
        )
        if mapping is None:
            return None

        from agent.claude_code_auto_routing import strip_bypass_prefix

        raw_text = event.text or ""
        bypassed_text = strip_bypass_prefix(raw_text)
        if bypassed_text is not None:
            logger.debug(
                "Claude auto-routing bypassed via 'chat:' prefix in channel %s",
                source.chat_id,
            )
            event.text = bypassed_text.strip()
            return None

        user_text = raw_text.strip()
        if not user_text:
            return None

        return await self._dispatch_auto_routed_delegation(event, source, mapping, user_text)

    async def _dispatch_auto_routed_delegation(
        self,
        event: MessageEvent,
        source: SessionSource,
        mapping: "RoutingMapping",
        user_text: str,
    ) -> str:
        """Build and fire the delegation request, returning the ack text.

        Split out from ``_maybe_auto_route_to_claude`` so the "should this
        message be routed" decision and the "go route it" action are each
        independently testable.
        """
        from agent.claude_code_auto_routing import (
            build_auto_routed_prompt,
            format_auto_routing_ack,
        )
        from agent.claude_code_delegate import DelegateValidationError, build_request

        try:
            request = build_request(
                mapping.worker,
                mapping.workspace,
                build_auto_routed_prompt(mapping, user_text),
            )
        except DelegateValidationError as exc:
            logger.warning(
                "Claude auto-routing misconfigured for channel %s (worker=%s, workspace=%s): %s",
                source.chat_id, mapping.worker, mapping.workspace, exc,
            )
            return (
                f"⚠️ Automatic Claude Code routing is misconfigured for this "
                f"channel: {exc} Contact an admin to fix the `claude_routing` "
                f"config entry."
            )

        event_message_id = self._reply_anchor_for_event(event)

        # Fire-and-forget, exactly like /cc-delegate: the bridge call blocks
        # on tmux polling for up to its configured timeout, so it must not
        # hold up the gateway event loop or this message's dispatch.
        _task = asyncio.create_task(
            self._run_cc_delegate_task(request, source, event_message_id=event_message_id)
        )
        self._background_tasks.add(_task)
        _task.add_done_callback(self._background_tasks.discard)

        logger.info(
            "Claude auto-routing activated: channel=%s worker=%s workspace=%s",
            source.chat_id, mapping.worker, mapping.workspace,
        )
        return format_auto_routing_ack(mapping, request)
