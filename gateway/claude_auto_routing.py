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

        Idempotent per triggering Discord message: the task_id is derived
        deterministically from (session, message_id, worker), so a
        duplicate/replayed event (gateway RESUME replay, missed-message
        backfill, a retry) for the exact same message recomputes the same
        task_id and is recognised -- via gateway.claude_task_registry -- as
        already dispatched, rather than starting a second tmux submission.
        A genuinely new message (even with identical text) has its own
        message_id and therefore its own task_id, and proceeds normally.
        """
        from agent.claude_code_auto_routing import (
            build_auto_routed_prompt,
            format_auto_routing_ack,
            format_duplicate_task_notice,
        )
        from agent.claude_code_delegate import DelegateValidationError, build_request
        from gateway.claude_task_registry import (
            compute_task_id,
            find_task,
            record_task_started,
        )

        session_key = self._session_key_for_source(source)
        message_ref = event.message_id or f"no-message-id-{id(event)}"
        task_id = compute_task_id(session_key, message_ref, mapping.worker)

        existing = await asyncio.to_thread(find_task, task_id)
        if existing is not None:
            logger.info(
                "Claude auto-routing: duplicate event suppressed for task=%s "
                "(execution=%s, delivery=%s)",
                task_id, existing.get("execution_state"), existing.get("delivery_state"),
            )
            return format_duplicate_task_notice(task_id, existing)

        try:
            request = build_request(
                mapping.worker,
                mapping.workspace,
                build_auto_routed_prompt(mapping, user_text),
                timeout=mapping.background_timeout_seconds,
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

        await asyncio.to_thread(
            record_task_started,
            task_id,
            session_key=session_key,
            platform="discord",
            chat_id=source.chat_id,
            thread_id=source.thread_id,
            worker=mapping.worker,
            workspace=mapping.workspace,
            prompt_preview=user_text,
        )

        event_message_id = self._reply_anchor_for_event(event)

        # Fire-and-forget: acknowledge now, run the (potentially long)
        # bridge call in the background, and deliver the result whenever it
        # actually finishes -- up to mapping.background_timeout_seconds
        # later. This coroutine returning (and the originating Discord
        # request/interaction being "done" from Discord's perspective) has
        # no bearing on the background task: it is an independent
        # asyncio.create_task, not awaited here, so it is never mistaken
        # for timed-out just because nothing is still waiting on it.
        _task = asyncio.create_task(
            self._run_cc_delegate_task(
                request, source,
                event_message_id=event_message_id,
                task_id=task_id,
                message_ref=message_ref,
            )
        )
        self._background_tasks.add(_task)
        _task.add_done_callback(self._background_tasks.discard)

        logger.info(
            "Claude auto-routing activated: task=%s channel=%s worker=%s workspace=%s",
            task_id, source.chat_id, mapping.worker, mapping.workspace,
        )
        return format_auto_routing_ack(mapping, request, task_id=task_id)

    async def _handle_cc_tasks_command(self, event: MessageEvent) -> str:
        """Handle /cc-tasks [limit] -- operator-facing auto-routed task state.

        Lists recent gateway.claude_task_registry rows: task ID, worker,
        workspace, execution state (running/finished/interrupted), and
        delivery state (pending/delivered/failed). Read-only; does not
        retry, cancel, or otherwise act on any task.
        """
        from gateway.claude_task_registry import list_recent_tasks

        raw_limit = event.get_command_args().strip()
        limit = 10
        if raw_limit:
            try:
                limit = max(1, min(50, int(raw_limit)))
            except ValueError:
                return f"Usage: /cc-tasks [limit] -- {raw_limit!r} is not a number."

        tasks = await asyncio.to_thread(list_recent_tasks, limit)
        if not tasks:
            return "No automatically-routed Claude Code tasks recorded yet."

        lines = [f"**Recent Claude Code auto-routed tasks** (showing {len(tasks)}):"]
        for task in tasks:
            lines.append(
                f"`{task['task_id']}` — worker `{task['worker']}`, "
                f"workspace `{task['workspace']}`, "
                f"execution: `{task['execution_state']}`"
                + (f" ({task['bridge_status']})" if task.get("bridge_status") else "")
                + f", delivery: `{task['delivery_state']}`"
            )
        return "\n".join(lines)

    async def _sweep_interrupted_claude_tasks(self) -> int:
        """Startup recovery: mark auto-routed tasks abandoned by a dead
        gateway process as 'interrupted' (does not resume them -- see
        gateway.claude_task_registry.sweep_interrupted_tasks for exactly
        what is and is not recoverable). Returns the count found, for the
        caller's boot-log line.
        """
        from gateway.claude_task_registry import sweep_interrupted_tasks

        try:
            interrupted = await asyncio.to_thread(sweep_interrupted_tasks)
        except Exception:
            logger.debug("claude task registry startup sweep failed", exc_info=True)
            return 0
        for task in interrupted:
            logger.warning(
                "Claude auto-routing: task %s was still running when the previous "
                "gateway process exited -- marked interrupted (worker=%s, "
                "workspace=%s). Its result, if any, was not recovered; a new "
                "message will start a fresh task.",
                task["task_id"], task["worker"], task["workspace"],
            )
        return len(interrupted)
