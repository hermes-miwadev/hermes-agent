"""Regression tests for Discord auto-thread reliability (Phase 2.1).

Covers three related production issues in the Discord auto-threading path
that a delegation command (e.g. ``/cc-delegate``) can trip over:

1. A failed thread creation used to discard the whole request — including
   an explicit slash command — via the #20243 fail-closed guard. That
   guard is correct for open-ended chat, but a deliberate command must not
   be dropped just because a cosmetic per-task thread couldn't be made; it
   should continue inline in the originating channel instead.
2. ``_auto_create_thread``'s own retry loop could post the
   "Thread created by Hermes" seed message twice for a single request
   (once per attempt) before giving up.
3. Two overlapping attempts to auto-thread the *same* incoming message
   must not race each other into creating (or trying to create) a thread
   twice.

Mirrors the fixture/mocking conventions in
tests/gateway/test_discord_double_dispatch.py.
"""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig

# tests/gateway/conftest.py installs a comprehensive discord mock in
# sys.modules at collection time before this import runs.
import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


class _TextChannel:
    """Fake Discord text channel (not a DM, not a Thread)."""

    def __init__(self, channel_id: int = 100, name: str = "general", guild_name: str = "Test Server"):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(name=guild_name, id=1)
        self.topic = None
        self.send = AsyncMock(return_value=SimpleNamespace(id=999, create_thread=AsyncMock()))

    def history(self, *, limit, before, after=None, oldest_first=None):
        async def _empty():
            return
            yield
        return _empty()


class _Thread:
    """Fake Discord thread (not a DM, not a top-level channel)."""

    def __init__(self, thread_id: int, name: str = "thread", parent=None, guild_name: str = "Test Server"):
        self.id = thread_id
        self.name = name
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        self.guild = getattr(parent, "guild", None) or SimpleNamespace(name=guild_name, id=1)
        self.topic = None

    def history(self, *, limit, before, after=None, oldest_first=None):
        async def _empty():
            return
            yield
        return _empty()


def _make_message(
    *,
    msg_id: int = 42,
    channel,
    content: str = "hello",
    mentions=None,
    author=None,
    msg_type=None,
    create_thread=None,
):
    if author is None:
        author = SimpleNamespace(id=7, display_name="Alice", name="Alice", bot=False)
    return SimpleNamespace(
        id=msg_id,
        content=content,
        mentions=list(mentions or []),
        attachments=[],
        reference=None,
        message_snapshots=None,
        created_at=datetime.now(timezone.utc),
        channel=channel,
        author=author,
        type=(msg_type if msg_type is not None else discord_platform.discord.MessageType.default),
        create_thread=create_thread if create_thread is not None else AsyncMock(),
    )


@pytest.fixture
def adapter(monkeypatch):
    for var in (
        "DISCORD_REQUIRE_MENTION",
        "DISCORD_AUTO_THREAD",
        "DISCORD_NO_THREAD_CHANNELS",
        "DISCORD_FREE_RESPONSE_CHANNELS",
        "DISCORD_ALLOWED_CHANNELS",
        "DISCORD_IGNORED_CHANNELS",
        "DISCORD_HISTORY_BACKFILL",
        "DISCORD_ALLOW_BOTS",
        "DISCORD_IGNORE_NO_MENTION",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")

    config = PlatformConfig(enabled=True, token="***")
    a = DiscordAdapter(config)
    a._client = SimpleNamespace(user=SimpleNamespace(id=999, bot=True))
    a._text_batch_delay_seconds = 0  # disable batching so dispatch is synchronous
    a.handle_message = AsyncMock()
    return a


# ---------------------------------------------------------------------------
# Issue 2: a failed thread must not discard an otherwise-valid command
# ---------------------------------------------------------------------------

class TestThreadFailureFallback:
    @pytest.mark.asyncio
    async def test_command_message_continues_inline_when_thread_creation_fails(self, adapter, monkeypatch):
        """A slash command must reach the agent even if auto-threading fails."""
        channel = _TextChannel(channel_id=100)

        async def fake_auto_create_thread_fail(message):
            return None

        monkeypatch.setattr(adapter, "_auto_create_thread", fake_auto_create_thread_fail)

        message = _make_message(msg_id=42, channel=channel, content="/cc-delegate claude-momentum do the thing")
        result = await adapter._handle_message(message)

        assert result is not False or adapter.handle_message.await_count == 1
        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "/cc-delegate claude-momentum do the thing"
        # Stayed in the originating channel -- no thread was substituted in.
        assert event.source.chat_id == "100"
        # No scary "could not create a Discord thread" notice for a command
        # that actually got through.
        channel.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_plain_chat_message_still_discarded_when_thread_creation_fails(self, adapter, monkeypatch):
        """Non-command free-text mentions keep the #20243 fail-closed behaviour."""
        channel = _TextChannel(channel_id=100)

        async def fake_auto_create_thread_fail(message):
            return None

        monkeypatch.setattr(adapter, "_auto_create_thread", fake_auto_create_thread_fail)

        message = _make_message(msg_id=43, channel=channel, content="hey can you help me with something")
        result = await adapter._handle_message(message)

        assert result is False
        adapter.handle_message.assert_not_awaited()
        channel.send.assert_awaited_once()
        notice = channel.send.await_args.args[0]
        assert "could not create a Discord thread" in notice


# ---------------------------------------------------------------------------
# Issue 3: duplicate/racing thread creation for one incoming message
# ---------------------------------------------------------------------------

class TestDuplicateThreadCreationPrevention:
    @pytest.mark.asyncio
    async def test_inflight_guard_skips_second_attempt_for_same_message(self, adapter, monkeypatch):
        """A message already mid-flight through auto-threading isn't re-attempted."""
        channel = _TextChannel(channel_id=100)
        mock_create = AsyncMock(return_value=_Thread(thread_id=555, parent=channel))
        monkeypatch.setattr(adapter, "_auto_create_thread", mock_create)

        message = _make_message(msg_id=77, channel=channel, content="hello there")
        adapter._auto_threading_inflight.add("77")

        result = await adapter._handle_message(message)

        mock_create.assert_not_awaited()
        # Falls through and still gets handled in the origin channel rather
        # than being dropped.
        assert result is not False
        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.source.chat_id == "100"

    @pytest.mark.asyncio
    async def test_concurrent_handling_of_same_message_creates_thread_once(self, adapter, monkeypatch):
        """Two overlapping _handle_message calls for the same message race safely."""
        channel = _TextChannel(channel_id=100)
        call_count = 0

        async def slow_auto_create_thread(message):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return _Thread(thread_id=888, parent=channel)

        monkeypatch.setattr(adapter, "_auto_create_thread", slow_auto_create_thread)

        # Same underlying message object/id delivered twice in close succession,
        # as could happen with a gateway replay or a retried dispatch.
        message = _make_message(msg_id=99, channel=channel, content="please help")

        await asyncio.gather(
            adapter._handle_message(message),
            adapter._handle_message(message),
        )

        assert call_count == 1, "auto-thread creation must not run twice for one message"


# ---------------------------------------------------------------------------
# Issue 2 (root cause): the retry loop itself must not double-post the seed
# ---------------------------------------------------------------------------

class TestAutoCreateThreadSeedMessageDedup:
    @pytest.mark.asyncio
    async def test_seed_message_sent_only_once_across_both_attempts(self, adapter, monkeypatch):
        """_auto_create_thread must not post 'Thread created by Hermes' twice."""
        monkeypatch.setattr(discord_platform.asyncio, "sleep", AsyncMock())

        seed_msg = SimpleNamespace(create_thread=AsyncMock(side_effect=RuntimeError("still down")))
        channel = SimpleNamespace(send=AsyncMock(return_value=seed_msg))

        message = SimpleNamespace(
            content="do the thing",
            author=SimpleNamespace(display_name="Alice"),
            channel=channel,
            create_thread=AsyncMock(side_effect=RuntimeError("direct create failed")),
        )

        result = await adapter._auto_create_thread(message)

        assert result is None
        assert channel.send.await_count == 1, (
            "the seed message must be sent at most once across both retry attempts"
        )
        assert seed_msg.create_thread.await_count == 2, (
            "the same seed message should be retried, not replaced by a new one"
        )


# ---------------------------------------------------------------------------
# Busy-worker-adjacent sanity: reply messages never attempt auto-threading,
# so they can never trip the in-flight guard or the failure-fallback path.
# ---------------------------------------------------------------------------

class TestReplyMessagesSkipAutoThreadEntirely:
    @pytest.mark.asyncio
    async def test_reply_message_never_calls_auto_create_thread(self, adapter, monkeypatch):
        channel = _TextChannel(channel_id=100)
        mock_create = AsyncMock()
        monkeypatch.setattr(adapter, "_auto_create_thread", mock_create)

        message = _make_message(
            msg_id=101,
            channel=channel,
            content="/cc-delegate claude-momentum do it",
            msg_type=discord_platform.discord.MessageType.reply,
        )
        await adapter._handle_message(message)

        mock_create.assert_not_awaited()
        adapter.handle_message.assert_awaited_once()
