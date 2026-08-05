"""Tests for agent.claude_code_auto_routing.

Pure unit tests for the configurable Discord-channel-to-Claude-Code-worker
automatic routing registry: config parsing/validation, channel-mapping
lookup (including disabled entries and thread-parent matching), the
``chat:`` bypass syntax, and prompt/ack composition. No gateway, no
Discord, no tmux.
"""

from __future__ import annotations

import pytest

from agent.claude_code_auto_routing import (
    CLAUDE_DEFAULT_MODE,
    DEFAULT_STANDARD_INSTRUCTIONS,
    RoutingConfigError,
    RoutingMapping,
    build_auto_routed_prompt,
    find_routing_mapping,
    format_auto_routing_ack,
    load_routing_mappings,
    parse_routing_mapping,
    strip_bypass_prefix,
)
from agent.claude_code_delegate import build_request
from agent.claude_code_tmux_bridge import WorkerConfig, _WORKERS


@pytest.fixture(autouse=True)
def _restore_worker_registry():
    snapshot = dict(_WORKERS)
    yield
    _WORKERS.clear()
    _WORKERS.update(snapshot)


def _entry(**overrides):
    base = {
        "platform": "discord",
        "channel_id": "111222333444555666",
        "enabled": True,
        "worker": "claude-momentum",
        "workspace": "/home/michael/code/momentum-studio",
        "mode": "claude_default",
    }
    base.update(overrides)
    return base


# ── parse_routing_mapping ───────────────────────────────────────────────────

class TestParseRoutingMapping:
    def test_valid_entry_parses(self):
        mapping = parse_routing_mapping(_entry())
        assert mapping.platform == "discord"
        assert mapping.channel_id == "111222333444555666"
        assert mapping.enabled is True
        assert mapping.worker == "claude-momentum"
        assert mapping.workspace == "/home/michael/code/momentum-studio"
        assert mapping.mode == CLAUDE_DEFAULT_MODE
        assert mapping.standard_instructions == DEFAULT_STANDARD_INSTRUCTIONS

    def test_defaults_platform_to_discord(self):
        entry = _entry()
        del entry["platform"]
        assert parse_routing_mapping(entry).platform == "discord"

    def test_defaults_enabled_to_true(self):
        entry = _entry()
        del entry["enabled"]
        assert parse_routing_mapping(entry).enabled is True

    def test_defaults_mode_to_claude_default(self):
        entry = _entry()
        del entry["mode"]
        assert parse_routing_mapping(entry).mode == CLAUDE_DEFAULT_MODE

    def test_non_dict_entry_rejected(self):
        with pytest.raises(RoutingConfigError, match="must be a mapping"):
            parse_routing_mapping("not-a-dict")

    def test_missing_channel_id_rejected(self):
        entry = _entry()
        del entry["channel_id"]
        with pytest.raises(RoutingConfigError, match="channel_id"):
            parse_routing_mapping(entry)

    def test_missing_worker_rejected(self):
        entry = _entry()
        del entry["worker"]
        with pytest.raises(RoutingConfigError, match="worker"):
            parse_routing_mapping(entry)

    def test_missing_workspace_rejected(self):
        entry = _entry()
        del entry["workspace"]
        with pytest.raises(RoutingConfigError, match="workspace"):
            parse_routing_mapping(entry)

    def test_unsupported_mode_rejected(self):
        entry = _entry(mode="auto_classify_everything")
        with pytest.raises(RoutingConfigError, match="unsupported mode"):
            parse_routing_mapping(entry)

    def test_enabled_false_parses(self):
        assert parse_routing_mapping(_entry(enabled=False)).enabled is False

    def test_enabled_accepts_string_yes_no(self):
        assert parse_routing_mapping(_entry(enabled="no")).enabled is False
        assert parse_routing_mapping(_entry(enabled="yes")).enabled is True

    def test_standard_instructions_true_uses_default(self):
        mapping = parse_routing_mapping(_entry(standard_instructions=True))
        assert mapping.standard_instructions == DEFAULT_STANDARD_INSTRUCTIONS

    def test_standard_instructions_false_omits(self):
        mapping = parse_routing_mapping(_entry(standard_instructions=False))
        assert mapping.standard_instructions is None

    def test_standard_instructions_custom_text_used_verbatim(self):
        mapping = parse_routing_mapping(_entry(standard_instructions="Always open a draft PR."))
        assert mapping.standard_instructions == "Always open a draft PR."


# ── load_routing_mappings ────────────────────────────────────────────────────

class TestLoadRoutingMappings:
    def test_loads_from_dict_config(self):
        config = {"claude_routing": [_entry()]}
        mappings = load_routing_mappings(config)
        assert len(mappings) == 1
        assert mappings[0].channel_id == "111222333444555666"

    def test_loads_from_attribute_config(self):
        class _Cfg:
            claude_routing = [_entry()]
        mappings = load_routing_mappings(_Cfg())
        assert len(mappings) == 1

    def test_missing_key_returns_empty(self):
        assert load_routing_mappings({}) == []

    def test_none_config_returns_empty(self):
        assert load_routing_mappings(None) == []

    def test_invalid_entry_skipped_not_fatal(self):
        config = {"claude_routing": [_entry(), {"channel_id": "no-worker-here"}]}
        mappings = load_routing_mappings(config)
        assert len(mappings) == 1
        assert mappings[0].channel_id == "111222333444555666"

    def test_non_list_value_returns_empty(self):
        assert load_routing_mappings({"claude_routing": "oops"}) == []


# ── find_routing_mapping ─────────────────────────────────────────────────────

class TestFindRoutingMapping:
    def setup_method(self):
        self.mappings = [parse_routing_mapping(_entry())]

    def test_matches_configured_channel(self):
        found = find_routing_mapping(self.mappings, platform="discord", channel_id="111222333444555666")
        assert found is not None
        assert found.worker == "claude-momentum"

    def test_unconfigured_channel_returns_none(self):
        found = find_routing_mapping(self.mappings, platform="discord", channel_id="999999999999999999")
        assert found is None

    def test_disabled_mapping_returns_none(self):
        mappings = [parse_routing_mapping(_entry(enabled=False))]
        found = find_routing_mapping(mappings, platform="discord", channel_id="111222333444555666")
        assert found is None

    def test_matches_via_parent_channel_id_for_threads(self):
        # A message inside a thread under the configured channel: chat_id is
        # the thread's own id, parent_channel_id is the configured channel.
        found = find_routing_mapping(
            self.mappings,
            platform="discord",
            channel_id="777_thread_id",
            parent_channel_id="111222333444555666",
        )
        assert found is not None

    def test_platform_mismatch_returns_none(self):
        found = find_routing_mapping(self.mappings, platform="slack", channel_id="111222333444555666")
        assert found is None

    def test_no_channel_id_returns_none(self):
        found = find_routing_mapping(self.mappings, platform="discord", channel_id=None)
        assert found is None

    def test_empty_mappings_returns_none(self):
        assert find_routing_mapping([], platform="discord", channel_id="111222333444555666") is None


# ── strip_bypass_prefix ──────────────────────────────────────────────────────

class TestStripBypassPrefix:
    def test_no_bypass_returns_none(self):
        assert strip_bypass_prefix("Continue getting the site launch-ready.") is None

    def test_bypass_prefix_stripped(self):
        assert strip_bypass_prefix("chat: help me think through this") == "help me think through this"

    def test_bypass_case_insensitive(self):
        assert strip_bypass_prefix("CHAT: help me think") == "help me think"
        assert strip_bypass_prefix("Chat:help me think") == "help me think"

    def test_bypass_with_extra_whitespace(self):
        assert strip_bypass_prefix("  chat  :   help me think") == "help me think"

    def test_bypass_word_mid_sentence_not_treated_as_bypass(self):
        assert strip_bypass_prefix("Let's chat: about the roadmap") is None

    def test_empty_text_returns_none(self):
        assert strip_bypass_prefix("") is None

    def test_bypass_with_no_remaining_text(self):
        assert strip_bypass_prefix("chat:") == ""


# ── build_auto_routed_prompt ─────────────────────────────────────────────────

class TestBuildAutoRoutedPrompt:
    def test_task_leads_for_ack_preview(self):
        mapping = parse_routing_mapping(_entry())
        prompt = build_auto_routed_prompt(mapping, "Ship the pricing page.")
        assert prompt.startswith("Task: Ship the pricing page.")

    def test_default_standard_instructions_included(self):
        mapping = parse_routing_mapping(_entry())
        prompt = build_auto_routed_prompt(mapping, "Ship it.")
        assert "new branch" in prompt
        assert "pull request" in prompt

    def test_standard_instructions_disabled_omits_block(self):
        mapping = parse_routing_mapping(_entry(standard_instructions=False))
        prompt = build_auto_routed_prompt(mapping, "Ship it.")
        assert prompt == "Ship it."

    def test_custom_standard_instructions_used(self):
        mapping = parse_routing_mapping(_entry(standard_instructions="Always ask before deploying."))
        prompt = build_auto_routed_prompt(mapping, "Ship it.")
        assert "Always ask before deploying." in prompt


# ── format_auto_routing_ack ──────────────────────────────────────────────────

class TestFormatAutoRoutingAck:
    def test_includes_activation_framing_and_delegate_ack(self):
        mapping = parse_routing_mapping(_entry())
        request = build_request(mapping.worker, mapping.workspace, "Task: ship it.\n\ninstructions")
        ack = format_auto_routing_ack(mapping, request, task_id="cc-testtask123")
        assert "Automatic routing activated" in ack
        assert CLAUDE_DEFAULT_MODE in ack
        assert "claude-momentum" in ack
        assert "/home/michael/code/momentum-studio" in ack
        assert "cc-testtask123" in ack
        assert "post" in ack.lower()  # confirms the result-posted-later framing


# ── RoutingMapping is a plain, hashable-by-value dataclass ──────────────────

def test_routing_mapping_equality():
    a = parse_routing_mapping(_entry())
    b = parse_routing_mapping(_entry())
    assert a == b
    assert isinstance(a, RoutingMapping)
