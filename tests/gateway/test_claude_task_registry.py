"""Tests for gateway.claude_task_registry.

Pure unit tests for the durable execution-state registry backing Phase 4
(asynchronous completion delivery for auto-routed Claude Code tasks): task
ID derivation, record/find/update, and the startup interrupted-task sweep.
Each test gets its own isolated state.db via the autouse fixture below,
mirroring tests/gateway/test_delivery_ledger.py's explicit _db_path patch
(more robust than an env-var-only redirect: a context-local HERMES_HOME
override, if one is active, takes precedence over the HERMES_HOME env var).
"""

from __future__ import annotations

import sqlite3

import pytest


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path, monkeypatch):
    import gateway.claude_task_registry as registry

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(registry, "_db_path", lambda: home / "state.db")

    yield registry


def _start(registry, task_id="cc-test1", **overrides):
    kwargs = dict(
        session_key="discord:100:200",
        platform="discord",
        chat_id="100",
        thread_id=None,
        worker="claude-momentum",
        workspace="/home/michael/code/momentum-studio",
        prompt_preview="do the thing",
    )
    kwargs.update(overrides)
    registry.record_task_started(task_id, **kwargs)
    return task_id


class TestComputeTaskId:
    def test_deterministic_for_same_inputs(self, _isolated_hermes_home):
        registry = _isolated_hermes_home
        a = registry.compute_task_id("session1", "msg1", "claude-momentum")
        b = registry.compute_task_id("session1", "msg1", "claude-momentum")
        assert a == b

    def test_differs_by_message_id(self, _isolated_hermes_home):
        registry = _isolated_hermes_home
        a = registry.compute_task_id("session1", "msg1", "claude-momentum")
        b = registry.compute_task_id("session1", "msg2", "claude-momentum")
        assert a != b

    def test_differs_by_worker(self, _isolated_hermes_home):
        registry = _isolated_hermes_home
        a = registry.compute_task_id("session1", "msg1", "claude-momentum")
        b = registry.compute_task_id("session1", "msg1", "other-worker")
        assert a != b

    def test_differs_by_session(self, _isolated_hermes_home):
        registry = _isolated_hermes_home
        a = registry.compute_task_id("session1", "msg1", "claude-momentum")
        b = registry.compute_task_id("session2", "msg1", "claude-momentum")
        assert a != b


class TestRecordAndFind:
    def test_find_returns_none_when_absent(self, _isolated_hermes_home):
        registry = _isolated_hermes_home
        assert registry.find_task("cc-nonexistent") is None

    def test_record_then_find_roundtrip(self, _isolated_hermes_home):
        registry = _isolated_hermes_home
        task_id = _start(registry)
        row = registry.find_task(task_id)
        assert row is not None
        assert row["worker"] == "claude-momentum"
        assert row["workspace"] == "/home/michael/code/momentum-studio"
        assert row["execution_state"] == registry.EXECUTION_RUNNING
        assert row["delivery_state"] == registry.DELIVERY_PENDING
        assert row["bridge_status"] is None

    def test_duplicate_start_is_a_noop(self, _isolated_hermes_home):
        """INSERT OR IGNORE: a race between two near-simultaneous duplicate
        dispatches must not overwrite the first writer's row."""
        registry = _isolated_hermes_home
        task_id = _start(registry, chat_id="100")
        _start(registry, task_id=task_id, chat_id="999")  # second writer
        row = registry.find_task(task_id)
        assert row["chat_id"] == "100"  # first writer's value preserved

    def test_long_prompt_preview_is_truncated(self, _isolated_hermes_home):
        registry = _isolated_hermes_home
        task_id = _start(registry, prompt_preview="x" * 500)
        row = registry.find_task(task_id)
        assert len(row["prompt_preview"]) <= 200


class TestMarkTaskFinished:
    def test_updates_execution_state_and_bridge_status(self, _isolated_hermes_home):
        registry = _isolated_hermes_home
        task_id = _start(registry)
        registry.mark_task_finished(task_id, bridge_status="success", response_preview="Shipped it.")
        row = registry.find_task(task_id)
        assert row["execution_state"] == registry.EXECUTION_FINISHED
        assert row["bridge_status"] == "success"
        assert "Shipped it." in row["response_preview"]

    def test_records_non_success_bridge_status(self, _isolated_hermes_home):
        registry = _isolated_hermes_home
        task_id = _start(registry)
        registry.mark_task_finished(task_id, bridge_status="timeout", response_preview="")
        row = registry.find_task(task_id)
        assert row["bridge_status"] == "timeout"
        # delivery_state is untouched by finishing execution -- still pending.
        assert row["delivery_state"] == registry.DELIVERY_PENDING


class TestMarkTaskDelivery:
    def test_marks_delivered(self, _isolated_hermes_home):
        registry = _isolated_hermes_home
        task_id = _start(registry)
        registry.mark_task_delivery(task_id, registry.DELIVERY_DELIVERED)
        row = registry.find_task(task_id)
        assert row["delivery_state"] == registry.DELIVERY_DELIVERED
        assert row["delivery_error"] is None

    def test_marks_failed_with_error(self, _isolated_hermes_home):
        registry = _isolated_hermes_home
        task_id = _start(registry)
        registry.mark_task_delivery(task_id, registry.DELIVERY_FAILED, "adapter unavailable")
        row = registry.find_task(task_id)
        assert row["delivery_state"] == registry.DELIVERY_FAILED
        assert row["delivery_error"] == "adapter unavailable"


class TestListRecentTasks:
    def test_empty_registry_returns_empty_list(self, _isolated_hermes_home):
        registry = _isolated_hermes_home
        assert registry.list_recent_tasks() == []

    def test_returns_most_recently_updated_first(self, _isolated_hermes_home):
        registry = _isolated_hermes_home
        _start(registry, task_id="cc-old", chat_id="1")
        _start(registry, task_id="cc-new", chat_id="2")
        registry.mark_task_finished("cc-old", bridge_status="success", response_preview="")
        tasks = registry.list_recent_tasks(limit=10)
        assert tasks[0]["task_id"] == "cc-old"  # most recently updated (just finished)

    def test_respects_limit(self, _isolated_hermes_home):
        registry = _isolated_hermes_home
        for i in range(5):
            _start(registry, task_id=f"cc-{i}", chat_id=str(i))
        assert len(registry.list_recent_tasks(limit=2)) == 2


class TestSweepInterruptedTasks:
    def test_dead_owner_row_marked_interrupted(self, _isolated_hermes_home):
        registry = _isolated_hermes_home
        task_id = _start(registry)

        with sqlite3.connect(registry._db_path()) as conn:
            conn.execute(
                "UPDATE claude_auto_tasks SET owner_pid=? WHERE task_id=?",
                (999999999, task_id),
            )

        interrupted = registry.sweep_interrupted_tasks()
        assert len(interrupted) == 1
        assert interrupted[0]["task_id"] == task_id
        assert interrupted[0]["execution_state"] == registry.EXECUTION_INTERRUPTED

        row = registry.find_task(task_id)
        assert row["execution_state"] == registry.EXECUTION_INTERRUPTED

    def test_live_owner_row_is_not_swept(self, _isolated_hermes_home):
        """record_task_started stamps the CURRENT (this test process's) pid,
        which is alive by definition -- the row must survive the sweep."""
        registry = _isolated_hermes_home
        task_id = _start(registry)

        interrupted = registry.sweep_interrupted_tasks()

        assert interrupted == []
        row = registry.find_task(task_id)
        assert row["execution_state"] == registry.EXECUTION_RUNNING

    def test_finished_row_is_not_swept(self, _isolated_hermes_home):
        """A task that already finished (even if the process later died)
        is not 'interrupted' -- interruption only applies to still-'running'
        rows; a finished-but-undelivered result is delivery_ledger's job."""
        registry = _isolated_hermes_home
        task_id = _start(registry)
        registry.mark_task_finished(task_id, bridge_status="success", response_preview="done")

        with sqlite3.connect(registry._db_path()) as conn:
            conn.execute(
                "UPDATE claude_auto_tasks SET owner_pid=? WHERE task_id=?",
                (999999999, task_id),
            )

        interrupted = registry.sweep_interrupted_tasks()
        assert interrupted == []
        row = registry.find_task(task_id)
        assert row["execution_state"] == registry.EXECUTION_FINISHED
