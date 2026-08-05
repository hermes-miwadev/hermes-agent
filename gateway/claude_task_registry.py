"""Durable execution-state registry for auto-routed Claude Code tasks.

Phase 4 of the Hermes <-> Claude Code integration adds asynchronous
completion delivery: an auto-routed message acknowledges immediately and
the actual (potentially long-running) tmux bridge call happens in a
background ``asyncio`` task. That fire-and-forget shape needs its own
small amount of durable bookkeeping that ``gateway/delivery_ledger.py``
doesn't cover -- delivery_ledger only knows about a *finished* response
that's owed to a chat; it has no concept of "a bridge call is currently
running". This module fills that specific gap:

  - a stable ``task_id`` derived from the triggering Discord message, so a
    duplicate/replayed event (gateway RESUME replay, missed-message
    backfill, retry) can be recognised and skipped rather than starting a
    second tmux submission for the same request;
  - an ``execution_state`` (running -> finished, or interrupted on restart
    recovery) and ``delivery_state`` (pending -> delivered/failed) so an
    operator can see what's actually happening, via ``/cc-tasks``;
  - a startup sweep that marks rows abandoned by a dead gateway process as
    'interrupted' -- see ``sweep_interrupted_tasks`` for exactly what is
    (and, honestly, is not) recoverable across a restart.

Deliberately record-only for the *finished* result's completion delivery:
the actual delivery durability (crash-safe redelivery of a response that
was generated but not yet confirmed-sent) is handled by the existing
``gateway/delivery_ledger.py`` -- reused as-is, not duplicated here. This
module tracks the execution side and stores a copy of the resulting
delivery_state for a unified operator-facing view.

Shares ``state.db`` (same file, WAL, owner pid+start-time liveness pattern)
with ``gateway/delivery_ledger.py`` and ``tools/async_delegation.py`` by
convention, not by importing their (intentionally private) internals --
each module owns its own table and its own small liveness/transaction
helpers.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_DB_LOCK = threading.Lock()

_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_ROWS = 500

EXECUTION_RUNNING = "running"
EXECUTION_FINISHED = "finished"
EXECUTION_INTERRUPTED = "interrupted"

DELIVERY_PENDING = "pending"
DELIVERY_DELIVERED = "delivered"
DELIVERY_FAILED = "failed"

_TASK_ID_PREFIX = "cc-"


def _db_path():
    return get_hermes_home() / "state.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        _initialize_schema(conn)
    except Exception:
        conn.close()
        raise
    return conn


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_state import apply_wal_with_fallback

    apply_wal_with_fallback(conn, db_label="state.db (claude_task_registry)")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS claude_auto_tasks (
            task_id TEXT PRIMARY KEY,
            session_key TEXT NOT NULL,
            platform TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            thread_id TEXT,
            worker TEXT NOT NULL,
            workspace TEXT NOT NULL,
            prompt_preview TEXT,
            execution_state TEXT NOT NULL,
            bridge_status TEXT,
            response_preview TEXT,
            delivery_state TEXT NOT NULL DEFAULT 'pending',
            delivery_error TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            owner_pid INTEGER,
            owner_started_at INTEGER
        )"""
    )


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    """Open, commit/rollback, and ALWAYS close (see delivery_ledger._transaction
    for the leaked-fd history this pattern avoids)."""
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _owner_stamp() -> tuple[int, Optional[int]]:
    pid = os.getpid()
    try:
        from gateway.status import get_process_start_time

        return pid, get_process_start_time(pid)
    except Exception:
        return pid, None


def _owner_alive(pid: Any, started_at: Any) -> bool:
    """True when the recorded owning process still exists (pid + start time).

    Identical liveness contract to gateway.delivery_ledger._owner_alive --
    duplicated rather than imported (that helper is module-private) so this
    module has no hidden coupling to delivery_ledger's internals, only to
    the shared state.db file and PID-liveness convention.
    """
    if not pid:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        from gateway.status import get_process_start_time

        current_start = get_process_start_time(pid)
    except Exception:
        current_start = None
    if current_start is None:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True
    if started_at is None:
        return True
    try:
        return int(current_start) == int(started_at)
    except (TypeError, ValueError):
        return True


def compute_task_id(session_key: str, message_id: str, worker: str) -> str:
    """Stable id derived from the triggering message + worker.

    Deterministic (not random) on purpose: the same Discord message
    replayed later (gateway RESUME replay, missed-message backfill, a
    retry) recomputes the *same* task_id, so ``find_task`` recognises it as
    already handled instead of starting a second tmux submission for what
    is really the same request.
    """
    payload = f"{session_key}|{message_id}|{worker}"
    return _TASK_ID_PREFIX + hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:20]


def _row_to_dict(row: tuple) -> Dict[str, Any]:
    (
        task_id, session_key, platform, chat_id, thread_id, worker, workspace,
        prompt_preview, execution_state, bridge_status, response_preview,
        delivery_state, delivery_error, created_at, updated_at,
        owner_pid, owner_started_at,
    ) = row
    return {
        "task_id": task_id,
        "session_key": session_key,
        "platform": platform,
        "chat_id": chat_id,
        "thread_id": thread_id,
        "worker": worker,
        "workspace": workspace,
        "prompt_preview": prompt_preview,
        "execution_state": execution_state,
        "bridge_status": bridge_status,
        "response_preview": response_preview,
        "delivery_state": delivery_state,
        "delivery_error": delivery_error,
        "created_at": created_at,
        "updated_at": updated_at,
        "owner_pid": owner_pid,
        "owner_started_at": owner_started_at,
    }


_COLUMNS = (
    "task_id, session_key, platform, chat_id, thread_id, worker, workspace, "
    "prompt_preview, execution_state, bridge_status, response_preview, "
    "delivery_state, delivery_error, created_at, updated_at, "
    "owner_pid, owner_started_at"
)


def find_task(task_id: str) -> Optional[Dict[str, Any]]:
    """Return the task row for *task_id*, or None if no such task exists."""
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            f"SELECT {_COLUMNS} FROM claude_auto_tasks WHERE task_id=?",
            (task_id,),
        ).fetchone()
    return _row_to_dict(row) if row else None


def record_task_started(
    task_id: str,
    *,
    session_key: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    worker: str,
    workspace: str,
    prompt_preview: str,
) -> None:
    """Record a newly-dispatched auto-routed task (execution_state='running').

    ``INSERT OR IGNORE``: if a row for this task_id already exists (a race
    between two near-simultaneous duplicate events both passing the
    pre-dispatch ``find_task`` check), the first writer wins and the second
    is a silent no-op here -- the caller's own ``find_task`` check is the
    primary duplicate guard; this is just defense in depth against the
    narrow race between that check and this insert.
    """
    now = time.time()
    pid, started = _owner_stamp()
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            f"""INSERT OR IGNORE INTO claude_auto_tasks
               ({_COLUMNS})
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id, session_key, platform, str(chat_id),
                str(thread_id) if thread_id else None, worker, workspace,
                prompt_preview[:200] if prompt_preview else "",
                EXECUTION_RUNNING, None, None,
                DELIVERY_PENDING, None, now, now, pid, started,
            ),
        )
    _prune()


def mark_task_finished(task_id: str, *, bridge_status: str, response_preview: str) -> None:
    """Record that the bridge call returned (any outcome) -- execution_state='finished'."""
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """UPDATE claude_auto_tasks
               SET execution_state=?, bridge_status=?, response_preview=?, updated_at=?
               WHERE task_id=?""",
            (EXECUTION_FINISHED, bridge_status, (response_preview or "")[:200], time.time(), task_id),
        )


def mark_task_delivery(task_id: str, delivery_state: str, error: str = "") -> None:
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """UPDATE claude_auto_tasks
               SET delivery_state=?, delivery_error=?, updated_at=?
               WHERE task_id=?""",
            (delivery_state, error[:500] if error else None, time.time(), task_id),
        )


def sweep_interrupted_tasks(now: Optional[float] = None) -> List[Dict[str, Any]]:
    """Mark 'running' rows owned by a dead process as 'interrupted'.

    Honest about what this is NOT: it does not resume the underlying
    ``submit_prompt()`` poll loop, which died with the old gateway process.
    Claude Code itself may still be working (or may have already finished)
    inside the tmux session -- but nothing is watching for that completion
    anymore, so it will not be auto-delivered. Returned rows are for
    operator visibility (startup log line + ``/cc-tasks``) so the gap is
    surfaced, not silent. A follow-up message in the same channel/thread
    gets a fresh message_id and therefore a fresh task_id, and proceeds
    normally -- this only prevents automatic recovery of the specific
    interrupted attempt, not future ones.
    """
    now = now if now is not None else time.time()
    interrupted: List[Dict[str, Any]] = []
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM claude_auto_tasks WHERE execution_state=?",
            (EXECUTION_RUNNING,),
        ).fetchall()
        for row in rows:
            record = _row_to_dict(row)
            if _owner_alive(record["owner_pid"], record["owner_started_at"]):
                continue
            conn.execute(
                """UPDATE claude_auto_tasks
                   SET execution_state=?, updated_at=? WHERE task_id=?""",
                (EXECUTION_INTERRUPTED, now, record["task_id"]),
            )
            record["execution_state"] = EXECUTION_INTERRUPTED
            interrupted.append(record)
    return interrupted


def list_recent_tasks(limit: int = 20) -> List[Dict[str, Any]]:
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM claude_auto_tasks ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def _prune(now: Optional[float] = None) -> None:
    now = now if now is not None else time.time()
    cutoff = now - _RETENTION_SECONDS
    try:
        with _transaction() as conn:
            conn.execute(
                """DELETE FROM claude_auto_tasks
                   WHERE execution_state IN (?, ?) AND delivery_state != ? AND updated_at < ?""",
                (EXECUTION_FINISHED, EXECUTION_INTERRUPTED, DELIVERY_PENDING, cutoff),
            )
            total = conn.execute("SELECT COUNT(*) FROM claude_auto_tasks").fetchone()[0]
            excess = max(0, total - _MAX_ROWS)
            if excess:
                conn.execute(
                    """DELETE FROM claude_auto_tasks WHERE task_id IN (
                         SELECT task_id FROM claude_auto_tasks
                         ORDER BY CASE execution_state
                                    WHEN ? THEN 0
                                    WHEN ? THEN 1
                                    ELSE 2
                                  END, updated_at ASC
                         LIMIT ?)""",
                    (EXECUTION_FINISHED, EXECUTION_INTERRUPTED, excess),
                )
    except Exception:
        logger.debug("claude task registry prune failed", exc_info=True)
