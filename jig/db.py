"""SQLite database: one local file holding all of Jig's state.

All access goes through a single connection guarded by a lock. Statements are
short, so they run inline on the event loop; this is a deliberate simplicity
trade-off for the foundations.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL,
    plan_json TEXT,
    result TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    goal_id TEXT REFERENCES goals(id),
    schedule_id TEXT REFERENCES schedules(id),
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    depends_on_json TEXT NOT NULL DEFAULT '[]',
    run_after TEXT NOT NULL,
    result TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS tasks_status ON tasks(status, run_after);

CREATE TABLE IF NOT EXISTS schedules (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    prompt TEXT NOT NULL,
    mode TEXT NOT NULL,
    interval_s REAL NOT NULL,
    next_run_at TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    last_task_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    messages_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    task_id TEXT REFERENCES tasks(id),
    session_id TEXT,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    messages_json TEXT NOT NULL DEFAULT '[]',
    final TEXT,
    error TEXT,
    steps INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS run_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    idx INTEGER NOT NULL,
    type TEXT NOT NULL,
    name TEXT,
    input_json TEXT,
    output_json TEXT,
    status TEXT NOT NULL,
    error TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS run_steps_run ON run_steps(run_id, idx);

CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    task_id TEXT,
    run_id TEXT,
    tool_call_id TEXT,
    tool TEXT NOT NULL,
    args_json TEXT NOT NULL,
    reasons_json TEXT NOT NULL,
    sentinel_json TEXT,
    status TEXT NOT NULL,
    note TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    resolved_json TEXT
);
CREATE INDEX IF NOT EXISTS approvals_status ON approvals(status);
CREATE UNIQUE INDEX IF NOT EXISTS approvals_call ON approvals(run_id, tool_call_id);

CREATE TABLE IF NOT EXISTS rules (
    id TEXT PRIMARY KEY,
    tool TEXT NOT NULL,
    arg TEXT,
    pattern TEXT,
    decision TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    note TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'fact',
    tags TEXT NOT NULL DEFAULT '',
    source TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, tags, content='memories', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags) VALUES (new.id, new.content, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
    VALUES ('delete', old.id, old.content, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
    VALUES ('delete', old.id, old.content, old.tags);
    INSERT INTO memories_fts(rowid, content, tags) VALUES (new.id, new.content, new.tags);
END;

CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Keyed long-term state (jig.facts): author preferences, book-world state and project continuity, keyed by
-- scope instead of flat. The `memories` table above stays the user-facing store and is not repurposed.
-- UNIQUE(scope, subject, content) deduplicates exact repeats, which memories deliberately does not do.
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope TEXT NOT NULL DEFAULT 'author',
    subject TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT 'preference',
    content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    supersedes INTEGER REFERENCES facts(id),
    source TEXT,
    source_run_id TEXT,
    source_task_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(scope, subject, content)
);
CREATE INDEX IF NOT EXISTS facts_scope ON facts(scope, status, id);
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    content, subject, content='facts', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, subject) VALUES (new.id, new.content, new.subject);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, subject)
    VALUES ('delete', old.id, old.content, old.subject);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, subject)
    VALUES ('delete', old.id, old.content, old.subject);
    INSERT INTO facts_fts(rowid, content, subject) VALUES (new.id, new.content, new.subject);
END;

-- The skill bank (jig.skills): agent-authored procedures. A candidate is never injected into a prompt; a
-- candidate becomes active only through an explicit human action (jig.skills and the PATCH /skills route).
CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    body TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'author',
    status TEXT NOT NULL DEFAULT 'candidate',
    source TEXT,
    source_run_id TEXT,
    source_task_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    promoted_at TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
    name, title, summary, body, content='skills', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS skills_ai AFTER INSERT ON skills BEGIN
    INSERT INTO skills_fts(rowid, name, title, summary, body) VALUES (new.id, new.name, new.title, new.summary, new.body);
END;
CREATE TRIGGER IF NOT EXISTS skills_ad AFTER DELETE ON skills BEGIN
    INSERT INTO skills_fts(skills_fts, rowid, name, title, summary, body)
    VALUES ('delete', old.id, old.name, old.title, old.summary, old.body);
END;
CREATE TRIGGER IF NOT EXISTS skills_au AFTER UPDATE ON skills BEGIN
    INSERT INTO skills_fts(skills_fts, rowid, name, title, summary, body)
    VALUES ('delete', old.id, old.name, old.title, old.summary, old.body);
    INSERT INTO skills_fts(rowid, name, title, summary, body) VALUES (new.id, new.name, new.title, new.summary, new.body);
END;

-- The trajectory index (jig.trajectory): a bounded, queryable index over what runs did, so a past run can
-- inform a later one. Rows are never edited (the trigger below refuses it); they are deleted only by the
-- bounded prune and by the forget-everything wipe, both of which remove whole rows and audit only counts.
-- This is deliberately not the audit rule: audit is immutable forever by promise, the trajectory is a
-- working index. It holds ids, names, statuses and a short redacted digest, never payloads.
CREATE TABLE IF NOT EXISTS trajectory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    run_id TEXT,
    task_id TEXT,
    tool TEXT,
    status TEXT,
    error_type TEXT,
    digest TEXT NOT NULL DEFAULT '',
    run_kind TEXT,
    mode TEXT
);
CREATE INDEX IF NOT EXISTS trajectory_run ON trajectory(run_id, id);
CREATE INDEX IF NOT EXISTS trajectory_tool ON trajectory(tool, id);
CREATE INDEX IF NOT EXISTS trajectory_status ON trajectory(status, id);
CREATE TRIGGER IF NOT EXISTS trajectory_no_update BEFORE UPDATE ON trajectory BEGIN
    SELECT RAISE(ABORT, 'trajectory log is append-only');
END;

CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    actor TEXT NOT NULL,
    task_id TEXT,
    run_id TEXT,
    summary TEXT NOT NULL,
    data_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS audit_kind ON audit(kind, id);
CREATE INDEX IF NOT EXISTS audit_task ON audit(task_id, id);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit BEGIN
    SELECT RAISE(ABORT, 'audit log is append-only');
END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit BEGIN
    SELECT RAISE(ABORT, 'audit log is append-only');
END;

CREATE TABLE IF NOT EXISTS secrets (
    name TEXT PRIMARY KEY,
    backend TEXT NOT NULL,
    ciphertext BLOB,
    allowed_tools_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Connected accounts (jig.connectors). Tokens live only in the vault; this is the metadata.
CREATE TABLE IF NOT EXISTS connections (
    provider TEXT PRIMARY KEY,
    account TEXT,
    access TEXT NOT NULL,
    scopes_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL,
    last_error TEXT,
    connected_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- MCP servers the user added in Settings. The program and its arguments are not secrets; tokens
-- live only in the vault as mcp.<id>.env.<NAME>.
CREATE TABLE IF NOT EXISTS mcp_servers (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    command TEXT NOT NULL,
    args_json TEXT NOT NULL,
    access TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    last_error TEXT,
    tool_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

# Columns added after a table first shipped: (table, column, definition).
_ADDED_COLUMNS = (("approvals", "resolved_json", "TEXT"),
                  # A schedule's repeat (jig.recurrence); NULL for schedules made before calendar times.
                  ("schedules", "repeat_json", "TEXT"),
                  ("schedules", "timezone", "TEXT"),
                  ("schedules", "created_by", "TEXT"),
                  ("notes", "updated_at", "TEXT"),
                  # The goal a planning run planned; NULL for chat and task runs.
                  ("runs", "goal_id", "TEXT"),
                  # How a background task ended (jig.agent.prompts.OUTCOME_SCHEMA); NULL if it was not recorded.
                  ("tasks", "outcome_json", "TEXT"))


def now() -> datetime:
    return datetime.now(UTC)


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="milliseconds")


def now_iso() -> str:
    return iso(now())


def later_iso(seconds: float) -> str:
    return iso(now() + timedelta(seconds=seconds))


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class Database:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        # Deleted rows (a forgotten memory, say) are overwritten with zeros, not left in free pages.
        self._conn.execute("PRAGMA secure_delete=ON")
        self._conn.executescript(SCHEMA)
        for table, column, definition in _ADDED_COLUMNS:
            existing = {r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        # Recorded, never branched on (as always): an upgraded database records the version it now has.
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(SCHEMA_VERSION),),
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def query(self, sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def one(self, sql: str, params: tuple | dict = ()) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def checkpoint(self) -> bool:
        """Copy the write-ahead log into the database file and truncate it, so deleted text is not left in the
        log. False only if another connection (a CLI command, say) was reading; the next checkpoint clears it."""
        busy, _log, _done = self.one("PRAGMA wal_checkpoint(TRUNCATE)").values()  # type: ignore[union-attr]
        return busy == 0

    def vacuum(self) -> bool:
        """Rebuild the database file without its free pages, which can still hold text deleted before
        secure_delete was switched on, then clear the write-ahead log. Returns what checkpoint() returns."""
        with self._lock:
            self._conn.execute("VACUUM")
        return self.checkpoint()
