"""Keyed long-term state: author preferences, book-world state and project continuity.

``memories`` stays exactly what it has always been — the flat, user-managed store the README promises
("Jig never rewrites your memories by itself"). ``facts`` is its keyed, machine-maintained complement: every
row carries a ``scope`` (``author``, ``book:<slug>`` or ``project:<slug>``), a ``subject`` and a ``kind``, so
"the author prefers British English" (a preference), "Wren is the protagonist" (book-world state) and "act two
is still unwritten" (project continuity) are told apart instead of pooling into one flat list.

Like memory: no embeddings, full-text search is the only index, and no fact's content is ever written to the
audit log. Exact repeats are deduplicated by ``UNIQUE(scope, subject, content)``; a genuine correction
supersedes the old row rather than deleting it, so the history stays visible. Rows the user wrote
(``source = 'user'``) are never superseded by this path — changing one remains the user's, as ever.
"""

from __future__ import annotations

import re
from typing import Any

from .db import Database, now_iso
from .errors import NotFound

# What a fact is about, in a small fixed vocabulary: a preference of the author, a fact of the book's world,
# the state of the work in progress, or a question a failed run left open.
FACT_KINDS = ("preference", "world", "continuity", "open_question")
# One sentence. The same cap the prompt renders at, so nothing stored is ever silently cut in front of the model.
FACT_CHARS = 500
SUBJECT_CHARS = 80
# 'author', or an opaque slug the author or the agent names: book:<slug> / project:<slug>.
_SCOPE = re.compile(r"^(?:author|book:[a-z0-9][a-z0-9_-]{0,63}|project:[a-z0-9][a-z0-9_-]{0,63})$")
_WORD = re.compile(r"\w+", re.UNICODE)


class FactStore:
    def __init__(self, db: Database):
        self.db = db

    def remember(self, *, content: str, kind: str = "preference", scope: str = "author", subject: str = "",
                 source: str | None = None, run_id: str | None = None, task_id: str | None = None,
                 supersedes: int | None = None) -> dict[str, Any]:
        """Save one keyed fact. An exact repeat (same scope, subject and content) returns the row that is
        already there instead of growing a second one. ``supersedes`` marks an earlier agent-authored row
        superseded; rows the user wrote are never touched."""
        content = " ".join((content or "").split())
        if not content:
            raise ValueError("a fact's content must not be empty")
        if len(content) > FACT_CHARS:
            raise ValueError(f"a fact must be one sentence of at most {FACT_CHARS} characters "
                             f"(this one is {len(content)})")
        if kind not in FACT_KINDS:
            raise ValueError(f"kind must be one of {', '.join(FACT_KINDS)}, not {kind!r}")
        if not _SCOPE.match(scope):
            raise ValueError("scope must be 'author', 'book:<slug>' or 'project:<slug>' (lower-case letters, "
                             "digits, '-' and '_')")
        subject = (subject or "").strip()
        if len(subject) > SUBJECT_CHARS:
            raise ValueError(f"a subject must be at most {SUBJECT_CHARS} characters (this one is {len(subject)})")
        ts = now_iso()
        if supersedes is not None:
            old = self.get(supersedes)
            if old["source"] == "user":
                raise ValueError(f"fact {supersedes} was written by the user and cannot be superseded by Jig; "
                                 "ask the user to change it")
        with self.db.transaction() as conn:
            if supersedes is not None:
                conn.execute("UPDATE facts SET status = 'superseded', updated_at = ? WHERE id = ? AND status = 'active'",
                             (ts, supersedes))
            conn.execute(
                "INSERT OR IGNORE INTO facts(scope, subject, kind, content, status, supersedes, source, "
                "source_run_id, source_task_id, created_at, updated_at) VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?)",
                (scope, subject, kind, content, supersedes, source, run_id, task_id, ts, ts),
            )
        # Whether the insert landed or the exact fact was already there, the unique key names one row.
        row = self.db.one("SELECT * FROM facts WHERE scope = ? AND subject = ? AND content = ?",
                          (scope, subject, content))
        assert row is not None  # the INSERT OR IGNORE above guarantees the key exists
        return row

    def get(self, fact_id: int) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM facts WHERE id = ?", (fact_id,))
        if row is None:
            raise NotFound(f"fact {fact_id} does not exist")
        return row

    def list(self, *, scope: str | None = None, kind: str | None = None, status: str = "active",
             limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM facts WHERE 1=1", []
        if status:
            sql += " AND status = ?"
            params.append(status)
        if scope:
            sql += " AND scope = ?"
            params.append(scope)
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        return [self._out(r) for r in self.db.query(sql, tuple(params))]

    def search(self, query: str, *, scope: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Active facts whose words match, best first. Superseded rows never come back."""
        tokens = _WORD.findall(query or "")
        if not tokens:
            return []
        # Quote every token so user text can never inject FTS5 syntax (as in jig.memory).
        match = " OR ".join(f'"{t}"' for t in tokens)
        sql = ("SELECT f.*, bm25(facts_fts) AS score FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid "
               "WHERE facts_fts MATCH ? AND f.status = 'active'")
        params: list[Any] = [match]
        if scope:
            sql += " AND f.scope IN (?, 'author')"
            params.append(scope)
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)
        out = []
        for row in self.db.query(sql, tuple(params)):
            score = row.pop("score", 0.0)
            out.append({**self._out(row), "score": round(-score, 4)})
        return out

    def count(self, *, status: str | None = None) -> int:
        if status:
            row = self.db.one("SELECT COUNT(*) AS n FROM facts WHERE status = ?", (status,))
        else:
            row = self.db.one("SELECT COUNT(*) AS n FROM facts")
        return int(row["n"])  # type: ignore[index]

    def wipe(self) -> dict[str, Any]:
        """Forget every fact, active or superseded, and its search index, the way memories do: the freed
        pages are zeroed (``secure_delete``) and the write-ahead log is checkpointed and truncated."""
        with self.db.transaction() as conn:
            n = int(conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
            conn.execute("DELETE FROM facts")
            conn.execute("INSERT INTO facts_fts(facts_fts) VALUES ('delete-all')")
            conn.execute("INSERT INTO facts_fts(facts_fts) VALUES ('optimize')")
        return {"forgotten": n, "wal_cleared": self.db.checkpoint()}

    @staticmethod
    def _out(row: dict[str, Any]) -> dict[str, Any]:
        return row
