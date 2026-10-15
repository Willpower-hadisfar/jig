"""The skill bank: procedures the agent writes and a human promotes.

A skill is a short procedural document — how to do something that took several iterations to get right —
written by the agent as a *candidate*. A candidate is never injected into any prompt and never steers
anything: it becomes ``active`` only through an explicit human action (``set_status`` here, reached from
``PATCH /skills/{id}``; automatic promotion was deliberately rejected in the design record, because a skill
is agent-authored text that will later steer the agent's behaviour). Active skills enter a later turn's
context as a name-and-summary index only; the full body is read on demand with the ``skill_read`` tool.

Skills say *how to produce* something and never contain the thing itself: the body is capped and a body
that quotes long passages of prose (which would be manuscript text leaking into a new store) is refused
with a clear error. Privacy works exactly as it does for memories: same database, same secure_delete,
same wipe in the forget-everything endpoint; no skill content ever reaches the audit log.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from .db import Database, now_iso
from .errors import NotFound

SKILL_STATUSES = ("candidate", "active", "retired", "rejected")
NAME_CHARS = 64
TITLE_CHARS = 120
SUMMARY_CHARS = 240
# A storage cap, separate from the smaller injection cap in jig.agent.prompts: a skill that fits the bank
# but is longer than what a turn may carry is still readable in parts with skill_read.
BODY_CHARS = 4_000
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
# A quoted span of more than ~120 characters, or more than a quarter of the body inside quotation marks,
# is a passage of prose, not a procedure. Neither belongs in a skill.
_QUOTED_RUN = re.compile(r"[\"“”'’][^\"“”'’\n]{120,}[\"“”'’]?|\"[^\"\n]{120,}")
_QUOTED_ANY = re.compile(r"\"[^\"\n]*\"|“[^”\n]*”")
_PROSE_QUOTED_SHARE = 0.25


def _looks_like_prose(body: str) -> bool:
    quoted = sum(len(m.group()) for m in _QUOTED_ANY.finditer(body))
    return bool(_QUOTED_RUN.search(body)) or (quoted > _PROSE_QUOTED_SHARE * max(len(body), 1))


class SkillBank:
    def __init__(self, db: Database):
        self.db = db

    def propose(self, *, name: str, title: str, summary: str, body: str, scope: str = "author",
                source: str | None = None, run_id: str | None = None,
                task_id: str | None = None) -> dict[str, Any]:
        """Write a candidate skill. It lands as ``status='candidate'`` whatever the caller would like:
        this method has no way to make a skill active."""
        name = (name or "").strip()
        if not _NAME.match(name):
            raise ValueError(f"a skill name must be 2-{NAME_CHARS} characters of lower-case letters, digits, "
                             "'-' or '_' (got something like " + repr(name[:NAME_CHARS + 8]) + ")")
        title = " ".join((title or "").split())
        if not title or len(title) > TITLE_CHARS:
            raise ValueError(f"a skill title must be 1-{TITLE_CHARS} characters")
        summary = " ".join((summary or "").split())
        if not summary or len(summary) > SUMMARY_CHARS:
            raise ValueError(f"a skill summary must be 1-{SUMMARY_CHARS} characters (this one is "
                             f"{len(summary)})")
        body = (body or "").strip()
        if not body or len(body) > BODY_CHARS:
            raise ValueError(f"a skill body must be 1-{BODY_CHARS} characters (this one is {len(body)})")
        if _looks_like_prose(body):
            raise ValueError("a skill body must be a procedure, not quoted prose: it quotes a passage too long "
                             "to be instructions (over ~120 characters inside quotation marks, or more than a "
                             "quarter of the body quoted). Describe how to produce the text; never include it.")
        ts = now_iso()
        try:
            cur = self.db.execute(
                "INSERT INTO skills(name, title, summary, body, scope, status, source, source_run_id, "
                "source_task_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'candidate', ?, ?, ?, ?, ?)",
                (name, title, summary, body, scope, source, run_id, task_id, ts, ts),
            )
        except sqlite3.IntegrityError:
            existing = self.get_by_name(name)
            raise ValueError(f"a skill named {name!r} already exists (id {existing['id']}, status "
                             f"{existing['status']}); propose a differently-named candidate or ask the user "
                             "to change that one") from None
        return self.get(int(cur.lastrowid))

    def get(self, skill_id: int) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM skills WHERE id = ?", (skill_id,))
        if row is None:
            raise NotFound(f"skill {skill_id} does not exist")
        return row

    def get_by_name(self, name: str) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM skills WHERE name = ?", (name,))
        if row is None:
            raise NotFound(f"no skill named {name!r}")
        return row

    def list(self, *, status: str | None = None, scope: str | None = None,
             limit: int = 50) -> list[dict[str, Any]]:
        sql, params = ("SELECT id, name, title, summary, scope, status, created_at, updated_at, promoted_at "
                       "FROM skills WHERE 1=1"), []
        if status:
            sql += " AND status = ?"
            params.append(status)
        if scope:
            sql += " AND scope = ?"
            params.append(scope)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return self.db.query(sql, tuple(params))

    def search(self, query: str, *, status: str = "active", limit: int = 10) -> list[dict[str, Any]]:
        tokens = re.findall(r"\w+", query or "")
        if not tokens:
            return []
        match = " OR ".join(f'"{t}"' for t in tokens)
        rows = self.db.query(
            "SELECT s.id, s.name, s.title, s.summary, s.scope, s.status, "
            "bm25(skills_fts) AS score FROM skills_fts JOIN skills s ON s.id = skills_fts.rowid "
            "WHERE skills_fts MATCH ? AND s.status = ? ORDER BY score LIMIT ?",
            (match, status, limit),
        )
        for r in rows:
            r["score"] = round(-r["score"], 4)
        return rows

    def set_status(self, skill_id: int, status: str) -> dict[str, Any]:
        """Promote, retire or reject a skill. This is the only path that makes one active, and it belongs
        to the human: nothing the agent calls reaches it."""
        if status not in SKILL_STATUSES:
            raise ValueError(f"status must be one of {', '.join(SKILL_STATUSES)}, not {status!r}")
        skill = self.get(skill_id)
        if skill["status"] == status:
            return skill
        ts = now_iso()
        if status == "active" and not skill["promoted_at"]:
            self.db.execute("UPDATE skills SET status = ?, updated_at = ?, promoted_at = ? WHERE id = ?",
                            (status, ts, ts, skill_id))
        else:
            self.db.execute("UPDATE skills SET status = ?, updated_at = ? WHERE id = ?", (status, ts, skill_id))
        return self.get(skill_id)

    def active_index(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """The active skills, oldest first: what a turn's index is built from."""
        return self.db.query("SELECT id, name, title, summary, scope FROM skills WHERE status = 'active' "
                             "ORDER BY id LIMIT ?", (limit,))

    def count(self) -> int:
        return int(self.db.one("SELECT COUNT(*) AS n FROM skills")["n"])  # type: ignore[index]

    def wipe(self) -> dict[str, Any]:
        """Forget every skill at every status and its search index, the way memories do, so the forget-
        everything endpoint leaves nothing of them in the database files."""
        with self.db.transaction() as conn:
            n = int(conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0])
            conn.execute("DELETE FROM skills")
            conn.execute("INSERT INTO skills_fts(skills_fts) VALUES ('delete-all')")
            conn.execute("INSERT INTO skills_fts(skills_fts) VALUES ('optimize')")
        return {"forgotten": n, "wal_cleared": self.db.checkpoint()}
