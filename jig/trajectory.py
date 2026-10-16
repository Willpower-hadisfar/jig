"""The trajectory index: a bounded, queryable record of what runs did, so a past run can inform a later one.

The full transcript already exists in ``runs`` / ``run_steps`` (with every tool argument and result, and the
user's own to delete). What was missing was a *reader*: an index small and structured enough to ask
"have I tried this tool and failed before, and how did it end?". This is that index.

The rules it keeps:

* **Rows are added, never edited.** A SQLite trigger refuses updates, the way the audit log refuses them,
  so what a run recorded cannot be quietly rewritten later. Whole rows may be removed — by the bounded
  prune below and by the forget-everything wipe — which is the opposite of the audit promise on purpose:
  audit is immutable forever by promise and holds no content; the trajectory is a working index, so it is
  allowed to be bounded, and both removals take whole rows and audit only counts.
* **No payload text.** A row carries ids, tool names, statuses, error *types* and a short digest built
  from those, run through the instance's redactions before it is stored. Nothing here may ever hold what
  a tool was given or what it returned: that is what keeps it safe to read a year later.
"""

from __future__ import annotations

from typing import Any, Callable

from .db import Database, now_iso

TRAJECTORY_KINDS = ("step", "failure", "outcome", "skill.candidate", "skill.used")
DIGEST_CHARS = 160
# The index is a working record, not a promise: past this many rows the oldest whole rows go (at start-up,
# audited as a count). The full transcript in run_steps is where the detail lives, as ever.
MAX_ROWS = 200_000


class TrajectoryLog:
    def __init__(self, db: Database, redact: Callable[[str], str] | None = None):
        self.db = db
        # Vault.redact with the instance's always-redact map (model and connector keys), passed in by the
        # runtime: the digest is built from metadata, and this makes sure a name that happens to be a
        # secret value never rides through it.
        self.redact = redact or (lambda text: text)

    def record(self, kind: str, *, run_id: str | None = None, task_id: str | None = None, tool: str | None = None,
               status: str | None = None, error_type: str | None = None, digest: str = "",
               run_kind: str | None = None, mode: str | None = None) -> int:
        if kind not in TRAJECTORY_KINDS:
            raise ValueError(f"trajectory kind must be one of {', '.join(TRAJECTORY_KINDS)}, not {kind!r}")
        # Redact first, then cap: a redaction marker must never be what pushes the digest over the bound.
        text = self.redact(" ".join(str(digest or "").split()))[:DIGEST_CHARS]
        cur = self.db.execute(
            "INSERT INTO trajectory(ts, kind, run_id, task_id, tool, status, error_type, digest, run_kind, mode) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (now_iso(), kind, run_id, task_id, tool, status, error_type, text, run_kind, mode),
        )
        return int(cur.lastrowid)

    def query(self, *, kind: str | None = None, tool: str | None = None, status: str | None = None,
              error_type: str | None = None, run_id: str | None = None, task_id: str | None = None,
              since: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """Newest first: what a later run (``history_search``) or the human (``GET /trajectory``) asks with."""
        sql, params = "SELECT * FROM trajectory WHERE 1=1", []
        if kind:
            sql += " AND (kind = ? OR kind LIKE ?)"
            params += [kind, f"{kind}.%"]
        if tool:
            sql += " AND tool = ?"
            params.append(tool)
        if status:
            sql += " AND status = ?"
            params.append(status)
        if error_type:
            sql += " AND error_type = ?"
            params.append(error_type)
        if run_id:
            sql += " AND run_id = ?"
            params.append(run_id)
        if task_id:
            sql += " AND task_id = ?"
            params.append(task_id)
        if since:
            sql += " AND ts >= ?"
            params.append(since)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return self.db.query(sql, tuple(params))

    def count(self) -> int:
        return int(self.db.one("SELECT COUNT(*) AS n FROM trajectory")["n"])  # type: ignore[index]

    def prune(self, max_rows: int = MAX_ROWS) -> int:
        """Delete the oldest whole rows past the bound and clear the log. Returns how many went."""
        row = self.db.one("SELECT COUNT(*) AS n, MAX(id) AS hi FROM trajectory")
        n, hi = int(row["n"]), int(row["hi"] or 0)  # type: ignore[index]
        if n <= max_rows:
            return 0
        cut = hi - max_rows
        with self.db.transaction() as conn:
            gone = conn.execute("DELETE FROM trajectory WHERE id <= ?", (cut,)).rowcount
        self.db.checkpoint()
        return int(gone)

    def wipe(self) -> dict[str, Any]:
        """Delete the whole index, so the forget-everything endpoint leaves nothing of it in the database
        files (freed pages are zeroed by ``secure_delete``; the log is checkpointed and truncated)."""
        with self.db.transaction() as conn:
            n = int(conn.execute("SELECT COUNT(*) FROM trajectory").fetchone()[0])
            conn.execute("DELETE FROM trajectory")
        return {"deleted": n, "wal_cleared": self.db.checkpoint()}
