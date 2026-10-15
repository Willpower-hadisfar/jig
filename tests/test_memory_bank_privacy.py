"""Privacy of the new stores: the forget-everything wipe reaches them, the schema upgrades in place, and
nothing of their content is left anywhere a later reader could find it.

Byte-level checks on the real database file, in the manner of tests/test_memory_wipe.py: what the author
wrote must not survive the wipe in the database file or its write-ahead log.
"""

from __future__ import annotations

from jig.db import SCHEMA_VERSION, Database
from jig.facts import FactStore
from jig.memory import MemoryStore
from jig.skills import SkillBank
from jig.store import Store
from jig.trajectory import TrajectoryLog

SECRET = "zanzibarquill"


def _files(db: Database) -> bytes:
    wal = db.path.with_name(db.path.name + "-wal")
    return db.path.read_bytes() + (wal.read_bytes() if wal.exists() else b"")


def _seed(tmp_path):
    db = Database(tmp_path / "jig.db")
    facts, skills, trajectory = FactStore(db), SkillBank(db), TrajectoryLog(db)
    first = facts.remember(content=f"The vault door opens with {SECRET}", kind="world", scope="book:ember-island")
    # a superseded row referencing a live one: the wipe must still take the whole table
    facts.remember(content="The vault door opens with a brass key", kind="world", scope="book:ember-island",
                   supersedes=first["id"])
    skills.propose(name="vault-lines", title="Opening lines about the vault",
                   summary=f"Draft lines that mention {SECRET}.", body="1. read notes\n2. draft\n3. shelve")
    trajectory.record("step", tool="fact_remember", status="ok", digest="fact_remember ok")
    return db, facts, skills, trajectory


def test_forget_everything_leaves_nothing_of_the_new_stores(tmp_path):
    db, facts, skills, trajectory = _seed(tmp_path)
    memory, notes = MemoryStore(db), Store(db)
    memory.add(f"{SECRET} is also in a plain memory")
    notes.add_note(title="vault", body=f"the code is {SECRET}", task_id=None)
    db.checkpoint()
    assert SECRET.encode() in _files(db), "the test must see the text before the wipe"

    # The five wipes POST /forget runs, in the order it runs them (jobs and conversations have nothing here).
    memory.wipe()
    facts.wipe()
    skills.wipe()
    notes.wipe_notes()
    trajectory.wipe()
    after = _files(db)
    assert SECRET.encode() not in after, f"{SECRET!r} survived the wipe in the database files"
    assert facts.count() == 0 and skills.count() == 0 and trajectory.count() == 0
    assert db.query("SELECT rowid FROM facts_fts WHERE facts_fts MATCH 'zanzibarquill'") == []
    assert db.query("SELECT rowid FROM skills_fts WHERE skills_fts MATCH 'zanzibarquill'") == []

    # And the stores work on afterwards, as they do after any memory wipe.
    assert facts.remember(content="Wren is the protagonist")["id"]
    assert skills.propose(name="afterword", title="T", summary="S.", body="steps")["status"] == "candidate"


def test_an_installed_database_upgrades_in_place(tmp_path):
    """A live database at the old schema keeps everything it holds and gains the new tables at open; the
    recorded schema version moves with it."""
    db = Database(tmp_path / "jig.db")
    db.execute("INSERT INTO memories(content, kind, tags, source, created_at, updated_at) "
               "VALUES (?, 'fact', '', 'user', ?, ?)", ("Robyn prefers tea", "2026-01-01T00:00:00+00:00",
                                                         "2026-01-01T00:00:00+00:00"))
    # Back the file to what an older Jig left: no new tables, version recorded as 1.
    for table in ("facts_fts", "facts", "skills_fts", "skills", "trajectory"):
        db.execute(f"DROP TABLE IF EXISTS {table}")
    db.execute("UPDATE meta SET value = '1' WHERE key = 'schema_version'")
    db.close()

    db = Database(tmp_path / "jig.db")
    assert db.one("SELECT value FROM meta WHERE key = 'schema_version'")["value"] == str(SCHEMA_VERSION)
    tables = {r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"facts", "facts_fts", "skills", "skills_fts", "trajectory"} <= tables
    assert db.one("SELECT content FROM memories")["content"] == "Robyn prefers tea", "an upgrade loses nothing"
    FactStore(db).remember(content="Wren is the protagonist")  # the new tables are usable
    db.close()
