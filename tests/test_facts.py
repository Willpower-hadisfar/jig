"""Keyed long-term memory: author preferences, book-world state and project continuity, told apart.

All against the real SQLite database and the real gate. The flat ``memories`` store is untouched by any
of this (its own tests are tests/test_memory.py); ``facts`` is the keyed complement.
"""

from __future__ import annotations

import pytest

from jig.db import Database
from jig.errors import NotFound
from jig.facts import FACT_CHARS, FactStore

from .memory_bank_helpers import BankStack


def _facts(tmp_path) -> FactStore:
    return FactStore(Database(tmp_path / "jig.db"))


def test_a_fact_remembers_who_it_is_about(tmp_path):
    facts = _facts(tmp_path)
    preference = facts.remember(content="The author prefers British English spelling", kind="preference",
                                subject="style")
    world = facts.remember(content="Wren is the protagonist", kind="world", scope="book:ember-island",
                           subject="protagonist")
    continuity = facts.remember(content="Act two is still unwritten", kind="continuity",
                                scope="book:ember-island", subject="act-two")

    assert (preference["scope"], preference["kind"]) == ("author", "preference")
    assert world["scope"] == "book:ember-island" and world["subject"] == "protagonist"
    assert [f["id"] for f in facts.list(scope="book:ember-island")] == [continuity["id"], world["id"]]
    assert [f["id"] for f in facts.list(scope="author")] == [preference["id"]]
    assert facts.get(world["id"])["content"] == "Wren is the protagonist"
    # Search inside a book scope finds the book's facts and the author's, but not another book's.
    facts.remember(content="The lighthouse burns green", kind="world", scope="book:other-novel")
    assert {f["content"] for f in facts.search("protagonist", scope="book:ember-island")} == {"Wren is the protagonist"}
    assert {f["content"] for f in facts.search("English", scope="book:ember-island")} == \
        {"The author prefers British English spelling"}
    assert facts.search("protagonist", scope="book:other-novel") == []


def test_exact_repeats_are_one_fact_a_correction_supersedes(tmp_path):
    facts = _facts(tmp_path)
    first = facts.remember(content="Wren is the protagonist", kind="world", scope="book:ember-island")
    again = facts.remember(content="Wren is the protagonist", kind="world", scope="book:ember-island")
    assert again["id"] == first["id"] and facts.count() == 1

    corrected = facts.remember(content="Wren Ashgrove is the protagonist", kind="world",
                               scope="book:ember-island", supersedes=first["id"])
    assert corrected["supersedes"] == first["id"]
    assert facts.get(first["id"])["status"] == "superseded"
    assert facts.count() == 2, "the superseded row is kept as history, never deleted"
    assert [f["id"] for f in facts.list()] == [corrected["id"]], "only the active one is listed"
    assert [f["id"] for f in facts.list(status="superseded")] == [first["id"]]


def test_a_fact_the_user_wrote_is_never_superseded_by_jig(tmp_path):
    facts = _facts(tmp_path)
    theirs = facts.remember(content="The author's name is Robyn", source="user")
    with pytest.raises(ValueError, match="written by the user"):
        facts.remember(content="The author's name is Robin", supersedes=theirs["id"])
    assert facts.get(theirs["id"])["status"] == "active"


def test_a_fact_is_one_checked_sentence(tmp_path):
    facts = _facts(tmp_path)
    with pytest.raises(ValueError, match="one sentence"):
        facts.remember(content="x" * (FACT_CHARS + 1))
    with pytest.raises(ValueError, match="kind must be one of"):
        facts.remember(content="Wren is the protagonist", kind="gossip")
    with pytest.raises(ValueError, match="scope"):
        facts.remember(content="Wren is the protagonist", scope="book:The Ember Island")
    with pytest.raises(ValueError, match="must not be empty"):
        facts.remember(content="   ")
    with pytest.raises(NotFound):
        facts.get(999)


def test_facts_survive_a_reopen(tmp_path):
    path = tmp_path / "jig.db"
    facts = FactStore(Database(path))
    kept = facts.remember(content="Wren is the protagonist", kind="world", scope="book:ember-island")
    facts.db.close()

    facts = FactStore(Database(path))
    assert facts.get(kept["id"])["content"] == "Wren is the protagonist"
    assert facts.search("protagonist")[0]["id"] == kept["id"]
    facts.db.close()


async def test_the_agent_writes_and_reads_facts_through_the_gate(tmp_path):
    stack = BankStack(tmp_path)
    added = await stack.call("fact_remember", {"content": "The author hates the word moist",
                                               "kind": "preference", "subject": "language"})
    assert added.ok, added.error
    assert added.result["source"] == "run:r1" and added.result["status"] == "active"

    # A repeat changes nothing; a correction supersedes, all without leaving research mode.
    again = await stack.call("fact_remember", {"content": "The author hates the word moist",
                                               "kind": "preference", "subject": "language"})
    assert again.ok and again.result["id"] == added.result["id"]
    fixed = await stack.call("fact_remember", {"content": "The author avoids the word moist",
                                               "kind": "preference", "subject": "language",
                                               "supersedes": added.result["id"]})
    assert fixed.ok and fixed.result["supersedes"] == added.result["id"]

    found = await stack.call("fact_search", {"query": "moist"})
    assert [r["id"] for r in found.result["results"]] == [fixed.result["id"]]

    bad = await stack.call("fact_remember", {"content": "fine", "kind": "gossip"})
    assert not bad.ok and bad.error_type == "ToolArgumentError"

    # The audit log keeps ids and sizes, never the sentence itself.
    blob = stack.audit_blob()
    assert "moist" not in blob
    stack.close()
