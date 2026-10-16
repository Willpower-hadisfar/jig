"""Keyed long-term memory: author preferences, book-world state and project continuity, told apart.

All against the real SQLite database and the real gate. The flat ``memories`` store is untouched by any
of this (its own tests are tests/test_memory.py); ``facts`` is the keyed complement.
"""

from __future__ import annotations

import secrets
from dataclasses import replace

import pytest

from jig.agent.prompts import CONTEXT_KEY, chosen_facts, shown_facts, turn_scope
from jig.config import load_config
from jig.db import Database
from jig.errors import NotFound
from jig.facts import FACT_CHARS, FactStore
from jig.runtime import Jig
from jig.store import Store

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


# --------------------------------------------------------------------------- #
# Scope isolation on the automatic prompt path: a turn about one book must not be
# handed another book's canon. The store filters correctly; the defect was that the
# turn never asked it to.

EMBER_FACT = "Wren is the protagonist of Ember Island"
SALT_FACT = "Marek is the protagonist of the Salt Road"
SHED_FACT = "Kestrel is the protagonist of the shed rebuild"
AUTHOR_FACT = "The author prefers British English spelling"


def _book_jig(tmp_path, monkeypatch):
    """A real runtime over a real database: composing a turn's prompt is not a model call, so no
    model server is needed (the same arrangement as tests/test_prompt_cache_stability.py)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    key = tmp_path / "vault.key"
    key.write_text(secrets.token_urlsafe(32) + "\n", encoding="utf-8")
    monkeypatch.setenv("JIG_VAULT_BACKEND", "keyfile")
    monkeypatch.setenv("JIG_VAULT_KEY_FILE", str(key))
    config = load_config(data_dir=tmp_path / "data", sandbox_dir=tmp_path / "sandbox")
    return Jig(config)


def _three_scopes(facts: FactStore) -> dict[str, dict]:
    """One distinguishing fact per scope, over three scopes, plus enough of Ember Island's own notes
    that a ``relevant`` turn really takes its matching path rather than fitting everything."""
    rows = {}
    for i in range(1, 36):
        facts.remember(content=f"Ember Island note {i}: tide tables for chapter {i}", kind="world",
                       scope="book:ember-island", subject=f"notes-{i}")
    rows["author"] = facts.remember(content=AUTHOR_FACT, kind="preference", subject="style")
    rows["salt"] = facts.remember(content=SALT_FACT, kind="world", scope="book:salt-road", subject="protagonist")
    rows["shed"] = facts.remember(content=SHED_FACT, kind="continuity", scope="project:shed-rebuild",
                                  subject="cast")
    rows["ember"] = facts.remember(content=EMBER_FACT, kind="world", scope="book:ember-island",
                                   subject="protagonist")
    return rows


def test_a_scope_said_once_is_kept_for_the_conversation(tmp_path):
    """The pin belongs to the conversation, not the message: said once, kept through every later turn of it,
    changed when someone says otherwise, and gone with the conversation."""
    store = Store(Database(tmp_path / "jig.db"))
    ask = [{"role": "user", "content": "Who is the protagonist?"}]
    store.save_session("s1", ask, scope="book:ember-island")
    assert store.session_scope("s1") == "book:ember-island"

    store.save_session("s1", ask)  # a later turn says nothing about scope
    assert store.session_scope("s1") == "book:ember-island", "the conversation keeps what it was pinned to"

    store.save_session("s1", ask, scope="all")  # widened on purpose
    assert store.session_scope("s1") == "all"
    store.save_session("s1", ask, scope="auto")  # back to Jig choosing per turn
    assert store.session_scope("s1") == "auto"
    store.save_session("s1", ask, scope="")
    assert store.session_scope("s1") == ""
    assert store.session_scope("a-conversation-that-does-not-exist") == ""


def test_which_work_a_turn_is_about_is_decided_in_one_order(tmp_path):
    """Message first, then the conversation, then the message read by Jig, then everything — and only a
    message pointing at exactly one work narrows the turn."""
    facts = _facts(tmp_path)
    _three_scopes(facts)
    about_ember = "How does Wren reach the lighthouse?"
    about_two = "Who is the protagonist in this book?"  # Ember Island and the Salt Road both answer

    assert turn_scope(named="book:salt-road", stored="book:ember-island", message=about_ember, facts=facts) == \
        ("book:salt-road", "book:salt-road", "named"), "what this message says outranks what it said"
    assert turn_scope(named="", stored="book:ember-island", message=about_two, facts=facts) == \
        ("book:ember-island", "book:ember-island", "conversation"), "a scope said once is kept"
    assert turn_scope(named="", stored="", message=about_ember, facts=facts) == \
        ("", "book:ember-island", "chosen"), "one work in the message decides it"
    assert turn_scope(named="", stored="", message=about_two, facts=facts) == ("", "", "everything"), \
        "two works in the message decide nothing: narrowing here would hide one of them"
    assert turn_scope(named="all", stored="book:ember-island", message=about_two, facts=facts) == ("all", "", "all")
    assert turn_scope(named="auto", stored="book:ember-island", message=about_ember, facts=facts) == \
        ("auto", "", "auto"), "'auto' widens a pinned conversation back to choosing per turn"

    # And the answer is what the turn is actually given: the chosen scope's facts and the author's, no others.
    _keep, filter_for, _how = turn_scope(named="", stored="", message=about_two, facts=facts)
    turn = chosen_facts(facts, about_two, how="relevant", scope=filter_for or None)
    assert SALT_FACT in turn and EMBER_FACT in turn, "an undecided turn spans the shelf, as it always did"


async def test_a_turn_for_one_book_is_never_given_another_book_s_facts(tmp_path, monkeypatch):
    """Both ``memory_prompt`` settings, at the level a turn actually receives."""
    for how in ("recent", "relevant"):
        jig = _book_jig(tmp_path / how, monkeypatch)
        try:
            jig.config = replace(jig.config, runtime=replace(jig.config.runtime, memory_prompt=how))
            rows = _three_scopes(jig.facts)

            turn = jig._turn_context("Who is the protagonist in this book?", scope="book:ember-island")

            assert EMBER_FACT in turn, f"{how}: the book's own canon never reached the turn"
            assert AUTHOR_FACT in turn, f"{how}: the author's facts carry across books and belong here"
            assert SALT_FACT not in turn, f"{how}: another book's canon reached a turn about Ember Island"
            assert SHED_FACT not in turn, f"{how}: a project's facts reached a turn about a book"
            # And the bleed is visible in the transcript: every line names the scope it came from.
            assert f"(fact #{rows['ember']['id']} in book:ember-island)" in turn
            assert f"(fact #{rows['author']['id']} in author)" in turn
        finally:
            await jig._close_resources()


async def test_a_turn_that_names_no_scope_still_sees_every_scope(tmp_path, monkeypatch):
    """Asking no scope changes nothing: an author-level (or unset) turn keeps the whole shelf."""
    jig = _book_jig(tmp_path, monkeypatch)
    try:
        _three_scopes(jig.facts)
        turn = jig._turn_context("Who is the protagonist in this book?")
        assert EMBER_FACT in turn and AUTHOR_FACT in turn
        assert SALT_FACT in turn and SHED_FACT in turn, "an unset scope still spans every scope"
    finally:
        await jig._close_resources()


async def test_a_turn_refuses_a_scope_jig_could_never_have_filed(tmp_path, monkeypatch):
    """A scope is checked before the model is asked: never quietly treated as "no facts", nor widened to
    everything."""
    jig = _book_jig(tmp_path / "bad", monkeypatch)
    try:
        with pytest.raises(ValueError, match="scope must be 'author', 'book:<slug>' or 'project:<slug>'"):
            async for _item in jig.chat("Who is the protagonist?", scope="book:The Ember Island"):
                pass
    finally:
        await jig._close_resources()


def test_a_scope_labelled_line_is_still_known_as_already_shown(tmp_path):
    """The label must not cost the prompt cache: a second turn of the same conversation adds nothing
    the conversation has not already seen, and still reports a fact that has gone."""
    facts = _facts(tmp_path)
    rows = _three_scopes(facts)
    ask = "Who is the protagonist in this book?"
    first = chosen_facts(facts, ask, scope="book:ember-island")
    assert f"(fact #{rows['ember']['id']} in book:ember-island)" in first

    shown = shown_facts([{"role": "user", CONTEXT_KEY: first}])
    assert shown[rows["ember"]["id"]] == EMBER_FACT and shown[rows["author"]["id"]] == AUTHOR_FACT
    assert all(line in (EMBER_FACT, AUTHOR_FACT) or line.startswith("Ember Island note")
               for line in shown.values()), "a shown block of an Ember Island turn holds another scope's fact"
    assert chosen_facts(facts, ask, scope="book:ember-island", shown=shown) == ""

    facts.remember(content="Wren Ashgrove is the protagonist of Ember Island", kind="world",
                   scope="book:ember-island", subject="protagonist", supersedes=rows["ember"]["id"])
    assert "Wren Ashgrove" in chosen_facts(facts, ask, scope="book:ember-island", shown=shown)


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
