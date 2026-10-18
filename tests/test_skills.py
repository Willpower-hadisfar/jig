"""The skill bank: the agent writes candidates, only a human promotes, and only active skills inform a turn.

Real SQLite and the real gate throughout. Promotion is what the ``PATCH /skills/{id}`` route calls
(``SkillBank.set_status``); nothing the agent's own tools can do reaches it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jig.agent.prompts import chosen_skills
from jig.db import Database
from jig.skills import SkillBank

from .memory_bank_helpers import BankStack

BODY = ("1. list the manuscript files\n2. read each with read_file\n3. collect every proper noun per chapter\n"
        "4. report names that slip between chapters")


def _bank(tmp_path) -> SkillBank:
    return SkillBank(Database(tmp_path / "jig.db"))


def test_write_and_read_round_trip(tmp_path):
    bank = _bank(tmp_path)
    skill = bank.propose(name="consistency-report", title="Chapter consistency report",
                         summary="Build the consistency report.", body=BODY)
    again = bank.get_by_name("consistency-report")
    assert again["id"] == skill["id"]
    assert again["body"] == BODY and again["title"] == "Chapter consistency report"
    assert bank.get(skill["id"])["summary"] == "Build the consistency report."
    assert [s["name"] for s in bank.search("consistency", status="candidate")] == ["consistency-report"]
    assert bank.search("consistency") == [], "the default search is over approved skills"


def test_a_candidate_is_only_a_candidate(tmp_path):
    """The promotion rule: nothing that happens on a successful run makes a skill active."""
    bank = _bank(tmp_path)
    skill = bank.propose(name="consistency-report", title="T", summary="S.", body="1. steps")
    assert skill["status"] == "candidate"
    assert bank.active_index() == []
    assert chosen_skills(bank) == "", "a candidate never enters a prompt, in a new session or any other"

    promoted = bank.set_status(skill["id"], "active")
    assert promoted["status"] == "active" and promoted["promoted_at"], "only set_status promotes, and it records when"
    index = chosen_skills(bank)
    assert "consistency-report" in index and "S." in index
    assert "steps" not in index, "the index carries name and summary, never the body"

    retired = bank.set_status(skill["id"], "retired")
    assert retired["status"] == "retired" and bank.active_index() == []
    with pytest.raises(ValueError, match="status must be one of"):
        bank.set_status(skill["id"], "weaponised")


def test_bodies_that_quote_the_work_are_refused(tmp_path):
    bank = _bank(tmp_path)
    prose = ('Say: "the lighthousekeeper\'s daughter walked the long wet strand at dusk, counting the '
             "waves until the hundredth one brought the ship home\"")
    with pytest.raises(ValueError, match="not quoted prose"):
        bank.propose(name="bad", title="T", summary="S.", body=prose)
    with pytest.raises(ValueError, match="not quoted prose"):
        bank.propose(name="bad2", title="T", summary="S.",
                     body='The whole chapter opening: "the lighthousekeeper\'s daughter walked the long wet '
                          'strand at dusk, counting the waves until the hundredth one brought the ship home "'
                          'and then: "she never counted again".')
    with pytest.raises(ValueError, match="name"):
        bank.propose(name="Two Words", title="T", summary="S.", body="steps")
    with pytest.raises(ValueError, match="summary"):
        bank.propose(name="ok-name", title="T", summary="x" * 241, body="steps")
    bank.propose(name="ok-name", title="T", summary="S.", body="steps")
    with pytest.raises(ValueError, match="already exists"):
        bank.propose(name="ok-name", title="T2", summary="S.", body="steps")


async def test_skill_tools_through_the_gate(tmp_path):
    stack = BankStack(tmp_path)
    proposed = await stack.call("skill_propose", {"name": "name-sweep", "title": "Sweep names",
                                                  "summary": "Find inconsistent spellings.",
                                                  "body": "1. grep\n2. list\n3. fix"})
    assert proposed.ok, proposed.error
    assert proposed.result["status"] == "candidate"

    listed = await stack.call("skill_list", {})
    assert listed.ok and listed.result["skills"][0]["status"] == "candidate"

    read = await stack.call("skill_read", {"name": "name-sweep"})
    assert read.ok and read.result["status"] == "candidate" and "not approved" in read.result["warning"]

    # Promotion: exactly what PATCH /skills/{id} does, and only there.
    stack.skills.set_status(proposed.result["skill_id"], "active")
    read = await stack.call("skill_read", {"name": "name-sweep"})
    assert read.ok and read.result["status"] == "active" and "warning" not in read.result
    assert stack.trajectory.query(kind="skill.used", tool="name-sweep"), "a read of an approved skill is recorded"
    stack.close()


def test_the_shipped_painter_definition_resolves_like_any_skill(tmp_path):
    """The painter skill ships as a versioned definition that imports into the same bank, is discovered by
    the same tools, and promotes only by a human action - exactly like a skill Jig wrote itself."""
    import json as _json

    from jig.agent.prompts import chosen_skills

    definition = _json.loads((Path(__file__).parent.parent / "skills"/ "rembrandt-tech-painter.json")
                             .read_text(encoding="utf-8"))
    bank = _bank(tmp_path)
    assert definition["name"] == "rembrandt-tech-painter" and definition["scope"] == "author"
    assert len(definition["references"]) == 3, "the versioned source records the locked reference set"

    skill = bank.import_definition(definition)
    assert skill["status"] == "candidate"
    assert bank.search("rembrandt", status="candidate")[0]["name"] == "rembrandt-tech-painter"
    assert chosen_skills(bank) == "", "a candidate never enters a prompt"
    assert bank.get_by_name("rembrandt-tech-painter")["body"] == definition["body"]

    promoted = bank.set_status(skill["id"], "active")
    assert promoted["status"] == "active" and promoted["promoted_at"]
    assert "rembrandt-tech-painter" in chosen_skills(bank), "active: discoverable the way existing skills are"
