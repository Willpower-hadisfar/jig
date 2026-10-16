"""The prompt-cache invariant, tested: memory and skill content may only ever arrive at the END.

Some model servers (llama.cpp among them) cannot reuse a cached prompt that changed anywhere before its
end, so Jig's system prompt is byte-identical for every turn of a mode and everything that varies goes
in the ``<jig-context>`` block after the newest user message. A memory or skill layer that splices itself
into the system prompt or an earlier message would silently cost every cached token on every turn — on a
local-model deployment, the difference between usable and unusable — and would never show an error. This
test is the regression net: it fails if injected content changes the system prompt bytes or the bytes of
any earlier message, as seen by the model.

It drives the real runtime's real prompt path (``_system_prompt`` / ``_turn_context``); no model server
is contacted, because composing a prompt is not a model call.
"""

from __future__ import annotations

import json
import secrets


from jig.agent.prompts import (CONTEXT_KEY, agent_system_prompt, for_model, shown_facts, shown_memories,
                               shown_skills)
from jig.config import load_config
from jig.constants import Mode
from jig.runtime import Jig

FACT_A = "The protagonist is called Wrenwoodquill"
FACT_B = "The lighthouse burns green in the final chapter"
SKILL_SUMMARY = "Build the Wrenwoodquill consistency report."


async def _runtime(tmp_path, monkeypatch) -> Jig:
    key = tmp_path / "vault.key"
    key.write_text(secrets.token_urlsafe(32) + "\n", encoding="ascii")
    monkeypatch.setenv("JIG_VAULT_BACKEND", "keyfile")
    monkeypatch.setenv("JIG_VAULT_KEY_FILE", str(key))
    config = load_config(data_dir=tmp_path / "data", sandbox_dir=tmp_path / "sandbox")
    return Jig(config)


def _as_sent(history: list[dict]) -> list[str]:
    """The conversation as the model server receives it: Jig's additions put back after each message."""
    return [json.dumps(for_model(m), sort_keys=True) for m in history]


async def test_memory_and_skill_content_never_changes_the_prompt_prefix(tmp_path, monkeypatch):
    jig = await _runtime(tmp_path, monkeypatch)
    try:
        system = jig._system_prompt(Mode.RESEARCH)
        ask1 = "Outline act two."
        turn1 = jig._turn_context(ask1)
        history = [{"role": "system", "content": system},
                   {"role": "user", "content": ask1, CONTEXT_KEY: turn1}]
        sent1 = _as_sent(history)

        # Between turns the world changes: the agent records keyed facts, and the user promotes a skill.
        jig.facts.remember(content=FACT_A, kind="world", scope="book:ember-island")
        jig.facts.remember(content=FACT_B, kind="world", scope="book:ember-island")
        skill = jig.skills.propose(name="consistency-report", title="Chapter consistency report",
                                   summary=SKILL_SUMMARY, body="1. list files\n2. compare names")
        jig.skills.set_status(skill["id"], "active")

        ask2 = "Now check the names."
        turn2 = jig._turn_context(ask2, shown_memories(history), shown_facts(history), shown_skills(history))
        history2 = [*history, {"role": "assistant", "content": "Here is the outline."},
                    {"role": "user", "content": ask2, CONTEXT_KEY: turn2}]
        sent2 = _as_sent(history2)

        # Everything that was sent before is sent again, byte for byte: the cache prefix still matches.
        assert sent2[:len(sent1)] == sent1
        # The system prompt is what it was, byte for byte, and holds none of the new content.
        assert jig._system_prompt(Mode.RESEARCH) == system
        assert agent_system_prompt(Mode.RESEARCH, jig.config.runtime.timezone,
                                   can_run_code=jig.container is not None) == system, \
            "the system prompt is static per mode; nothing memory- or skill-shaped may enter it"
        assert "Wrenwoodquill" not in system
        # And the new content reaches the model exactly once: in the newest context block.
        assert "Wrenwoodquill" in turn2 and "consistency-report" in turn2
        assert all("Wrenwoodquill" not in sent for sent in sent2[:-1]), \
            "injected content appeared in an earlier message: the prompt cache would break every turn"

        # A third turn, with nothing changed since, adds neither facts nor skills again: the conversation
        # still only ever grows at the end.
        ask3 = "And the chapter titles?"
        turn3 = jig._turn_context(ask3, shown_memories(history2), shown_facts(history2), shown_skills(history2))
        history3 = [*history2, {"role": "assistant", "content": "Titles: one, two."},
                    {"role": "user", "content": ask3, CONTEXT_KEY: turn3}]
        sent3 = _as_sent(history3)
        assert sent3[:len(sent2)] == sent2
        assert "Wrenwoodquill" not in turn3 and "consistency-report" not in turn3

        # A fact corrected between turns appears as the change, in the new block only.
        old = [f for f in jig.facts.list() if f["content"] == FACT_A][0]
        jig.facts.remember(content="The protagonist is called Wrenwoodquill Ashgrove", kind="world",
                           scope="book:ember-island", supersedes=old["id"])
        turn4 = jig._turn_context("Fine. Names?", shown_memories(history3), shown_facts(history3),
                                  shown_skills(history3))
        assert "Ashgrove" in turn4 and "No longer current" in turn4
        assert _as_sent([*history3, {"role": "user", "content": "Fine. Names?", CONTEXT_KEY: turn4}])[:len(sent3)] == sent3
    finally:
        await jig._close_resources()


async def test_the_memory_sections_of_the_flat_store_still_arrive_only_at_the_end(tmp_path, monkeypatch):
    """The pre-existing guarantee (tests/test_memory.py) holds next to the new sections: memories, facts
    and the skill index share the end-of-conversation channel and nothing else."""
    jig = await _runtime(tmp_path, monkeypatch)
    try:
        jig.memory.add("Robyn prefers tea to coffee")
        jig.facts.remember(content="Robyn writes literary fantasy", kind="preference")
        context = jig._turn_context("What shall I drink while I write?")
        assert "<jig-context>" in context and context.rstrip().endswith("</jig-context>")
        assert "Robyn prefers tea" in context and "literary fantasy" in context
        for mode in Mode:
            assert "Robyn" not in jig._system_prompt(mode)
        # A plain user message still gets its context appended after it, never woven into earlier turns.
        assert context.startswith("\n\n<jig-context>")
    finally:
        await jig._close_resources()
