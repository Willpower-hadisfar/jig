"""A real gate over the real stores for the keyed-memory, skill-bank and trajectory tests.

Nothing is mocked: the database, the gate, the rules and the Vault are the real ones, and the safety
checker is the real Sentinel pointed at an address nothing is listening on (as in tests/test_mcp.py).
The tools exercised through it are READ and PRIVATE_WRITE, which the gate never sends to the Sentinel,
so no model server is needed to run them; model-backed behaviour lives in the ``jig``-fixture tests.
"""

from __future__ import annotations

import json
import secrets

from jig.audit import AuditLog
from jig.config import EndpointConfig, VaultConfig, load_config
from jig.constants import Mode
from jig.db import Database
from jig.events import EventBus
from jig.facts import FactStore
from jig.memory import MemoryStore
from jig.model import ModelClient, ToolCall
from jig.policy.approvals import ApprovalQueue
from jig.policy.gate import CallContext, ToolExecutor
from jig.policy.rules import RuleStore
from jig.policy.sentinel import Sentinel
from jig.skills import SkillBank
from jig.store import Store
from jig.tools.builtin import build_registry
from jig.tools.registry import ToolContext
from jig.trajectory import TrajectoryLog
from jig.vault import Vault

# Nothing listens here: a call that needs Sentinel review fails with SentinelError, which is the point.
DEAD_SENTINEL = "http://127.0.0.1:9/v1"


class BankStack:
    def __init__(self, tmp_path):
        self.config = load_config(data_dir=tmp_path / "data", sandbox_dir=tmp_path / "sandbox")
        self.db = Database(self.config.db_path)
        self.audit = AuditLog(self.db)
        # The keyfile backend with a real key file, so the stack works on any machine (the way
        # tests/test_vault_keyfile.py opens one); the vault is used only to run the gate.
        key = tmp_path / "vault.key"
        key.write_text(secrets.token_urlsafe(32) + "\n", encoding="ascii")
        self.vault = Vault(self.db, VaultConfig(backend="keyfile", key_file=str(key)))
        self.bus = EventBus()
        self.rules = RuleStore(self.db)
        self.approvals = ApprovalQueue(self.db, self.bus, self.audit)
        self.store = Store(self.db)
        self.memory = MemoryStore(self.db)
        self.facts = FactStore(self.db)
        self.skills = SkillBank(self.db)
        self.trajectory = TrajectoryLog(self.db)
        self.registry = build_registry()
        sentinel_model = ModelClient(
            EndpointConfig(base_url=DEAD_SENTINEL, name="nothing-listening", api_key_env=""),
            label="sentinel")
        self.executor = ToolExecutor(registry=self.registry, rules=self.rules, sentinel=Sentinel(sentinel_model),
                                     approvals=self.approvals, vault=self.vault, audit=self.audit, bus=self.bus,
                                     context_factory=self._context, always_redact={}, config=self.config,
                                     trajectory=self.trajectory)

    def _context(self, ctx: CallContext) -> ToolContext:
        return ToolContext(sandbox=None, memory=self.memory, store=self.store, config=self.config, http=None,
                           mode=ctx.mode, run_id=ctx.run_id, task_id=ctx.task_id, facts=self.facts,
                           skills=self.skills, trajectory=self.trajectory)

    async def call(self, tool: str, args: dict, *, run_id: str = "r1", mode: Mode = Mode.RESEARCH,
                   call_id: str | None = None):
        call = ToolCall(id=call_id or f"c-{tool}-{run_id}", name=tool, arguments_raw=json.dumps(args))
        return await self.executor.execute(call, CallContext(run_id, None, mode, "test intent"))

    def audit_blob(self) -> str:
        """Everything the audit log holds, as one string, for the 'content never reaches it' assertions."""
        return json.dumps([{"summary": r["summary"], "data": r["data_json"]} for r in self.audit.query(limit=5000)])

    def close(self) -> None:
        self.db.close()
