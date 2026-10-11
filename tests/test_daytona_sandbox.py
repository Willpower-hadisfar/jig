"""Daytona sandbox backend, tested without a live Daytona instance.

The container sandbox tests (``test_container_sandbox.py``) drive real Docker; these drive a
stubbed self-hosted Daytona control plane over ``httpx.MockTransport``, so they run anywhere
pytest runs. They cover what the backend promises: configuration validation (no hardcoded
host), request construction, dispatch selection (choosing Daytona means Docker is never
consulted), the result shape the exec tools and their truncation handling rely on, the
10-minute execution ceiling, session cleanup, and the policy differences written down in
``jig/sandbox_daytona/policy.py``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import tarfile
import io

import httpx
import pytest

from jig.config import SandboxConfig, load_config
from jig.errors import ConfigError, SandboxUnavailable, ToolError
from jig.runtime import Jig
from jig.sandbox_daytona import DaytonaSandbox
from jig.sandbox_daytona import backend as daytona_backend
from jig.sandbox_daytona import policy
from jig.sandbox_daytona.backend import CAPTURE_BYTES, build_command
from jig.sandbox_daytona.client import DaytonaClient

BASE = "http://daytona.test:3000/api"


@pytest.fixture(autouse=True)
def daytona_env(monkeypatch, tmp_path):
    for var in ("DAYTONA_API_URL", "DAYTONA_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # These tests never touch an OS keyring: the vault rides on a throwaway key file.
    key_file = tmp_path / "vault.key"
    key_file.write_bytes(base64.b64encode(bytes(range(32))))
    monkeypatch.setenv("JIG_VAULT_BACKEND", "keyfile")
    monkeypatch.setenv("JIG_VAULT_KEY_FILE", str(key_file))
    # State polling waits 300s at 2s per loop in production; tests settle instantly.
    monkeypatch.setattr(daytona_backend, "START_POLL_S", 0.01)
    monkeypatch.setattr(daytona_backend, "START_WAIT_S", 0.05)


def daytona_config(tmp_path, **sandbox_settings):
    settings = {"sandbox": {"backend": "daytona", "daytona_api_base_url": BASE, "daytona_api_key": "k-1",
                            **sandbox_settings}}
    return load_config(data_dir=tmp_path / "data", sandbox_dir=tmp_path / "sandbox", settings=settings)


def default_execute(command: str) -> dict:
    """What the stub's sh answers when a test has no opinion: the sync plumbing succeeds and
    the start-up self-check prints its marker as an unprivileged user."""
    if "echo jig-daytona-ok" in command:
        return {"exitCode": 0, "stdout": "jig-daytona-ok\n10001\n", "stderr": ""}
    return {"exitCode": 0, "stdout": "", "stderr": ""}


class DaytonaStub:
    """A stubbed self-hosted Daytona control plane under the base path /api: records every
    request and answers with canned results, so no instance (or Docker) is needed."""

    def __init__(self, *, healthy=True, states=(), fixed_state=None, execute=None, download=b""):
        self.healthy = healthy
        self.execute = execute or default_execute
        self.download = download
        self.requests: list[tuple[str, str]] = []
        self.auth: list[str | None] = []
        self.create_body: dict = {}
        self.uploads: dict[str, bytes] = {}
        self.executions: list[dict] = []
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.deleted: list[str] = []
        self.processes: list[list[str]] = []
        self.stdin: list[str] = []
        self.killed: list = []
        self.events: list[str] = []
        self.states = list(states)  # served one per state poll before the sandbox settles on "started"
        self.fixed_state = fixed_state
        self.sandbox: dict | None = None

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))

    def push_event(self, text: str) -> None:
        self.events.append(text)

    def handle(self, request: httpx.Request) -> httpx.Response:
        parts = httpx.URL(str(request.url))
        assert str(request.url).startswith(BASE), f"the client called {request.url}, outside the configured base"
        tail = parts.path[len("/api"):]
        body = request.content or b""
        self.requests.append((request.method, tail))
        self.auth.append(request.headers.get("authorization"))
        seg = [p for p in tail.split("/") if p]
        if seg == ["health"]:
            if not self.healthy:
                return httpx.Response(503, json={"message": "builder offline"})
            return httpx.Response(200, json={"status": "healthy"})
        if seg == ["sandbox"] and request.method == "GET":
            return httpx.Response(200, json=[] if self.sandbox is None else [self.sandbox])
        if seg == ["sandbox"] and request.method == "POST":
            payload = json.loads(body)
            self.create_body = payload
            self.sandbox = {"id": "sbx-1", "name": payload["name"], "state": "created"}
            return httpx.Response(201, json=self.sandbox)
        if seg[:2] == ["toolbox", "sandbox"]:
            return self._toolbox(seg, parts.query, body)
        if seg[:2] == ["sandbox", "sbx-1"]:
            if seg[-1] == "start":
                self.started.append("sbx-1")
                self.sandbox["state"] = "started"
                return httpx.Response(200, json=self.sandbox)
            if seg[-1] == "stop":
                self.stopped.append("sbx-1")
                self.sandbox["state"] = "stopped"
                return httpx.Response(200, json={})
            if request.method == "DELETE":
                self.deleted.append("sbx-1")
                self.sandbox = None
                return httpx.Response(204)
            if self.fixed_state:
                state = self.fixed_state
            else:
                state = self.states.pop(0) if self.states else "started"
            return httpx.Response(200, json={**self.sandbox, "state": state})
        return httpx.Response(404, json={"error": f"stub: no route for {tail}"})

    def _toolbox(self, seg: list[str], query: str, body: bytes) -> httpx.Response:
        if seg[-1] == "execute":
            payload = json.loads(body)
            self.executions.append(payload)
            result = self.execute(payload["input"])
            return httpx.Response(200, json={"type": "execute_result", "result": result})
        if seg[-1] == "upload":
            dest = httpx.QueryParams(query).get("dest", "?")
            self.uploads[dest] = body  # the raw multipart body: tests check their content travelled inside it
            return httpx.Response(200, json={"success": True})
        if seg[-1] == "download":
            return httpx.Response(200, content=self.download)
        if seg[-1] == "start":  # process/start
            payload = json.loads(body)
            self.processes.append([payload["command"], *payload.get("args", [])])
            return httpx.Response(200, json={"process": {"sessionId": "s-1", "pid": 7}})
        if seg[-1] == "send":  # process/stdin/send
            self.stdin.append(json.loads(body)["input"])
            return httpx.Response(200, json={"success": True})
        if seg[-1] == "events":
            index = int(httpx.QueryParams(query).get("index", "0"))
            chunks = self.events[index:]
            return httpx.Response(200, json={"events": [{"type": "stdout", "text": t} for t in chunks],
                                             "index": len(self.events)})
        if seg[-1] == "kill":
            self.killed.append(json.loads(body))
            return httpx.Response(200, json={"success": True})
        return httpx.Response(404, json={"error": f"stub: no toolbox route for {seg}"})


def make_sandbox(tmp_path, stub) -> DaytonaSandbox:
    cfg = daytona_config(tmp_path)
    workspace = cfg.sandbox_dir / "default"
    workspace.mkdir(parents=True, exist_ok=True)
    return DaytonaSandbox(cfg.sandbox, workspace, "default", client=DaytonaClient(BASE, "k-1",
                                                                                  client=stub.client()))


# ---------------------------------------------------------------------------------------------
# Configuration: the connection is configurable and validated, never a hardcoded host.

def test_daytona_needs_a_base_url(tmp_path, monkeypatch):
    monkeypatch.delenv("DAYTONA_API_URL", raising=False)
    with pytest.raises(ConfigError, match="daytona_api_base_url"):
        load_config(data_dir=tmp_path / "data", sandbox_dir=tmp_path / "sandbox",
                    settings={"sandbox": {"backend": "daytona", "daytona_api_key": "k"}})


def test_daytona_needs_an_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="daytona_api_key"):
        load_config(data_dir=tmp_path / "data", sandbox_dir=tmp_path / "sandbox",
                    settings={"sandbox": {"backend": "daytona", "daytona_api_base_url": BASE}})


def test_daytona_base_url_must_be_an_http_url(tmp_path):
    with pytest.raises(ConfigError, match="http"):
        daytona_config(tmp_path, daytona_api_base_url="daytona.lan:3000")


def test_daytona_env_vars_supply_the_connection(tmp_path, monkeypatch):
    monkeypatch.setenv("DAYTONA_API_URL", "https://daytona.lan:443")
    monkeypatch.setenv("DAYTONA_API_KEY", "env-key")
    cfg = load_config(data_dir=tmp_path / "data", sandbox_dir=tmp_path / "sandbox",
                      sandbox_backend="daytona")
    assert cfg.sandbox.daytona_api_base_url == "https://daytona.lan:443"
    assert cfg.sandbox.daytona_api_key == "env-key"


def test_other_backends_need_no_daytona_settings(tmp_path):
    cfg = load_config(data_dir=tmp_path / "data", sandbox_dir=tmp_path / "sandbox")
    assert cfg.sandbox.backend == "directory"
    assert cfg.sandbox.daytona_api_base_url == ""


# ---------------------------------------------------------------------------------------------
# Client: request construction against the stubbed endpoint.

async def test_client_sends_bearer_auth_under_the_configured_base():
    stub = DaytonaStub()
    client = DaytonaClient(BASE, "k-1", client=stub.client())
    out = await client.execute("sbx-1", "echo hi", 30)
    assert out["exit_code"] == 0
    assert stub.requests[-1] == ("POST", "/toolbox/sandbox/sbx-1/process/execute")
    assert stub.auth[-1] == "Bearer k-1"
    payload = stub.executions[-1]
    assert payload["input"] == "echo hi" and payload["shell"] == "sh" and payload["timeout"] == 30
    await client.close()


async def test_client_reports_daytona_http_errors_loudly():
    def boom(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "runner out of space"})

    client = DaytonaClient(BASE, "k-1", client=httpx.AsyncClient(transport=httpx.MockTransport(boom)))
    with pytest.raises(SandboxUnavailable, match="HTTP 500.*runner out of space"):
        await client.health()
    await client.close()


async def test_client_reports_unreachability_clearly():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = DaytonaClient(BASE, "k-1", client=httpx.AsyncClient(transport=httpx.MockTransport(refuse)))
    with pytest.raises(SandboxUnavailable, match="is not reachable.*self-hosted instance"):
        await client.health()
    await client.close()


def test_client_refuses_a_missing_or_malformed_connection():
    bare = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    with pytest.raises(SandboxUnavailable, match="daytona_api_base_url"):
        DaytonaClient("", "k", client=bare)
    with pytest.raises(SandboxUnavailable, match="http"):
        DaytonaClient("daytona.lan:3000", "k", client=bare)
    with pytest.raises(SandboxUnavailable, match="daytona_api_key"):
        DaytonaClient(BASE, "", client=bare)


def test_stdin_travels_base64_encoded():
    command = build_command(["python3", "-"], b"print('hi')\n")
    assert command.startswith("printf %s ") and "| base64 -d |" in command and "python3 -" in command
    assert build_command(["sh", "-c", "echo hi"], None) == "sh -c 'echo hi'"


# ---------------------------------------------------------------------------------------------
# Backend contract: what the exec and browser tools rely on.

async def test_start_creates_and_starts_the_sandbox(tmp_path):
    stub = DaytonaStub(states=["creating", "starting"])
    sandbox = make_sandbox(tmp_path, stub)
    caps = await sandbox.start()
    assert stub.create_body["name"] == sandbox.name
    assert stub.create_body["image"] == sandbox.image
    assert stub.started == ["sbx-1"]  # created, started, then polled until it settled
    assert caps["backend"] == "daytona"
    assert caps["base_url"] == BASE  # the logs say WHICH instance answered, never a guess
    assert caps["egress_gated"] is False  # the Daytona API cannot gate a sandbox's network
    assert sandbox.started
    await sandbox.stop()


async def test_start_fails_loudly_when_the_instance_is_unhealthy(tmp_path):
    stub = DaytonaStub(healthy=False)
    sandbox = make_sandbox(tmp_path, stub)
    with pytest.raises(SandboxUnavailable, match="builder offline"):
        await sandbox.start()
    assert stub.sandbox is None, "nothing is created against an instance that is not healthy"
    await sandbox.stop()  # idempotent cleanup


async def test_start_fails_when_the_sandbox_never_settles(tmp_path):
    stub = DaytonaStub(fixed_state="building")
    sandbox = make_sandbox(tmp_path, stub)
    with pytest.raises(SandboxUnavailable, match="did not reach state 'started'"):
        await sandbox.start()
    await sandbox.stop()


async def test_exec_result_matches_the_docker_backend_shape(tmp_path):
    stub = DaytonaStub(execute=lambda cmd: {"exitCode": 3, "stdout": "out", "stderr": "bad"}
                       if "sh -c false" in cmd else default_execute(cmd))
    sandbox = make_sandbox(tmp_path, stub)
    await sandbox.start()
    result = await sandbox.exec(["sh", "-c", "false"], timeout_s=30)
    assert set(result) == {"exit_code", "timed_out", "stdout", "stderr", "stdout_dropped", "stderr_dropped"}
    assert result["exit_code"] == 3 and result["stdout"] == "out" and result["stderr"] == "bad"
    assert not result["timed_out"]
    # A failing command is a result, not an exception, exactly like docker exec.
    executed = [p["input"] for p in stub.executions]
    assert any("sh -c false" in c for c in executed)


async def test_long_output_is_cut_to_its_tail_with_dropped_reported(tmp_path):
    flood = "x" * (CAPTURE_BYTES + 1000)
    stub = DaytonaStub(execute=lambda cmd: {"exitCode": 0, "stdout": flood, "stderr": ""}
                       if "flood" in cmd else default_execute(cmd))
    sandbox = make_sandbox(tmp_path, stub)
    await sandbox.start()
    result = await sandbox.exec(["sh", "-c", "flood"], timeout_s=30)
    assert len(result["stdout"].encode()) == CAPTURE_BYTES
    assert result["stdout_dropped"] == 1000  # the exec tools turn this into the "not kept" note
    assert result["stderr_dropped"] == 0


async def test_timeouts_beyond_daytonas_ten_minute_ceiling_fail_clearly(tmp_path):
    stub = DaytonaStub()
    sandbox = make_sandbox(tmp_path, stub)
    await sandbox.start()
    before = len(stub.executions)
    with pytest.raises(ToolError, match="10 minutes"):
        await sandbox.exec(["make"], timeout_s=policy.MAX_EXEC_TIMEOUT_S + 1)
    assert len(stub.executions) == before, "the doomed command is never sent"


async def test_daytona_killing_a_run_surfaces_as_timed_out(tmp_path):
    stub = DaytonaStub(execute=lambda cmd: {"exitCode": 137, "stdout": "partial",
                                            "stderr": "killed after 600s"} if "sleep 99999" in cmd
                       else default_execute(cmd))
    sandbox = make_sandbox(tmp_path, stub)
    await sandbox.start()
    result = await sandbox.exec(["sh", "-c", "sleep 99999"], timeout_s=600)
    assert result["timed_out"] and result["exit_code"] == 137


async def test_a_wedged_daytona_fails_with_the_ceiling_explained(tmp_path):
    def hang(command: str) -> dict:
        raise httpx.ReadTimeout("nothing back")

    stub = DaytonaStub(execute=hang)
    sandbox = make_sandbox(tmp_path, stub)
    with pytest.raises(SandboxUnavailable, match="did not answer.*10 minutes"):
        await sandbox.start()


async def test_stop_cleans_the_sandbox_up(tmp_path):
    stub = DaytonaStub()
    sandbox = make_sandbox(tmp_path, stub)
    await sandbox.start()
    await sandbox.stop()
    assert stub.deleted == ["sbx-1"], "the session ending removes the sandbox, like docker rm"
    assert not sandbox.started
    with pytest.raises(SandboxUnavailable, match="not running"):
        await sandbox.exec(["true"])
    await sandbox.stop()  # idempotent


async def test_stop_can_persist_when_the_policy_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "PERSIST_ACROSS_SESSIONS", True)
    stub = DaytonaStub()
    sandbox = make_sandbox(tmp_path, stub)
    await sandbox.start()
    await sandbox.stop()
    assert stub.stopped == ["sbx-1"] and stub.deleted == []


async def test_lease_is_bookkeeping_without_an_egress_gate(tmp_path):
    stub = DaytonaStub()
    sandbox = make_sandbox(tmp_path, stub)
    await sandbox.start()
    before = len(stub.requests)
    async with sandbox.lease(tool="run_command", run_id=None, task_id=None) as lease:
        assert lease.tool == "run_command"
    assert len(stub.requests) == before, "the lease opens and closes nothing on the Daytona API"


# ---------------------------------------------------------------------------------------------
# The workspace travels with the command (a Daytona sandbox cannot bind-mount Jig's folder).

async def test_the_workspace_is_synced_around_each_command(tmp_path):
    stub = DaytonaStub(download=tgz({"from-sandbox.txt": "written in the sandbox\n"}))
    sandbox = make_sandbox(tmp_path, stub)
    await sandbox.start()
    (sandbox.workspace / "input.txt").write_text("host material", encoding="utf-8")
    result = await sandbox.exec(["sh", "-c", "cat input.txt"], timeout_s=30)
    assert result["exit_code"] == 0
    executed = [p["input"] for p in stub.executions]
    assert any("tar -xzf" in c and policy.PUSH_TARBALL in c for c in executed), \
        "the workspace is unpacked into /workspace first"
    assert any("tar -czf" in c for c in executed), "and pulled back after"
    assert policy.PUSH_TARBALL in stub.uploads, "the workspace tarball was uploaded"
    assert (sandbox.workspace / "from-sandbox.txt").read_text(encoding="utf-8") == "written in the sandbox\n"


async def test_a_sync_never_writes_outside_the_workspace(tmp_path):
    stub = DaytonaStub(download=tgz({"../escaped.txt": "nope", "ok.txt": "yes"}))
    sandbox = make_sandbox(tmp_path, stub)
    await sandbox.start()
    await sandbox.exec(["true"], timeout_s=30)
    assert (sandbox.workspace / "ok.txt").exists()
    assert not (tmp_path / "sandbox" / "escaped.txt").exists()


def tgz(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


# ---------------------------------------------------------------------------------------------
# Dispatch: daytona selected means the Daytona path, and Docker is never consulted.

async def test_dispatch_selects_daytona_and_never_touches_docker(tmp_path, monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError(f"the daytona backend consulted Docker: {args}")

    monkeypatch.setenv("JIG_VAULT_BACKEND", "keyfile")
    monkeypatch.setenv("JIG_VAULT_KEY_FILE", str(tmp_path / "vault.key"))
    monkeypatch.setattr("jig.sandbox_container.docker.docker_path", never)
    monkeypatch.setattr("jig.sandbox_container.docker.docker", never)
    monkeypatch.setattr("jig.sandbox_container.egress.EgressProxy.start", never)
    runtime = Jig(daytona_config(tmp_path))
    try:
        assert isinstance(runtime.container, DaytonaSandbox)
        registered = {spec.name for spec in runtime.registry.all()}
        for name in ("run_command", "run_python", "browser_open"):
            assert name in registered, f"{name} must be registered for daytona"
        assert not hasattr(runtime.container, "egress"), "no local egress proxy: Daytona runs it remotely"
    finally:
        await runtime.container.stop()
        runtime.db.close()


async def test_directory_backend_selects_no_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("JIG_VAULT_BACKEND", "keyfile")
    monkeypatch.setenv("JIG_VAULT_KEY_FILE", str(tmp_path / "vault.key"))
    cfg = load_config(data_dir=tmp_path / "data", sandbox_dir=tmp_path / "sandbox")
    runtime = Jig(cfg)
    try:
        assert runtime.container is None
        assert "run_command" not in {spec.name for spec in runtime.registry.all()}
    finally:
        runtime.db.close()


# ---------------------------------------------------------------------------------------------
# Policy: where the Daytona API is less capable than Docker, stated not assumed.

def test_create_payload_carries_nothing_the_api_cannot_apply():
    payload = policy.create_payload("jig-sbx-x", "img:1", SandboxConfig())
    assert payload["name"] == "jig-sbx-x" and payload["image"] == "img:1"
    for docker_only in ("cpus", "memory", "pids_limit", "tmp_size", "shm_size"):
        assert docker_only not in payload
    assert payload["autoStopInterval"] == policy.AUTO_STOP_MINUTES
    assert "cpus" in ", ".join(policy.unsupported_limits(SandboxConfig()))


def test_exec_timeout_policy_passes_short_limits():
    assert policy.exec_timeout(60) == 60
    assert policy.exec_timeout(policy.MAX_EXEC_TIMEOUT_S) == policy.MAX_EXEC_TIMEOUT_S
    assert policy.EGRESS_GATE_ENFORCED is False


# ---------------------------------------------------------------------------------------------
# The browser: the same JSON-line protocol, over Daytona's process API.

async def test_browser_session_round_trip(tmp_path):
    stub = DaytonaStub()
    stub.push_event(json.dumps({"ready": True}) + "\n")
    sandbox = make_sandbox(tmp_path, stub)
    await sandbox.start()
    session = await sandbox.browser()
    assert session.alive
    assert stub.processes == [["python3", "-u", "/opt/jig/browser_server.py"]]

    async def answer_later():
        await asyncio.sleep(0.3)
        stub.push_event(json.dumps({"ok": True, "result": {"title": "Example", "text": "..."}}) + "\n")

    task = asyncio.create_task(answer_later())
    result = await session.call("open", url="https://example.com/")
    assert result["title"] == "Example"
    assert json.loads(stub.stdin[-1]) == {"command": "open", "url": "https://example.com/"}
    await session.close()
    assert stub.killed and not session.alive
    await task


async def test_browser_failures_are_tool_errors(tmp_path):
    stub = DaytonaStub()
    stub.push_event(json.dumps({"ready": True}) + "\n")
    sandbox = make_sandbox(tmp_path, stub)
    await sandbox.start()
    session = await sandbox.browser()

    async def refuse():
        await asyncio.sleep(0.3)
        stub.push_event(json.dumps({"ok": False, "error": "net::ERR_NAME_NOT_RESOLVED"}) + "\n")

    task = asyncio.create_task(refuse())
    with pytest.raises(ToolError, match="browser open failed.*ERR_NAME_NOT_RESOLVED"):
        await session.call("open", url="https://nowhere.invalid/")
    await task
