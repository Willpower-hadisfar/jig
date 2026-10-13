"""Daytona sandbox backend: Jig runs its code, shell and browser tools against a self-hosted Daytona instance.

Selected with ``[sandbox] backend = "daytona"``. The API base URL and key come from configuration
or the environment (``DAYTONA_API_URL`` / ``DAYTONA_API_KEY``) — nothing is hardcoded, because the
instance is self-hosted. If the base URL is missing, wrong or unreachable, start-up fails loudly;
there is no fallback to Docker or to the directory sandbox, and selecting this backend never
starts a Docker container.

The sandbox is created on demand under a deterministic name and deleted when the session ends.
A Daytona sandbox cannot bind-mount Jig's folder, so the workspace is synced (tar over the
files API) into ``/workspace`` before each command and back afterwards, within the policy's
sync budget. Where the Daytona API is less capable than Docker — gated egress, resource caps,
persistence, the 10-minute execution ceiling — ``policy.py`` states what that means for Jig.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import itertools
import json
import logging
import os
import shlex
import tarfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from ..config import SandboxConfig
from ..errors import SandboxUnavailable, ToolError
from ..sandbox_container.egress import Lease
from . import policy
from .client import DaytonaClient

log = logging.getLogger("jig.daytona")

# The most of each output stream kept (its end); jig.tools.sandbox_exec decides how much a result shows.
CAPTURE_BYTES = 256 * 1024
BROWSER_START_TIMEOUT_S = 60.0
BROWSER_CALL_TIMEOUT_S = 90.0
BROWSER_POLL_S = 0.25
SYNC_COMMAND_TIMEOUT_S = 60.0
START_WAIT_S = 300.0
START_POLL_S = 2.0


def build_command(argv: list[str], stdin: bytes | None) -> str:
    """The command string for Daytona's execute endpoint. Daytona runs it through sh and has no
    stdin of its own, so piped input travels base64-encoded through the command line."""
    command = shlex.join(argv)
    if stdin:
        command = f"printf %s {shlex.quote(base64.b64encode(stdin).decode())} | base64 -d | {{ {command}; }}"
    return command


class DaytonaSandbox:
    def __init__(self, config: SandboxConfig, workspace: Path, agent_id: str, *, client: DaytonaClient | None = None):
        self.config = config
        self.workspace = Path(workspace)
        # The same deterministic name the container backend derives, one sandbox per agent+workspace.
        key = f"{agent_id}-{hashlib.sha1(str(self.workspace).lower().encode()).hexdigest()[:10]}".lower()
        self.name = f"jig-sbx-{key}"
        self.image = config.daytona_image or config.image
        self.base_url = (config.daytona_api_base_url or "").strip()
        self.sandbox_id: str | None = None
        self.started = False
        self._client = client
        self._owns_client = client is None
        self._browser: DaytonaBrowserSession | None = None
        self._browser_lock = asyncio.Lock()
        self._leases = itertools.count(1)
        self._ws_signature: tuple | None = None
        self._sync_budget_exceeded = False

    # Lifecycle -------------------------------------------------------------
    async def start(self) -> dict[str, Any]:
        if self._client is None:
            # Validates the base URL and key; a missing or malformed one fails here with a clear message.
            self._client = DaytonaClient(self.config.daytona_api_base_url, self.config.daytona_api_key)
            self._owns_client = True
        try:
            await self._client.health()
            existing = await self._client.find_sandbox(self.name)
            if existing:
                self.sandbox_id = str(existing["id"])
                if str(existing.get("state", "")) != "started":
                    await self._client.start_sandbox(self.sandbox_id)
            else:
                created = await self._client.create_sandbox(policy.create_payload(self.name, self.image, self.config))
                self.sandbox_id = str(created["id"])
                if str(created.get("state", "")) != "started":
                    await self._client.start_sandbox(self.sandbox_id)
            await self._client.wait_for_state(self.sandbox_id, deadline_s=START_WAIT_S, poll_s=START_POLL_S)
            self.started = True
            self._ws_signature = None
            await self._self_check()
        except BaseException:
            await self.stop()
            raise
        return {"backend": "daytona", "base_url": self.base_url or self.config.daytona_api_base_url,
                "sandbox": self.name, "image": self.image,
                "egress_gated": policy.EGRESS_GATE_ENFORCED,
                "resource_limits": "the Daytona instance's own (the API cannot apply Jig's Docker caps: "
                                   + ", ".join(policy.unsupported_limits(self.config)) + ")",
                "persist_across_sessions": policy.PERSIST_ACROSS_SESSIONS}

    async def _self_check(self) -> None:
        """Prove commands actually run in the sandbox before Jig advertises code execution."""
        result = await self.exec(["sh", "-c", "echo jig-daytona-ok; id -u"], timeout_s=30)
        if result["exit_code"] != 0 or "jig-daytona-ok" not in result["stdout"]:
            raise SandboxUnavailable(
                f"the Daytona sandbox {self.name} starts but cannot run commands "
                f"(exit {result['exit_code']}, stderr {result['stderr'][-300:]!r})")
        lines = result["stdout"].split()
        if len(lines) > 1 and lines[-1] == "0":
            log.warning("the Daytona sandbox runs commands as root; build the sandbox image to run unprivileged")

    async def stop(self) -> None:
        if self._browser:
            try:
                await self._browser.close()
            finally:
                self._browser = None
        if self.sandbox_id and self.started:
            try:
                if policy.PERSIST_ACROSS_SESSIONS:
                    await self._client.stop_sandbox(self.sandbox_id)
                else:
                    await self._client.remove_sandbox(self.sandbox_id)
            except SandboxUnavailable as exc:
                # Cleanup must not mask the reason the session is ending; the instance keeps the rest.
                log.warning("could not clean up Daytona sandbox %s: %s", self.sandbox_id, exc)
        self.started = False
        self.sandbox_id = None
        if self._client is not None and self._owns_client:
            await self._client.close()
            self._client = None

    # Execution -------------------------------------------------------------
    def _require_started(self) -> None:
        if not self.started or self._client is None:
            raise SandboxUnavailable("the Daytona sandbox is not running")

    @asynccontextmanager
    async def lease(self, *, tool: str, run_id: str | None = None,
                    task_id: str | None = None) -> AsyncIterator[Lease]:
        """Bookkeeping only: the Daytona API cannot gate a sandbox's network, so nothing opens or closes
        (policy.EGRESS_GATE_ENFORCED is False). The tool still runs, and the lease still marks its span."""
        yield Lease(id=next(self._leases), tool=tool, run_id=run_id, task_id=task_id)

    async def exec(self, argv: list[str], *, stdin: bytes | None = None,
                   timeout_s: float | None = None) -> dict[str, Any]:
        """Run a command in the sandbox. Same result shape as the Docker backend, including the
        captured-tail truncation the exec tools rely on. A Daytona-side timeout surfaces as
        ``timed_out``; a Daytona that stops answering surfaces as ``SandboxUnavailable``, never a hang."""
        self._require_started()
        limit = policy.exec_timeout(timeout_s or self.config.command_timeout_s)
        await self._push_workspace()
        parsed = await self._raw_exec(build_command(argv, stdin), limit)
        result = self._capture(parsed)
        await self._pull_workspace()
        return result

    async def _raw_exec(self, command: str, timeout_s: float) -> dict[str, Any]:
        try:
            return await self._client.execute(self.sandbox_id, command, timeout_s)
        except SandboxUnavailable as exc:
            if "did not answer" in str(exc):
                raise SandboxUnavailable(
                    f"{exc} Daytona kills executions after {policy.MAX_EXEC_TIMEOUT_S // 60} minutes at the "
                    "latest; the instance is past even that and its sandbox may be wedged") from exc
            raise

    @staticmethod
    def _capture(parsed: dict[str, Any]) -> dict[str, Any]:
        """Keep the end of each stream, like the Docker backend, and report what was dropped."""
        out, err = parsed["stdout"].encode("utf-8", "replace"), parsed["stderr"].encode("utf-8", "replace")
        kept_out, kept_err = out[-CAPTURE_BYTES:], err[-CAPTURE_BYTES:]
        return {
            "exit_code": parsed["exit_code"],
            "timed_out": parsed["timed_out"],
            "stdout": kept_out.decode("utf-8", "replace"),
            "stderr": kept_err.decode("utf-8", "replace"),
            "stdout_dropped": len(out) - len(kept_out),
            "stderr_dropped": len(err) - len(kept_err),
        }

    async def browser(self) -> DaytonaBrowserSession:
        self._require_started()
        async with self._browser_lock:
            if self._browser is None or not self._browser.alive:
                session = DaytonaBrowserSession(self._client, self.sandbox_id)
                await session.start()
                self._browser = session
            return self._browser

    # Workspace sync ----------------------------------------------------------
    def _workspace_files(self) -> list[tuple[Path, str, int]]:
        files = []
        root = self.workspace
        if not root.is_dir():
            return files
        for path in sorted(root.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            files.append((path, path.relative_to(root).as_posix(), path.stat().st_size))
        return files

    @staticmethod
    def _signature(files: list[tuple[Path, str, int]]) -> tuple:
        return (len(files), sum(f[2] for f in files),
                max((os.stat(f[0]).st_mtime_ns for f in files), default=0))

    def _sync_skipped(self, files: list[tuple[Path, str, int]]) -> bool:
        total = sum(f[2] for f in files)
        skipped = total > policy.WORKSPACE_SYNC_BUDGET_BYTES
        if skipped and not self._sync_budget_exceeded:
            self._sync_budget_exceeded = True
            log.warning("the workspace (%d bytes) is over Daytona's sync budget (%d); it will not be synced "
                        "into the sandbox, so commands will not see or change these files",
                        total, policy.WORKSPACE_SYNC_BUDGET_BYTES)
        return skipped

    async def _push_workspace(self) -> None:
        files = self._workspace_files()
        if self._sync_skipped(files) or not files:
            return
        signature = self._signature(files)
        if signature == self._ws_signature:
            return
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for path, name, size in files:
                if size > policy.SYNC_MAX_FILE_BYTES:
                    log.warning("not syncing %s to the Daytona sandbox: larger than the per-file limit", name)
                    continue
                tar.add(path, arcname=name)
        await self._client.upload(self.sandbox_id, policy.PUSH_TARBALL, buf.getvalue())
        result = self._capture(await self._raw_exec(
            f"mkdir -p /workspace && tar -xzf {policy.PUSH_TARBALL} -C /workspace "
            f"&& rm -f {policy.PUSH_TARBALL}", SYNC_COMMAND_TIMEOUT_S))
        if result["exit_code"] != 0:
            raise SandboxUnavailable(f"could not unpack the workspace into the Daytona sandbox: "
                                     f"{result['stderr'][-300:]!r}")
        self._ws_signature = self._signature(self._workspace_files())

    async def _pull_workspace(self) -> None:
        files = self._workspace_files()
        if self._sync_skipped(files):
            return
        result = self._capture(await self._raw_exec(
            f"mkdir -p /workspace && tar -czf {policy.PULL_TARBALL} -C /workspace .", SYNC_COMMAND_TIMEOUT_S))
        if result["exit_code"] != 0:
            log.warning("could not archive the sandbox workspace: %s", result["stderr"][-300:])
            return
        self._extract_tar(await self._client.download(self.sandbox_id, policy.PULL_TARBALL))
        self._ws_signature = self._signature(self._workspace_files())

    def _extract_tar(self, blob: bytes) -> None:
        """Unpack into the workspace, refusing anything that would land outside it (jail rules, not tar's)."""
        if not blob:
            return
        try:
            tar = tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz")
        except tarfile.TarError:
            log.warning("the Daytona workspace archive did not unpack")
            return
        with tar:
            for member in tar.getmembers():
                name = PurePosixPath(member.name)
                if name.is_absolute() or ".." in name.parts or not member.isfile():
                    if not member.isdir():
                        log.warning("skipping sandbox file %r: it would escape the workspace", member.name)
                    continue
                target = self.workspace / Path(*name.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink() or (target.exists() and not target.is_file()):
                    continue
                source = tar.extractfile(member)
                if source is None:
                    continue
                target.write_bytes(source.read())


class DaytonaBrowserSession:
    """``browser_server.py`` inside the Daytona sandbox, driven over the toolbox process API.

    The container backend holds ``docker exec -i``'s stdin/stdout open; Daytona's equivalent is a
    started process, ``stdin/send`` for requests and event polling for replies. The wire protocol
    (one JSON object per line, ``{"ready": ...}`` / ``{"ok": ...}``) is the same one the sandbox
    image already speaks.
    """

    def __init__(self, client: DaytonaClient, sandbox_id: str):
        self.client = client
        self.sandbox_id = sandbox_id
        self.session_id: str | None = None
        self.pid: Any = None
        self._alive = False
        self._buffer = ""
        self._index = 0
        self._lock = asyncio.Lock()

    @property
    def alive(self) -> bool:
        return self._alive

    async def start(self) -> None:
        proc = await self.client.start_process(self.sandbox_id, ["python3", "-u", "/opt/jig/browser_server.py"])
        self.session_id = str(proc.get("sessionId") or proc.get("session"))
        self.pid = proc.get("pid", proc.get("processId"))
        self._alive = True
        ready = await self._read_line(BROWSER_START_TIMEOUT_S)
        if not ready.get("ready"):
            raise ToolError(f"the sandbox browser did not start: {ready}")

    async def _pump(self) -> None:
        chunks, self._index = await self.client.process_events(self.sandbox_id, self.session_id, self._index)
        self._buffer += "".join(chunks)

    async def _read_line(self, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            if "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                try:
                    return json.loads(line)
                except ValueError:
                    raise ToolError(f"the sandbox browser answered with a non-JSON line: {line[:200]!r}") from None
            if time.monotonic() >= deadline:
                await self.close()
                raise ToolError(f"the sandbox browser did not answer within {timeout:.0f}s and was restarted")
            await asyncio.sleep(BROWSER_POLL_S)
            await self._pump()

    async def call(self, command: str, **args: Any) -> dict[str, Any]:
        async with self._lock:
            if not self._alive:
                raise ToolError("the sandbox browser is not running")
            await self.client.send_stdin(self.sandbox_id, self.session_id, self.pid,
                                         json.dumps({"command": command, **args}) + "\n")
            reply = await self._read_line(BROWSER_CALL_TIMEOUT_S)
        if not reply.get("ok"):
            raise ToolError(f"browser {command} failed: {reply.get('error')}")
        return reply["result"]

    async def close(self) -> None:
        alive, self._alive = self._alive, False
        self._buffer = ""
        if alive:
            try:
                await self.client.kill_process(self.sandbox_id, self.session_id, self.pid)
            except SandboxUnavailable as exc:
                log.warning("could not stop the Daytona sandbox browser: %s", exc)
