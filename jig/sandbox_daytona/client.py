"""Thin HTTP client for a self-hosted Daytona control-plane API.

Every URL is relative to the configured base URL, so any host and port works; the only
configuration is the base URL and the API key. The endpoints used, on the Daytona model
(control plane at the base URL, toolbox under ``/toolbox``):

    GET    /health                                             control-plane liveness
    GET    /sandbox                                            list sandboxes
    POST   /sandbox                                            create {name, image, ...}
    GET    /sandbox/{id}                                       fetch one (state polling)
    POST   /sandbox/{id}/start | /stop                         lifecycle
    DELETE /sandbox/{id}                                       remove it for good
    POST   /toolbox/sandbox/{id}/process/execute               {input, shell, timeout}
    POST   /toolbox/sandbox/{id}/process/start                 long-lived process
    POST   /toolbox/sandbox/{id}/process/stdin/send            write to its stdin
    GET    /toolbox/sandbox/{id}/process/session/{sid}/events  read its output
    POST   /toolbox/sandbox/{id}/process/kill                  stop it
    POST   /toolbox/sandbox/{id}/files/upload?dest=...         multipart upload
    GET    /toolbox/sandbox/{id}/files/download?path=...       download bytes

Like ``sandbox_container.docker``, every failure raises ``SandboxUnavailable`` carrying Daytona's
own message; a command that runs and fails is a result, not an exception.
"""

from __future__ import annotations

import asyncio
import urllib.parse
from typing import Any

import httpx

from ..errors import SandboxUnavailable

# HTTP grace on top of the command's own limit: Daytona must kill it and answer before we give up.
EXEC_GRACE_S = 15.0
STATE_WAIT_S = 300.0
STATE_POLL_S = 2.0


def validate_base_url(base_url: str) -> str:
    """The trimmed base URL, or a clear error. No default host: the instance is self-hosted."""
    base_url = (base_url or "").strip().rstrip("/")
    if not base_url:
        raise SandboxUnavailable(
            "no Daytona API base URL is configured. Set [sandbox] daytona_api_base_url (or DAYTONA_API_URL) "
            "to the base URL of your self-hosted Daytona API, e.g. http://daytona.lan:3000"
        )
    parts = urllib.parse.urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise SandboxUnavailable(
            f"Daytona API base URL {base_url!r} is not a URL: it needs a scheme (http:// or https://) and a host"
        )
    return base_url


def validate_api_key(api_key: str) -> str:
    api_key = (api_key or "").strip()
    if not api_key:
        raise SandboxUnavailable(
            "no Daytona API key is configured. Set [sandbox] daytona_api_key (or DAYTONA_API_KEY) to the API "
            "key of your self-hosted Daytona instance"
        )
    return api_key


class DaytonaClient:
    """One authenticated session against one Daytona control plane. Never hardcodes a host."""

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 30.0,
                 client: httpx.AsyncClient | None = None):
        self.base_url = validate_base_url(base_url)
        self.api_key = validate_api_key(api_key)
        self._http = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    async def request(self, method: str, path: str, *, timeout: float | None = None,
                      **kw: Any) -> httpx.Response:
        try:
            resp = await self._http.request(method, self.url(path), headers=self.headers, timeout=timeout, **kw)
        except httpx.TimeoutException as exc:
            raise SandboxUnavailable(
                f"the Daytona API at {self.base_url} did not answer {method} {path} in time "
                f"({type(exc).__name__})") from exc
        except httpx.HTTPError as exc:
            raise SandboxUnavailable(
                f"the Daytona API at {self.base_url} is not reachable ({type(exc).__name__}: {exc}). "
                "Is your self-hosted instance running at that base URL?") from exc
        if resp.status_code >= 400:
            body = resp.text.strip().replace("\n", " ")[:400]
            raise SandboxUnavailable(f"Daytona {method} {path} failed (HTTP {resp.status_code}): {body[:300]}")
        return resp

    async def json(self, method: str, path: str, *, timeout: float | None = None,
                   **kw: Any) -> Any:
        resp = await self.request(method, path, timeout=timeout, **kw)
        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError as exc:
            raise SandboxUnavailable(f"the Daytona API answered {method} {path} with non-JSON: "
                                     f"{resp.text[:200]!r}") from exc

    async def close(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    # Control plane -----------------------------------------------------------
    async def health(self) -> dict[str, Any]:
        out = await self.json("GET", "/health", timeout=10.0)
        return out if isinstance(out, dict) else {}

    async def find_sandbox(self, name: str) -> dict[str, Any] | None:
        items = await self.json("GET", "/sandbox")
        items = items if isinstance(items, list) else items.get("items", []) if isinstance(items, dict) else []
        for item in items:
            if isinstance(item, dict) and item.get("name") == name:
                return item
        return None

    async def create_sandbox(self, payload: dict[str, Any]) -> dict[str, Any]:
        out = await self.json("POST", "/sandbox", json=payload)
        if not isinstance(out, dict) or not out.get("id"):
            raise SandboxUnavailable(f"Daytona did not create sandbox {payload.get('name')!r}: {str(out)[:300]}")
        return out

    async def sandbox_state(self, sandbox_id: str) -> str:
        out = await self.json("GET", f"/sandbox/{sandbox_id}")
        return str(out.get("state", "")) if isinstance(out, dict) else ""

    async def wait_for_state(self, sandbox_id: str, want: str = "started", *,
                             deadline_s: float = STATE_WAIT_S, poll_s: float = STATE_POLL_S) -> None:
        loops = max(1, int(deadline_s / poll_s) + 1)
        state = ""
        for _ in range(loops):
            state = await self.sandbox_state(sandbox_id)
            if state == want:
                return
            await asyncio.sleep(poll_s)
        raise SandboxUnavailable(
            f"the Daytona sandbox {sandbox_id} did not reach state {want!r} within {deadline_s:.0f}s "
            f"(last state {state!r}). Check the instance's builder and runner logs")

    async def start_sandbox(self, sandbox_id: str) -> None:
        await self.request("POST", f"/sandbox/{sandbox_id}/start")

    async def stop_sandbox(self, sandbox_id: str) -> None:
        await self.request("POST", f"/sandbox/{sandbox_id}/stop")

    async def remove_sandbox(self, sandbox_id: str) -> None:
        await self.request("DELETE", f"/sandbox/{sandbox_id}")

    # Toolbox: execution --------------------------------------------------------
    async def execute(self, sandbox_id: str, command: str, timeout_s: float) -> dict[str, Any]:
        payload = {"input": command, "shell": "sh", "timeout": int(timeout_s)}
        out = await self.json("POST", f"/toolbox/sandbox/{sandbox_id}/process/execute", json=payload,
                              timeout=timeout_s + EXEC_GRACE_S)
        return parse_execute_result(out)

    # Toolbox: files --------------------------------------------------------------
    async def upload(self, sandbox_id: str, dest: str, content: bytes) -> None:
        await self.request("POST", f"/toolbox/sandbox/{sandbox_id}/files/upload", params={"dest": dest},
                           files={"file": (dest.rsplit("/", 1)[-1] or "payload", content)}, timeout=120.0)

    async def download(self, sandbox_id: str, path: str) -> bytes:
        resp = await self.request("GET", f"/toolbox/sandbox/{sandbox_id}/files/download", params={"path": path},
                                  timeout=120.0)
        return resp.content

    # Toolbox: long-lived processes -------------------------------------------------
    async def start_process(self, sandbox_id: str, argv: list[str]) -> dict[str, Any]:
        out = await self.json("POST", f"/toolbox/sandbox/{sandbox_id}/process/start",
                              json={"command": argv[0], "args": argv[1:]})
        proc = out.get("process", out) if isinstance(out, dict) else {}
        if not isinstance(proc, dict) or not (proc.get("sessionId") or proc.get("session")):
            raise SandboxUnavailable(f"Daytona did not start the process {argv[0]!r}: {str(out)[:300]}")
        return proc

    async def send_stdin(self, sandbox_id: str, session_id: str, pid: Any, text: str) -> None:
        await self.request("POST", f"/toolbox/sandbox/{sandbox_id}/process/stdin/send",
                           json={"sessionId": session_id, "processId": pid, "input": text})

    async def process_events(self, sandbox_id: str, session_id: str, index: int) -> tuple[list[str], int]:
        """New stdout text chunks of the session, and the event index to continue reading from."""
        out = await self.json("GET", f"/toolbox/sandbox/{sandbox_id}/process/session/{session_id}/events",
                              params={"index": index})
        if not isinstance(out, dict):
            return [], index
        events = out.get("events") or []
        chunks = [str(ev.get("text") or ev.get("data") or "") for ev in events
                  if isinstance(ev, dict) and str(ev.get("type") or ev.get("event") or "stdout")
                  in ("stdout", "output")]
        return [c for c in chunks if c], int(out.get("index", index) or index)

    async def kill_process(self, sandbox_id: str, session_id: str, pid: Any) -> None:
        await self.request("POST", f"/toolbox/sandbox/{sandbox_id}/process/kill",
                           json={"sessionId": session_id, "processId": pid})


def parse_execute_result(out: Any) -> dict[str, Any]:
    """Daytona's execute reply (which nests the result under ``result`` in some releases) as jig fields."""
    data = out if isinstance(out, dict) else {}
    res = data.get("result") if isinstance(data.get("result"), dict) else data
    exit_code = res.get("exitCode", res.get("exit_code", data.get("exitCode", data.get("exit_code"))))
    exit_code = int(exit_code) if isinstance(exit_code, (int, float)) or str(exit_code).lstrip("-").isdigit() else -1
    stdout = res.get("stdout", data.get("stdout")) or ""
    stderr = res.get("stderr", data.get("stderr")) or ""
    timed_out = bool(res.get("timedOut") or data.get("timedOut")) or exit_code == 137
    return {"exit_code": exit_code, "stdout": str(stdout), "stderr": str(stderr), "timed_out": timed_out}
