"""Daytona sandbox backend: Jig runs its code/shell/browser tools against a self-hosted Daytona instance.

The Daytona backend uses Daytona's sandbox API to execute commands and run browser sessions.
It requires a configured Daytona API base URL and assumes the Daytona instance is properly
configured with the necessary sandbox capabilities.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiohttp

from ..config import SandboxConfig
from ..errors import SandboxUnavailable, ToolError
from ..sandbox_container.egress import EgressProxy, Lease

LINE_LIMIT = 64 * 1024 * 1024
SERVICE_WAIT_S = 120.0
BROWSER_START_TIMEOUT_S = 60.0
BROWSER_CALL_TIMEOUT_S = 90.0


async def _daytona_request(
    session: aiohttp.ClientSession,
    base_url: str,
    endpoint: str,
    method: str = "GET",
    json_data: dict[str, Any] | None = None,
    timeout: float = 30.0
) -> dict[str, Any]:
    """Make a request to the Daytona API."""
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        async with session.request(method, url, json=json_data, timeout=timeout) as response:
            response.raise_for_status()
            return await response.json()
    except aiohttp.ClientError as e:
        raise SandboxUnavailable(f"Daytona API request failed: {e}")
    except asyncio.TimeoutError:
        raise SandboxUnavailable(f"Daytona API request timed out after {timeout}s")


class DaytonaSandbox:
    def __init__(self, config: SandboxConfig, workspace: Path, agent_id: str, egress: EgressProxy):
        self.config = config
        self.workspace = workspace
        self.egress = egress
        self.agent_id = agent_id
        self.daytona_api_base_url = config.daytona_api_base_url
        self.started = False
        self._browser: DaytonaBrowserSession | None = None
        self._browser_lock = asyncio.Lock()
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> dict[str, Any]:
        """Start the Daytona sandbox backend and verify connection."""
        if not self.daytona_api_base_url:
            raise SandboxUnavailable("[sandbox] backend = 'daytona' requires daytona_api_base_url to be configured")
        
        self._session = aiohttp.ClientSession()
        
        # Verify Daytona connection
        try:
            health = await _daytona_request(self._session, self.daytona_api_base_url, "/health", timeout=10.0)
            if health.get("status") != "healthy":
                raise SandboxUnavailable(f"Daytona instance not healthy: {health}")
        except SandboxUnavailable as e:
            await self._session.close()
            self._session = None
            raise
        
        self.started = True
        return {"daytona": "connected", "api_base_url": self.daytona_api_base_url}

    async def stop(self) -> None:
        """Stop the Daytona sandbox backend."""
        if self._session:
            await self._session.close()
            self._session = None
        self.started = False

    def _require_started(self) -> None:
        """Ensure the sandbox has been started."""
        if not self.started:
            raise SandboxUnavailable("Daytona sandbox not started")

    async def lease(self, *, tool: str, run_id: str | None = None,
                   duration_s: float = 60.0) -> Lease:
        """Acquire a lease for network access."""
        self._require_started()
        return await self.egress.lease(tool=tool, run_id=run_id, duration_s=duration_s)

    async def exec(self, argv: list[str], *, stdin: bytes | None = None,
                  timeout_s: float | None = None) -> dict[str, Any]:
        """Execute a command in the Daytona sandbox."""
        self._require_started()
        
        timeout_s = timeout_s or self.config.command_timeout_s
        
        # Prepare command execution request
        command_request = {
            "command": argv,
            "stdin": base64.b64encode(stdin).decode() if stdin else None,
            "timeout": timeout_s,
            "workspace": str(self.workspace)
        }
        
        try:
            result = await _daytona_request(
                self._session,
                self.daytona_api_base_url,
                "/api/v1/sandbox/exec",
                method="POST",
                json_data=command_request,
                timeout=timeout_s + 10.0
            )
            
            # Convert base64 stdout/stderr back to bytes if needed
            if "stdout" in result and isinstance(result["stdout"], str):
                result["stdout"] = base64.b64decode(result["stdout"]).decode()
            if "stderr" in result and isinstance(result["stderr"], str):
                result["stderr"] = base64.b64decode(result["stderr"]).decode()
                
            return result
        except Exception as e:
            raise ToolError(f"Daytona command execution failed: {e}")

    async def browser(self) -> DaytonaBrowserSession:
        """Create a browser session in the Daytona sandbox."""
        self._require_started()
        
        async with self._browser_lock:
            if self._browser is None or not self._browser.alive():
                self._browser = DaytonaBrowserSession(self._session, self.daytona_api_base_url)
                await self._browser.start()
            return self._browser


class DaytonaBrowserSession:
    def __init__(self, session: aiohttp.ClientSession, base_url: str):
        self.session = session
        self.base_url = base_url
        self.browser_id = str(uuid.uuid4())
        self.alive_ = False

    def alive(self) -> bool:
        """Check if the browser session is still alive."""
        return self.alive_

    async def start(self) -> None:
        """Start a new browser session."""
        try:
            result = await _daytona_request(
                self.session,
                self.base_url,
                "/api/v1/sandbox/browser/start",
                method="POST",
                json_data={"browser_id": self.browser_id},
                timeout=BROWSER_START_TIMEOUT_S
            )
            self.alive_ = result.get("status") == "started"
        except Exception as e:
            raise SandboxUnavailable(f"Failed to start Daytona browser: {e}")

    async def call(self, command: str, **args: Any) -> dict[str, Any]:
        """Execute a browser command."""
        if not self.alive_:
            raise SandboxUnavailable("Daytona browser session not alive")
        
        request_data = {
            "browser_id": self.browser_id,
            "command": command,
            "args": args
        }
        
        try:
            return await _daytona_request(
                self.session,
                self.base_url,
                "/api/v1/sandbox/browser/call",
                method="POST",
                json_data=request_data,
                timeout=BROWSER_CALL_TIMEOUT_S
            )
        except Exception as e:
            raise ToolError(f"Daytona browser command failed: {e}")

    async def close(self) -> None:
        """Close the browser session."""
        if self.alive_:
            try:
                await _daytona_request(
                    self.session,
                    self.base_url,
                    f"/api/v1/sandbox/browser/stop/{self.browser_id}",
                    method="POST",
                    timeout=10.0
                )
            except Exception:
                # Ignore errors during cleanup
                pass
            self.alive_ = False