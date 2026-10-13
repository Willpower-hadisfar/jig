"""Where the Daytona API is less capable than Docker, and what Jig does about it.

The container backend gets its guarantees from the Docker CLI. A self-hosted Daytona instance
offers a smaller contract, and pretending otherwise would be a silent security regression:

* **Egress.** Docker sandboxes sit on an internal-only network whose single way out is Jig's
  egress proxy, opened only while a reviewed action holds a lease. The Daytona API has no way to
  attach a sandbox to such a network or force a proxy on it, so the lease is bookkeeping only:
  it grants nothing and closes nothing. Whatever network the Daytona instance gives a sandbox is
  what the sandbox gets; treat the Daytona deployment itself as the isolation boundary and
  configure its network policy accordingly.
* **Resource caps.** ``cpus``, ``memory``, ``pids_limit``, ``tmp_size`` and ``shm_size`` are
  Docker flags; sandbox creation has no equivalent fields. They cannot be enforced through this
  API, so they are not sent (sending keys the instance rejects would fail every create). Daytona
  is asked instead to auto-stop idle sandboxes, so an abandoned sandbox does not run forever.
* **Persistence.** ``docker`` containers are removed when the session ends; Daytona sandboxes
  persist across stop/start and keep their volume. Jig follows the Docker backend's contract and
  deletes the sandbox when the session ends (``PERSIST_ACROSS_SESSIONS = False``). Workspace
  continuity comes from syncing the workspace around each command, not from the sandbox.
* **Execution time.** The execute endpoint kills any command at 10 minutes. Jig refuses longer
  limits up front rather than letting Daytona cut a long run down mid-flight.
* **Workspace.** A Daytona sandbox cannot bind-mount Jig's folder, so /workspace is a copy kept
  in step around each command, within a sync budget; beyond it, syncing is skipped and said so.
"""

from __future__ import annotations

from ..errors import ToolError

# Daytona's execute endpoint hard-kills anything still running after 10 minutes.
MAX_EXEC_TIMEOUT_S = 600
# Lease gating cannot be enforced through the Daytona API (see the module docstring).
EGRESS_GATE_ENFORCED = False
# Delete the sandbox at session end, the way the container backend removes its containers.
PERSIST_ACROSS_SESSIONS = False
# Idle sandboxes are auto-stopped by Daytona this many minutes after last use, so a crashed Jig
# does not leave one burning capacity forever.
AUTO_STOP_MINUTES = 30
# Whole-workspace cap for the around-command sync; beyond it syncing is skipped (and reported).
WORKSPACE_SYNC_BUDGET_BYTES = 64 * 1024 * 1024
# Files larger than this are left out of a sync even when the whole fits (caches, stray dumps).
SYNC_MAX_FILE_BYTES = 16 * 1024 * 1024
# Paths inside the sandbox for the sync tarballs.
PUSH_TARBALL = "/tmp/jig-ws-push.tgz"
PULL_TARBALL = "/tmp/jig-ws-pull.tgz"

# SandboxConfig fields the Daytona API cannot apply; reported at start rather than assumed.
UNSUPPORTED_LIMITS = ("cpus", "memory", "pids_limit", "tmp_size", "shm_size")


def exec_timeout(limit_s: float) -> int:
    """The command limit to send, or a clear failure. Daytona would silently cut anything over 10 minutes."""
    if limit_s > MAX_EXEC_TIMEOUT_S:
        raise ToolError(
            f"the Daytona sandbox cannot run a command limited to {limit_s:.0f}s: its API kills anything "
            f"still running after {MAX_EXEC_TIMEOUT_S // 60} minutes. Lower [sandbox] command_timeout_s "
            f"to {MAX_EXEC_TIMEOUT_S} or less, or use the container backend for long builds")
    return max(1, int(limit_s))


def create_payload(name: str, image: str, config) -> dict:
    """The create body. Docker-style caps have no Daytona equivalent (see UNSUPPORTED_LIMITS)."""
    return {"name": name, "image": image, "autoStopInterval": AUTO_STOP_MINUTES}


def unsupported_limits(config) -> list[str]:
    return [f"{name}={getattr(config, name)}" for name in UNSUPPORTED_LIMITS]
