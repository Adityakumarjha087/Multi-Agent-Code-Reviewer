"""Low-level primitives: start a hardened container and run commands in it with a hard timeout.

Pattern (each step verified against Docker 29.7 / docker SDK 7.2 on 2026-10-09):
- PID 1 is `sleep`; real work runs through `exec`, so files in tmpfs can still be read
  after the command finishes (Docker's archive API cannot see tmpfs mounts).
- The timeout is a watchdog thread that kills the whole container; the in-flight exec
  then returns exit code 137.
- Output is streamed and capped. Excess is drained and discarded rather than left
  unread, because an unread pipe would block the process until the timeout.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import docker
from docker.errors import APIError
from docker.models.containers import Container

from prreview.config import SandboxSettings
from prreview.schemas.sandbox import CommandResult

logger = logging.getLogger(__name__)

SANDBOX_USER = "65534:65534"  # nobody:nogroup in Debian-based images
SANDBOX_LABEL = "prreview.sandbox"
_KILLED_EXIT_CODE = 137


def hardened_run_kwargs(settings: SandboxSettings, network: bool) -> dict[str, Any]:
    """Docker `run` options shared by every sandbox container."""
    memory = f"{settings.memory_mb}m"
    return {
        "network_mode": "bridge" if network else "none",
        "read_only": True,
        "user": SANDBOX_USER,
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges"],
        "mem_limit": memory,
        "memswap_limit": memory,  # equal to mem_limit = no swap
        "nano_cpus": int(settings.cpus * 1_000_000_000),
        "pids_limit": settings.pids_limit,
        "labels": {SANDBOX_LABEL: "1"},
    }


@contextmanager
def running_container(
    client: docker.DockerClient,
    settings: SandboxSettings,
    *,
    network: bool,
    volumes: dict[str, dict[str, str]],
    tmpfs: dict[str, str],
    lifetime_s: int,
) -> Iterator[Container]:
    """Start an idle hardened container and always force-remove it afterwards."""
    container = client.containers.run(
        settings.image,
        ["sleep", str(lifetime_s)],
        detach=True,
        volumes=volumes,
        tmpfs=tmpfs,
        **hardened_run_kwargs(settings, network),
    )
    try:
        yield container
    finally:
        container.remove(force=True)


def exec_capped(
    client: docker.DockerClient,
    container: Container,
    argv: list[str],
    *,
    timeout_s: float,
    max_output_bytes: int,
    workdir: str | None = None,
    environment: dict[str, str] | None = None,
) -> CommandResult:
    """Run `argv` in `container`, killing the container if it exceeds `timeout_s`."""
    api = client.api
    exec_id = api.exec_create(
        container.id, argv, workdir=workdir, environment=environment, user=SANDBOX_USER
    )["Id"]
    timed_out = threading.Event()
    watchdog = threading.Timer(timeout_s, _kill_on_timeout, args=(container, timed_out))
    started = time.monotonic()
    watchdog.start()
    try:
        output, truncated = _read_capped(api.exec_start(exec_id, stream=True), max_output_bytes)
    finally:
        watchdog.cancel()
    exit_code = api.exec_inspect(exec_id).get("ExitCode")
    return CommandResult(
        exit_code=exit_code,
        timed_out=timed_out.is_set(),
        killed_by_signal_9=exit_code == _KILLED_EXIT_CODE and not timed_out.is_set(),
        duration_s=round(time.monotonic() - started, 3),
        output=output.decode("utf-8", errors="replace"),
        output_truncated=truncated,
    )


def read_file_capped(client: docker.DockerClient, container: Container, path: str, max_bytes: int) -> str | None:
    """Read a file from inside the container (works for tmpfs). Returns None if it does not exist."""
    result = exec_capped(
        client, container, ["head", "-c", str(max_bytes), path], timeout_s=30, max_output_bytes=max_bytes
    )
    return result.output if result.exit_code == 0 else None


def _read_capped(chunks: Iterator[bytes], max_bytes: int) -> tuple[bytes, bool]:
    kept = bytearray()
    truncated = False
    for chunk in chunks:
        room = max_bytes - len(kept)
        if room > 0:
            kept.extend(chunk[:room])
        if len(chunk) > room:
            truncated = True
    return bytes(kept), truncated


def _kill_on_timeout(container: Container, timed_out: threading.Event) -> None:
    timed_out.set()
    try:
        container.kill()
    except APIError as exc:
        # Expected race: the command finished and the container stopped just before the kill.
        logger.info("kill after timeout failed (container likely already stopped): %s", exc)
