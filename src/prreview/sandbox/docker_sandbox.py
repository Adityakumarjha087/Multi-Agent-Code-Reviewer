"""Run a repository's tests (or a command) in a two-stage Docker sandbox.

Stage 1 (install): network ON, wheels only (`--only-binary :all:`, so no setup.py runs),
no secrets, writes only to a dedicated volume.
Stage 2 (run): network OFF. The source volume and the dependency volume are mounted
read-only, the source is copied into a size-capped tmpfs, and the command runs there.
"""

from __future__ import annotations

import io
import logging
import os
import tarfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import docker
from docker.errors import NotFound
from docker.models.containers import Container
from docker.models.volumes import Volume

from prreview.config import SandboxSettings
from prreview.sandbox.container import (
    SANDBOX_LABEL,
    exec_capped,
    read_file_capped,
    running_container,
)
from prreview.sandbox.deps import collect_requirements
from prreview.sandbox.junit import JUnitParseError, parse_junit
from prreview.schemas.sandbox import CommandResult, DependencyInstall, TestRunResult, TestRunStatus

logger = logging.getLogger(__name__)

SRC_MOUNT = "/src"
DEPS_MOUNT = "/deps-vol"
DEPS_DIR = f"{DEPS_MOUNT}/deps"
WORKDIR = "/workspace"
_JUNIT_MAX_BYTES = 5_000_000
_LOG_TAIL_CHARS = 4_000
_PYTEST_EXIT_STATUS: dict[int, TestRunStatus] = {0: "passed", 1: "failed", 5: "no_tests"}
RUN_ENV = {"HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": f"{DEPS_DIR}:{WORKDIR}:{WORKDIR}/src"}
INSTALL_ENV = {"HOME": "/tmp"}


class InvalidTestPathError(ValueError):
    """A requested test path could be read by pytest as an option or escape the repo."""


class DockerSandbox:
    """Executes untrusted repository code under the limits in `SandboxSettings`."""

    def __init__(self, settings: SandboxSettings, client: docker.DockerClient | None = None) -> None:
        self._settings = settings
        self._client = client or docker.from_env()

    def run_tests(self, repo_dir: Path, test_paths: list[str] | None = None) -> TestRunResult:
        """Install declared dependencies, then run pytest offline and parse its JUnit report."""
        paths = [validate_test_path(p) for p in test_paths or []]
        with self._source_volume(repo_dir) as src, self._deps_volume() as deps:
            install = self._install(repo_dir, deps)
            with self._run_container(src, deps) as container:
                prepared = self._prepare_workspace(container)
                if prepared is not None:
                    return TestRunResult(status=_failure_status(prepared), install=install, command=prepared,
                                         cases=[], notes=["copying the repository into the sandbox failed"])
                return self._pytest(container, install, paths)

    def run_command(self, repo_dir: Path, argv: list[str], install_deps: bool = False) -> tuple[DependencyInstall | None, CommandResult]:
        """Run an arbitrary command offline inside a fresh copy of the repository."""
        if not argv:
            raise ValueError("argv must not be empty")
        with self._source_volume(repo_dir) as src, self._deps_volume() as deps:
            install = self._install(repo_dir, deps) if install_deps else None
            with self._run_container(src, deps) as container:
                failed_copy = self._prepare_workspace(container)
                if failed_copy is not None:
                    return install, failed_copy
                return install, self._exec(container, argv, self._settings.run_timeout_s)

    def remove_orphans(self) -> int:
        """Remove sandbox containers/volumes left behind by a crashed process. Returns count removed."""
        label = {"label": SANDBOX_LABEL}
        containers = self._client.containers.list(all=True, filters=label)
        volumes = self._client.volumes.list(filters=label)
        for container in containers:
            container.remove(force=True)
        for volume in volumes:
            volume.remove(force=True)
        return len(containers) + len(volumes)

    def _install(self, repo_dir: Path, deps: Volume) -> DependencyInstall:
        collected = collect_requirements(repo_dir, self._settings.max_requirements, self._settings.default_pytest_spec)
        argv = ["python", "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check",
                "--no-input", "--only-binary", ":all:", "--target", DEPS_DIR, *collected.specs]
        volumes = {deps.name: {"bind": DEPS_MOUNT, "mode": "rw"}}
        lifetime = self._settings.install_timeout_s + 60
        with running_container(self._client, self._settings, network=True, volumes=volumes,
                               tmpfs={"/tmp": "size=512m"}, lifetime_s=lifetime) as container:
            result = self._exec(container, argv, self._settings.install_timeout_s, env=INSTALL_ENV, workdir=None)
            size_note = self._check_deps_size(container) if result.exit_code == 0 else None
        succeeded = result.exit_code == 0 and not result.timed_out and size_note is None
        log = result.output[-_LOG_TAIL_CHARS:] + (f"\n{size_note}" if size_note else "")
        return DependencyInstall(requested=collected.specs, skipped_lines=collected.skipped,
                                 succeeded=succeeded, log_tail=log)

    def _check_deps_size(self, container: Container) -> str | None:
        result = self._exec(container, ["du", "-sk", DEPS_DIR], 60, env=INSTALL_ENV, workdir=None)
        size_kb = int(result.output.split()[0]) if result.exit_code == 0 and result.output else 0
        limit_kb = self._settings.max_deps_mb * 1024
        return f"dependencies use {size_kb} KB, over the {limit_kb} KB limit" if size_kb > limit_kb else None

    def _pytest(self, container: Container, install: DependencyInstall, paths: list[str]) -> TestRunResult:
        report = f"/tmp/prreview-junit-{uuid.uuid4().hex}.xml"
        argv = ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={report}", *paths]
        command = self._exec(container, argv, self._settings.run_timeout_s)
        notes: list[str] = []
        cases = []
        xml = None if command.timed_out else read_file_capped(self._client, container, report, _JUNIT_MAX_BYTES)
        if xml:
            try:
                cases = parse_junit(xml)
            except JUnitParseError as exc:
                notes.append(str(exc))
        status = _test_status(command, install, report_found=xml is not None)
        if command.killed_by_signal_9:
            notes.append("process was SIGKILLed without a timeout: most likely the memory limit")
        return TestRunResult(status=status, install=install, command=command, cases=cases, notes=notes)

    def _prepare_workspace(self, container: Container) -> CommandResult | None:
        """Copy the read-only source into the writable tmpfs. Returns the failed result, or None on success."""
        result = self._exec(container, ["cp", "-r", f"{SRC_MOUNT}/.", WORKDIR], 120)
        return None if result.exit_code == 0 and not result.timed_out else result

    def _exec(self, container: Container, argv: list[str], timeout_s: int,
              env: dict[str, str] = RUN_ENV, workdir: str | None = WORKDIR) -> CommandResult:
        return exec_capped(self._client, container, argv, timeout_s=timeout_s,
                           max_output_bytes=self._settings.max_output_bytes, workdir=workdir, environment=env)

    @contextmanager
    def _run_container(self, src: Volume, deps: Volume) -> Iterator[Container]:
        volumes = {src.name: {"bind": SRC_MOUNT, "mode": "ro"}, deps.name: {"bind": DEPS_MOUNT, "mode": "ro"}}
        tmpfs = {WORKDIR: f"size={self._settings.workspace_tmpfs_mb}m,exec", "/tmp": "size=256m,exec"}
        with running_container(self._client, self._settings, network=False, volumes=volumes, tmpfs=tmpfs,
                               lifetime_s=self._settings.run_timeout_s + 300) as container:
            yield container

    @contextmanager
    def _source_volume(self, repo_dir: Path) -> Iterator[Volume]:
        with self._volume() as volume:
            self._load_volume(volume, SRC_MOUNT, _tar_directory(repo_dir))
            yield volume

    @contextmanager
    def _deps_volume(self) -> Iterator[Volume]:
        with self._volume() as volume:
            self._load_volume(volume, DEPS_MOUNT, _tar_owned_dir("deps", uid=65534))
            yield volume

    @contextmanager
    def _volume(self) -> Iterator[Volume]:
        volume = self._client.volumes.create(labels={SANDBOX_LABEL: "1"})
        try:
            yield volume
        finally:
            _remove_volume(volume)

    def _load_volume(self, volume: Volume, mount: str, archive: bytes) -> None:
        """Copy a tar archive into a volume through a container that is created but never started."""
        loader = self._client.containers.create(
            self._settings.image, ["true"], network_mode="none", labels={SANDBOX_LABEL: "1"},
            volumes={volume.name: {"bind": mount, "mode": "rw"}},
        )
        try:
            loader.put_archive(mount, archive)
        finally:
            loader.remove(force=True)


def validate_test_path(path: str) -> str:
    """Allow only relative repo paths (optionally with a `::test` selector) that cannot become pytest options."""
    file_part = path.split("::", 1)[0]
    if not path or path.startswith("-") or os.path.isabs(file_part) or ".." in Path(file_part).parts:
        raise InvalidTestPathError(f"invalid test path: {path!r}")
    return path


def _test_status(command: CommandResult, install: DependencyInstall, report_found: bool) -> TestRunStatus:
    if command.timed_out:
        return "timeout"
    if not install.succeeded and not report_found:
        return "install_failed"
    if command.exit_code is None:
        return "error"
    return _PYTEST_EXIT_STATUS.get(command.exit_code, "error")


def _failure_status(command: CommandResult) -> TestRunStatus:
    return "timeout" if command.timed_out else "error"


def _tar_directory(repo_dir: Path) -> bytes:
    """Archive the snapshot with ownership reset to root and permissions normalised to 644/755."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for entry in sorted(repo_dir.iterdir()):
            tar.add(entry, arcname=entry.name, filter=_normalise_member)
    return buffer.getvalue()


def _normalise_member(member: tarfile.TarInfo) -> tarfile.TarInfo:
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    executable = member.isdir() or member.mode & 0o111
    member.mode = 0o755 if executable else 0o644
    return member


def _tar_owned_dir(name: str, uid: int) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        info = tarfile.TarInfo(name)
        info.type, info.mode, info.uid, info.gid = tarfile.DIRTYPE, 0o755, uid, uid
        tar.addfile(info)
    return buffer.getvalue()


def _remove_volume(volume: Volume) -> None:
    try:
        volume.remove(force=True)
    except NotFound:
        logger.info("volume %s already removed", volume.name)
