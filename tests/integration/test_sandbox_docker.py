"""Sandbox guarantees, checked against a real Docker daemon.

The install stage needs internet access to reach PyPI. Run with: pytest -m docker
"""

from __future__ import annotations

from pathlib import Path

import docker
import pytest

from prreview.config import SandboxSettings
from prreview.sandbox.container import SANDBOX_LABEL
from prreview.sandbox.docker_sandbox import DockerSandbox

pytestmark = pytest.mark.docker


@pytest.fixture(scope="module")
def docker_client() -> docker.DockerClient:
    client = docker.from_env()
    try:
        client.ping()
    except docker.errors.DockerException as exc:
        pytest.skip(f"Docker daemon not reachable: {exc}")
    return client


def make_repo(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


def sandbox(client: docker.DockerClient, **overrides: object) -> DockerSandbox:
    return DockerSandbox(SandboxSettings(_env_file=None, **overrides), client)


def test_reports_passing_and_failing_tests(docker_client: docker.DockerClient, tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {
        "calc.py": "def add(a, b):\n    return a - b\n",
        "tests/test_calc.py": "from calc import add\n\ndef test_zero():\n    assert add(0, 0) == 0\n\n"
                              "def test_add():\n    assert add(2, 2) == 4\n",
    })
    result = sandbox(docker_client).run_tests(repo)
    assert result.install.succeeded, result.install.log_tail
    assert result.status == "failed"
    outcomes = {c.test_id: c.outcome for c in result.cases}
    assert outcomes == {"tests.test_calc::test_zero": "passed", "tests.test_calc::test_add": "failed"}


def test_network_is_unreachable_during_tests(docker_client: docker.DockerClient, tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"tests/test_net.py": (
        "import socket, pytest\n\n"
        "def test_no_egress():\n"
        "    with pytest.raises(OSError):\n"
        "        socket.create_connection(('1.1.1.1', 443), timeout=3)\n"
    )})
    result = sandbox(docker_client).run_tests(repo)
    assert result.status == "passed", result.command.output if result.command else result


def test_root_filesystem_is_read_only_and_user_is_nobody(docker_client: docker.DockerClient, tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"a.txt": "x"})
    _, result = sandbox(docker_client).run_command(repo, ["sh", "-c", "id -u; touch /etc/pwned"])
    assert result.exit_code != 0
    # stdout and stderr arrive as separate multiplexed streams, so their relative order is not fixed.
    assert "65534" in result.output.splitlines()
    assert "Read-only file system" in result.output


def test_timeout_kills_runaway_code(docker_client: docker.DockerClient, tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"spin.py": "while True:\n    pass\n"})
    _, result = sandbox(docker_client, run_timeout_s=3).run_command(repo, ["python", "spin.py"])
    assert result.timed_out
    assert result.duration_s < 10


def test_memory_limit_kills_allocation_bomb(docker_client: docker.DockerClient, tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"bomb.py": "x = bytearray(600 * 1024 * 1024)\n"})
    _, result = sandbox(docker_client, memory_mb=256).run_command(repo, ["python", "bomb.py"])
    assert result.exit_code == 137
    assert result.killed_by_signal_9 and not result.timed_out


def test_output_is_capped(docker_client: docker.DockerClient, tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"spam.py": "import sys\nsys.stdout.write('x' * 5_000_000)\n"})
    _, result = sandbox(docker_client, max_output_bytes=10_000).run_command(repo, ["python", "spam.py"])
    assert result.exit_code == 0
    assert result.output_truncated and len(result.output) == 10_000


def test_installs_wheels_and_skips_index_overrides(docker_client: docker.DockerClient, tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {
        "requirements.txt": "--extra-index-url https://evil.example/simple\nsix==1.17.0\n",
        "tests/test_dep.py": "import six\n\ndef test_six():\n    assert six.PY3\n",
    })
    result = sandbox(docker_client).run_tests(repo)
    assert result.install.requested == ["six==1.17.0", "pytest==9.1.1"]
    assert result.install.skipped_lines == ["requirements.txt: --extra-index-url https://evil.example/simple"]
    assert result.status == "passed", result.install.log_tail


def test_sdist_only_dependency_is_refused(docker_client: docker.DockerClient, tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"requirements.txt": "pycrypto==2.6.1\n", "tests/test_x.py": "def test_x():\n    pass\n"})
    result = sandbox(docker_client).run_tests(repo)
    assert not result.install.succeeded
    assert "pycrypto" in result.install.log_tail


def test_no_containers_or_volumes_left_behind(docker_client: docker.DockerClient, tmp_path: Path) -> None:
    sandbox(docker_client).run_command(make_repo(tmp_path, {"a.txt": "x"}), ["true"])
    label = {"label": SANDBOX_LABEL}
    assert docker_client.containers.list(all=True, filters=label) == []
    assert docker_client.volumes.list(filters=label) == []
