"""Static safety contract; not evidence that Docker build/runtime was executed."""

from pathlib import Path
import subprocess
import sys

import pytest


def test_build_context_excludes_secrets_and_runtime_mounts():
    root = Path(__file__).resolve().parents[1]
    rules = set((root / ".dockerignore").read_text(encoding="utf-8").splitlines())
    assert {".env", ".env.*", "!.env.example", ".git/", "data/", "storage/", "experiments/chroma_db/",
            "docker_data/", "docker_storage/", "docker_chroma_db/"} <= rules


def test_live_check_refuses_non_disposable_environment_before_http(monkeypatch):
    from scripts import smoke_local_persistence as smoke
    monkeypatch.setattr(Path, "exists", lambda _: False)
    monkeypatch.setattr(smoke.httpx, "Client", lambda **_: pytest.fail("Unsafe environment reached HTTP"))
    with pytest.raises(AssertionError):
        smoke.run()


def test_live_check_refuses_optimized_python_before_http():
    result = subprocess.run([sys.executable, "-O", "-m", "scripts.smoke_local_persistence"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "require assertions enabled" in result.stderr
