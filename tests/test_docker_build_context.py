"""Static safety contract; not evidence that Docker build/runtime was executed."""

from pathlib import Path
import json
import shutil
import subprocess
import sys

import pytest


def test_build_context_excludes_secrets_and_runtime_mounts():
    root = Path(__file__).resolve().parents[1]
    rules = set((root / ".dockerignore").read_text(encoding="utf-8").splitlines())
    assert {".env", ".env.*", "!.env.example", ".git/", "data/", "storage/", "experiments/chroma_db/",
            "docker_data/", "docker_storage/", "docker_chroma_db/", ".worktrees/"} <= rules


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


def test_native_compose_acceptance_config_is_disposable():
    # ponytail: let Compose interpret its own merge tags; no custom YAML merger.
    if not shutil.which("docker"):
        pytest.skip("Docker CLI unavailable; no engine required for config validation")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["docker", "compose", "--env-file", ".env.example", "-p", "copilot-config-check",
        "-f", "docker-compose.yml", "-f", "docker-compose.acceptance.yml", "config", "--format", "json"],
        cwd=root, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    api = config["services"]["api"]
    assert not api.get("env_file") and not api.get("container_name")
    assert len(api["ports"]) == 1
    assert (api["ports"][0]["host_ip"], str(api["ports"][0]["published"]), api["ports"][0]["target"]) == (
        "127.0.0.1", "18080", 8000)
    assert {(m["type"], m["source"], m["target"]) for m in api["volumes"]} == {
        ("volume", "sql", "/app/data"), ("volume", "storage", "/app/storage"),
        ("volume", "chroma", "/app/experiments/chroma_db")}
    assert set(api["networks"]) == {"default"}
    assert not config["networks"]["default"].get("internal", False)
    env = api["environment"]
    assert env["DATABASE_URL"] == "sqlite:////app/data/acceptance.db"
    assert env["DOCUMENT_STORAGE_ROOT"] == "/app/storage/documents"
    assert env["PYTHON_DOTENV_DISABLED"] == "1" and env["AUTH_DEMO_USERS_JSON"] == ""
    assert all(env[key] == "false" for key in ("RAG_HYBRID_SEARCH_ENABLED", "RAG_HYBRID_ASK_ENABLED"))
    assert not any("proxy" in key.lower() for key in env)
    for prefix in ("DASHSCOPE", "DASHSCOPE_RERANK", "DASHSCOPE_PHASE_B"):
        assert env[prefix + "_API_KEY"] == "offline-acceptance-not-a-key"
        assert env[prefix + "_BASE_URL"] == "http://127.0.0.1:9/v1"
