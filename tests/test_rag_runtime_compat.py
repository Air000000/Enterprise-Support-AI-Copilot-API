import importlib
import subprocess
import sys

from experiments.rag_local.query_chroma import build_where_filter as legacy_build_where_filter
from rag_runtime.query_chroma import build_where_filter as runtime_build_where_filter


def test_rag_runtime_query_chroma_importable():
    module = importlib.import_module("rag_runtime.query_chroma")
    assert hasattr(module, "build_where_filter")


def test_legacy_query_chroma_wrapper_importable():
    module = importlib.import_module("experiments.rag_local.query_chroma")
    assert hasattr(module, "build_where_filter")


def test_build_where_filter_matches_between_runtime_and_legacy_paths():
    filters = [
        {},
        {"tenant_id": "tenant_demo"},
        {"category": "it"},
        {"tenant_id": "tenant_demo", "category": "it"},
    ]

    for kwargs in filters:
        assert runtime_build_where_filter(**kwargs) == legacy_build_where_filter(**kwargs)


def test_runtime_query_chroma_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "rag_runtime.query_chroma", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "tenant-id" in result.stdout


def test_legacy_query_chroma_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "experiments.rag_local.query_chroma", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "tenant-id" in result.stdout