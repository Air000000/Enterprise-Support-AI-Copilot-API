from __future__ import annotations

from importlib import import_module

import pytest


def _load_fuse_rrf():
    try:
        module = import_module("experiments.evals.ir.rrf")
    except ModuleNotFoundError:
        pytest.fail("RRF fusion is not implemented yet")

    fuse_rrf = getattr(module, "fuse_rrf", None)
    if fuse_rrf is None:
        pytest.fail("fuse_rrf is not implemented yet")

    return fuse_rrf


def test_rrf_rewards_documents_present_in_both_rankings() -> None:
    fuse_rrf = _load_fuse_rrf()

    fused = fuse_rrf(
        [["a", "b", "c"], ["b", "d", "a"]],
        rrf_k=60,
        top_k=4,
    )

    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_breaks_ties_deterministically() -> None:
    fuse_rrf = _load_fuse_rrf()

    fused = fuse_rrf(
        [["b", "c"], ["a", "c"]],
        rrf_k=0,
        top_k=3,
    )

    assert fused == ["a", "b", "c"]