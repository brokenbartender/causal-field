"""Tests for causal-field standalone package."""
import math
import threading
from pathlib import Path

import numpy as np
import pytest

from causal_field import (
    CausalField,
    Splat,
    SplatOptimizer,
    benchmark,
    make_splat_from_keyword,
    render_ascii,
    retract_agent_splats,
    score_tools_gaussian,
)


# ── CausalField basics ───────────────────────────────────────────────────────

def test_soft_acquire_returns_dict():
    field = CausalField(dim=8)
    result = field.soft_acquire("agent-A", "file_write:/data/x.csv")
    assert "acquired" in result
    assert "interference" in result
    assert "resource" in result


def test_first_acquire_succeeds():
    field = CausalField(dim=8)
    result = field.soft_acquire("agent-A", "gpu:slot_0")
    assert result["acquired"] is True


def test_contested_resource_blocked():
    field = CausalField(dim=8)
    field.soft_acquire("agent-A", "file_write:/data/x.csv", strength=0.99)
    # Force interference well above threshold
    field.soft_acquire("agent-B", "file_write:/data/x.csv", strength=0.99)
    result = field.soft_acquire("agent-C", "file_write:/data/x.csv", strength=0.99)
    # At least one of the three should observe contention
    assert result["interference"] > 0.0


def test_soft_release_removes_splat():
    field = CausalField(dim=8)
    field.soft_acquire("agent-A", "file_write:/data/x.csv")
    assert len(field.active_splats) == 1
    field.soft_release("agent-A", "file_write:/data/x.csv")
    assert len(field.active_splats) == 0


def test_no_self_contention():
    field = CausalField(dim=8)
    field.soft_acquire("agent-A", "file_write:/data/x.csv")
    result = field.soft_acquire("agent-A", "file_write:/data/x.csv")
    # Re-acquiring same resource should not count own splat as interference
    assert result["interference"] == pytest.approx(0.0, abs=1e-6)


# ── Splat ────────────────────────────────────────────────────────────────────

def test_splat_intensity_at_centroid():
    mu    = np.zeros(8)
    sigma = np.ones(8)
    s = Splat(agent_id="A", label="res", mu=mu, sigma_diag=sigma, alpha=1.0)
    assert s.intensity_at(mu) == pytest.approx(1.0, abs=1e-6)


def test_splat_decay():
    import time
    s = Splat(agent_id="A", label="res", mu=np.zeros(8), sigma_diag=np.ones(8),
              alpha=1.0, decay_rate=100.0)
    time.sleep(0.01)
    assert s.effective_alpha < 1.0


def test_splat_roundtrip():
    s = Splat(agent_id="X", label="gpu:0", mu=np.array([1.0, 2.0]),
              sigma_diag=np.array([1.0, 1.0]), alpha=0.7, active=True)
    d    = s.to_dict()
    s2   = Splat.from_dict(d)
    assert s2.agent_id == "X"
    assert s2.alpha == pytest.approx(0.7)
    assert s2.active is True


# ── Save/load ────────────────────────────────────────────────────────────────

def test_save_and_load(tmp_path: Path):
    p = tmp_path / "field.json"
    field = CausalField(dim=8)
    field.soft_acquire("agent-A", "file_write:/tmp/x.csv")
    field.save(p)
    field2 = CausalField.load(p)
    assert len(field2.active_splats) == 1
    assert field2.active_splats[0].agent_id == "agent-A"


def test_load_nonexistent_returns_empty(tmp_path: Path):
    field = CausalField.load(tmp_path / "nonexistent.json")
    assert len(field.active_splats) == 0


# ── Adaptive sigma ───────────────────────────────────────────────────────────

def test_adjust_sigma_collision_shrinks():
    field = CausalField(dim=8)
    field.soft_acquire("agent-A", "file_write:/tmp/x.csv")
    sigma_before = field.active_splats[0].sigma_diag.copy()
    field.adjust_sigma("agent-A", "file_write:/tmp/x.csv", "collision")
    sigma_after = field.active_splats[0].sigma_diag
    assert sigma_after[0] < sigma_before[0]


def test_adjust_sigma_false_positive_grows():
    field = CausalField(dim=8)
    field.soft_acquire("agent-A", "file_write:/tmp/x.csv")
    sigma_before = field.active_splats[0].sigma_diag.copy()
    field.adjust_sigma("agent-A", "file_write:/tmp/x.csv", "false_positive")
    sigma_after = field.active_splats[0].sigma_diag
    assert sigma_after[0] > sigma_before[0]


# ── ASCII renderer ───────────────────────────────────────────────────────────

def test_render_ascii_returns_string():
    field = CausalField(dim=8)
    field.soft_acquire("agent-A", "file_write:/data/x.csv")
    output = render_ascii(field, width=20, height=5)
    assert isinstance(output, str)
    assert "+" in output


def test_render_ascii_empty_field():
    field = CausalField(dim=8)
    output = render_ascii(field, width=20, height=5)
    assert isinstance(output, str)


# ── Benchmark ────────────────────────────────────────────────────────────────

def test_benchmark_returns_dict():
    result = benchmark(n_agents=2, n_rounds=10, dim=8)
    assert "soft_lock_ops_per_sec" in result
    assert "mutex_ops_per_sec" in result
    assert "speedup" in result
    assert "note" in result


def test_benchmark_ops_positive():
    result = benchmark(n_agents=2, n_rounds=10, dim=8)
    assert result["soft_lock_ops_per_sec"] > 0
    assert result["mutex_ops_per_sec"] > 0


# ── SplatOptimizer ───────────────────────────────────────────────────────────

def test_optimizer_reduces_interference():
    field = CausalField(dim=8)
    for i in range(3):
        field.soft_acquire(f"idle-{i}", f"gpu:slot_{i}")
    opt    = SplatOptimizer(field, lr=0.05, max_steps=50)
    result = opt.run()
    assert result["final_loss"] <= result["initial_loss"] + 1e-6


def test_optimizer_skips_active_splats():
    field = CausalField(dim=8)
    field.soft_acquire("agent-A", "file_write:/active")
    field.set_active("agent-A", "file_write:/active", True)
    mu_before = field.active_splats[0].mu.copy()
    opt = SplatOptimizer(field, lr=0.1, max_steps=50)
    opt.run()
    np.testing.assert_array_equal(field.active_splats[0].mu, mu_before)


# ── Utilities ────────────────────────────────────────────────────────────────

def test_score_tools_gaussian():
    intent = np.array([1.0, 0.0, 0.0])
    tools  = {
        "tool_a": np.array([1.0, 0.0, 0.0]),
        "tool_b": np.array([0.0, 1.0, 0.0]),
    }
    scores = score_tools_gaussian(intent, tools)
    assert scores[0][0] == "tool_a"  # most similar


def test_make_splat_from_keyword():
    s = make_splat_from_keyword("agent-X", "label", "write_file", dim=8)
    assert s.agent_id == "agent-X"
    assert len(s.mu) == 8


def test_retract_agent_splats_no_state(tmp_path: Path):
    p = tmp_path / "no_file.json"
    n = retract_agent_splats("agent-A", path=p)
    assert n == 0
