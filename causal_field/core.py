"""
causal-field: Gaussian resource-contention fields for multi-agent coordination.

Probabilistic resource allocation using Gaussian kernel interference.
Replaces hard mutex locks with a differentiable Soft-Lock system inspired
by 3D Gaussian Splatting — agents project Gaussian "splats" into a shared
latent space; overlapping splats signal contention before conflict occurs.

Key concepts:
  - Splat: a Gaussian influence projection (centroid μ, diagonal covariance Σ, strength α)
  - CausalField: the sum of all active splats; measures interference at any point
  - soft_acquire: project a splat and check if the field is already contested
  - SplatOptimizer: gradient descent to minimize pairwise interference during idle cycles

Features:
  - Per-resource-type thresholds (file_read=0.95, file_write=0.82, gpu=0.78)
  - Semantic embeddings via nomic-embed-text (Ollama), SHA-256 hash fallback
  - Adaptive sigma: shrink on collision, grow on false positive
  - ASCII heatmap renderer — zero extra dependencies
  - matplotlib visualizer — optional
  - Benchmark: soft-lock vs threading.Lock throughput comparison
"""

from __future__ import annotations
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Dict, Optional

import numpy as np

# ── Field dimensionality ─────────────────────────────────────────────────────
EMBED_DIM         = 768   # nomic-embed-text output dimension
HASH_FALLBACK_DIM = 8     # used only when Ollama is unavailable
DEFAULT_FIELD_DIM = EMBED_DIM

CONTENTION_THRESHOLD_VALUE = 0.75  # legacy compatibility constant
STATE_PATH = Path(".causal-field-state.json")

# ── Per-resource-type soft-lock thresholds ───────────────────────────────────
SOFT_LOCK_THRESHOLDS: Dict[str, float] = {
    "file_read":  0.95,   # permissive — reads rarely conflict
    "file_write": 0.82,   # aggressive — writes need protection
    "gpu":        0.78,   # most aggressive — GPU is a singular resource
    "db_write":   0.80,
    "default":    0.90,   # fallback for unclassified resources
}

_mu_cache: Dict[str, np.ndarray] = {}

log = logging.getLogger("CausalField")


def _get_threshold(resource: str) -> float:
    """Return per-resource-type soft-lock threshold based on resource name prefix."""
    r = resource.lower()
    if "gpu" in r or "cuda" in r or "vram" in r:
        return SOFT_LOCK_THRESHOLDS["gpu"]
    if any(kw in r for kw in ("db", "database", "sql", "postgres", "mongo")):
        if any(kw in r for kw in ("write", "insert", "update", "delete", "drop")):
            return SOFT_LOCK_THRESHOLDS["db_write"]
    if any(kw in r for kw in ("write", "edit", "save", "delete", "create", "append")):
        return SOFT_LOCK_THRESHOLDS["file_write"]
    if any(kw in r for kw in ("read", "get", "fetch", "query", "scan", "list")):
        return SOFT_LOCK_THRESHOLDS["file_read"]
    return SOFT_LOCK_THRESHOLDS["default"]


@dataclass
class Splat:
    """A discrete Gaussian influence projection in the latent resource space."""
    agent_id: str
    label: str
    mu: np.ndarray           # Centroid (D,)
    sigma_diag: np.ndarray   # Covariance diagonal (D,)
    alpha: float = 0.8       # Influence strength [0, 1]
    ts: float = field(default_factory=time.time)
    active: bool = False     # Frozen from optimizer drift while True
    decay_rate: float = 0.0  # Per-second exponential decay; 0 = persistent

    @property
    def effective_alpha(self) -> float:
        if self.decay_rate <= 0.0:
            return self.alpha
        return self.alpha * math.exp(-self.decay_rate * (time.time() - self.ts))

    def intensity_at(self, x: np.ndarray) -> float:
        epsilon = 1e-10
        safe_sigma = np.maximum(self.sigma_diag, epsilon)
        d    = min(len(x), len(self.mu))
        diff = x[:d] - self.mu[:d]
        exponent = -0.5 * np.sum((diff ** 2) / safe_sigma[:d])
        return self.effective_alpha * math.exp(exponent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id":   self.agent_id,
            "label":      self.label,
            "mu":         self.mu.tolist(),
            "sigma_diag": self.sigma_diag.tolist(),
            "alpha":      self.alpha,
            "ts":         self.ts,
            "active":     self.active,
            "decay_rate": self.decay_rate,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Splat":
        return cls(
            agent_id=d["agent_id"],
            label=d["label"],
            mu=np.array(d["mu"], dtype=np.float64),
            sigma_diag=np.array(d["sigma_diag"], dtype=np.float64),
            alpha=float(d.get("alpha", 0.8)),
            ts=float(d.get("ts", time.time())),
            active=bool(d.get("active", False)),
            decay_rate=float(d.get("decay_rate", 0.0)),
        )


class CausalField:
    """Multi-agent resource contention field.

    Manages the superposition of all active Gaussian influence projections.
    Use ``soft_acquire`` / ``soft_release`` as you would a threading.Lock —
    except that the "lock" is probabilistic and agents sense each other before
    colliding.

    Example::

        field = CausalField(dim=64)
        result = field.soft_acquire("agent-A", "file_write:/data/out.csv")
        if result["acquired"]:
            # do the write
            field.soft_release("agent-A", "file_write:/data/out.csv")
    """

    def __init__(self, dim: int = DEFAULT_FIELD_DIM):
        self.dim = dim
        self.active_splats: List[Splat] = []

    def project(self, splat: Splat) -> None:
        self.retract(splat.agent_id, splat.label)
        self.active_splats.append(splat)

    def retract(self, agent_id: str, label: str = "") -> int:
        initial = len(self.active_splats)
        if label:
            self.active_splats = [
                s for s in self.active_splats
                if not (s.agent_id == agent_id and s.label == label)
            ]
        else:
            self.active_splats = [s for s in self.active_splats if s.agent_id != agent_id]
        return initial - len(self.active_splats)

    def intensity_at(self, x: np.ndarray) -> float:
        """Vectorized total field interference at point x."""
        if not self.active_splats:
            return 0.0
        d = len(x)
        try:
            mus    = np.stack([s.mu[:d] for s in self.active_splats])
            alphas = np.array([s.effective_alpha for s in self.active_splats])
            sigmas = np.array([s.sigma_diag[:d].mean() for s in self.active_splats])
            diffs  = mus - x[:d]
            distances = np.sum(diffs ** 2, axis=1) / (2.0 * np.maximum(sigmas, 1e-10) ** 2)
            return float((alphas * np.exp(-distances)).sum())
        except Exception:
            return sum(s.intensity_at(x) for s in self.active_splats)

    def render_grid(self, resolution: int = 20) -> np.ndarray:
        grid_range = np.linspace(-2.0, 2.0, resolution)
        grid_x, grid_y = np.meshgrid(grid_range, grid_range)
        coords = np.zeros((resolution, resolution, self.dim))
        coords[..., 0] = grid_x
        coords[..., 1] = grid_y
        total_intensity = np.zeros((resolution, resolution))
        for s in self.active_splats:
            diff = coords - s.mu
            safe_sigma = np.maximum(s.sigma_diag, 1e-10)
            exponent = -0.5 * np.sum((diff ** 2) / safe_sigma, axis=-1)
            total_intensity += s.alpha * np.exp(exponent)
        return total_intensity

    def find_contention_zones(self) -> list[dict[str, Any]]:
        zones = []
        n = len(self.active_splats)
        for i in range(n):
            for j in range(i + 1, n):
                s1, s2 = self.active_splats[i], self.active_splats[j]
                combined = (s1.intensity_at(s2.mu) + s2.intensity_at(s1.mu)) / 2.0
                if combined > CONTENTION_THRESHOLD_VALUE:
                    zones.append({
                        "agents":      [s1.agent_id, s2.agent_id],
                        "labels":      [s1.label, s2.label],
                        "interference": combined,
                        "suggestion":  "Drift mu or reduce alpha",
                    })
        return zones

    def soft_acquire(
        self,
        agent_id: str,
        resource: str,
        strength: float = 0.8,
        spread: float = 1.0,
    ) -> dict[str, Any]:
        """Probabilistic lock acquisition.

        1. Retracts any prior splat for this agent+resource (avoids self-contention).
        2. Measures external field intensity at resource coordinate.
        3. Projects the new splat (soft-blocked callers still project, for field continuity).

        Returns::

            {
                "acquired":     bool,   # False = field is contested above threshold
                "interference": float,  # Current interference level [0, 1+]
                "resource":     str,
                "agent_id":     str,
            }
        """
        mu = _resource_to_mu(resource, self.dim)
        self.retract(agent_id, resource)
        current_intensity = self.intensity_at(mu)
        threshold  = _get_threshold(resource)
        is_blocked = current_intensity > threshold

        new_splat = Splat(
            agent_id=agent_id,
            label=resource,
            mu=mu,
            sigma_diag=np.full(self.dim, spread),
            alpha=strength,
            active=not is_blocked,
        )
        self.project(new_splat)
        return {
            "acquired":     not is_blocked,
            "interference": current_intensity,
            "resource":     resource,
            "agent_id":     agent_id,
        }

    def soft_release(self, agent_id: str, resource: str) -> dict[str, Any]:
        """Release a soft-lock and unfreeze the splat."""
        self.set_active(agent_id, resource, active=False)
        removed = self.retract(agent_id, resource)
        return {"released": removed > 0, "resource": resource}

    def adjust_sigma(self, agent_id: str, resource: str, outcome: str) -> int:
        """Adapt splat spread based on observed outcome.

        outcome: ``"collision"``      → shrink sigma (was too wide)
                 ``"false_positive"`` → grow sigma  (was too narrow)
                 ``"success"``        → noop

        Returns number of splats updated.
        """
        _SHRINK, _GROW = 0.85, 1.15
        _MIN_S, _MAX_S = 0.1, 10.0
        updated = 0
        for s in self.active_splats:
            if s.agent_id == agent_id and s.label == resource:
                if outcome == "collision":
                    s.sigma_diag = np.maximum(s.sigma_diag * _SHRINK, _MIN_S)
                elif outcome == "false_positive":
                    s.sigma_diag = np.minimum(s.sigma_diag * _GROW, _MAX_S)
                updated += 1
        return updated

    def set_active(self, agent_id: str, resource: str, active: bool) -> int:
        """Freeze (active=True) or unfreeze (active=False) a splat.

        Active splats are never moved by SplatOptimizer — call this on
        task start/end to prevent optimizer drift mid-task.
        """
        updated = 0
        for s in self.active_splats:
            if s.agent_id == agent_id and s.label == resource:
                s.active = active
                updated += 1
        return updated

    def save(self, path: Optional[Path] = None) -> Path:
        target = path or STATE_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {"dim": self.dim, "splats": [s.to_dict() for s in self.active_splats]}
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        if path is None:
            global _global_field
            _global_field = None
        return target

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "CausalField":
        target = path or STATE_PATH
        if not target.exists():
            return cls()
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        cf = cls(dim=data["dim"])
        cf.active_splats = [Splat.from_dict(s) for s in data["splats"]]
        return cf


class SplatOptimizer:
    """Minimizes field interference via numerical gradient descent.

    Run during idle cycles to drift inactive agents apart, reducing future
    contention before it starts.

    Example::

        opt = SplatOptimizer(field)
        result = opt.run()
        print(result)
        # {"steps": 47, "initial_loss": 2.3, "final_loss": 0.1, "converged": True}
    """

    def __init__(
        self,
        field: CausalField,
        lr: float = 0.05,
        max_steps: int = 200,
        tol: float = 0.0001,
    ):
        self.field     = field
        self.lr        = lr
        self.max_steps = max_steps
        self.tol       = tol

    def _total_interference(self) -> float:
        total = 0.0
        splats = self.field.active_splats
        for i, s1 in enumerate(splats):
            for j, s2 in enumerate(splats):
                if i != j:
                    total += s1.intensity_at(s2.mu)
        return total

    def _grad_mu(self, target_splat: Splat, eps: float = 0.001) -> np.ndarray:
        grad = np.zeros(self.field.dim)
        original_mu = target_splat.mu.copy()
        for d in range(self.field.dim):
            target_splat.mu[d] = original_mu[d] + eps
            f_plus = self._total_interference()
            target_splat.mu[d] = original_mu[d] - eps
            f_minus = self._total_interference()
            grad[d] = (f_plus - f_minus) / (2 * eps)
            target_splat.mu[d] = original_mu[d]
        return grad

    def run(self) -> dict[str, Any]:
        """Drift inactive splats to reduce pairwise interference.

        Active splats (mid-task, active=True) are never moved.
        """
        initial_loss  = self._total_interference()
        current_loss  = initial_loss
        steps_run     = 0
        optimizable   = [s for s in self.field.active_splats if not s.active]
        frozen_count  = len(self.field.active_splats) - len(optimizable)

        for step in range(self.max_steps):
            steps_run = step + 1
            grads = [(s, self._grad_mu(s)) for s in optimizable]
            for s, g in grads:
                s.mu -= self.lr * g
            new_loss = self._total_interference()
            if abs(current_loss - new_loss) < self.tol:
                break
            current_loss = new_loss

        return {
            "steps":         steps_run,
            "initial_loss":  initial_loss,
            "final_loss":    current_loss,
            "converged":     steps_run < self.max_steps,
            "frozen_splats": frozen_count,
        }


def _resource_to_mu(resource: str, dim: int) -> np.ndarray:
    """Semantic embedding via nomic-embed-text (Ollama), SHA-256 hash fallback.

    Semantically similar resources (e.g. "write_file X" vs "save X") land near
    each other in the field — making contention detection actually meaningful.
    """
    cache_key = f"{resource}:{dim}"
    if cache_key in _mu_cache:
        return _mu_cache[cache_key]

    try:
        import ollama
        response  = ollama.embeddings(model="nomic-embed-text", prompt=resource)
        raw_embed = np.array(response["embedding"], dtype=np.float64)
        norm = np.linalg.norm(raw_embed)
        if norm > 0:
            raw_embed = raw_embed / norm
        raw_embed = raw_embed * 2.0
        embed_len = len(raw_embed)
        if dim <= embed_len:
            result = raw_embed[:dim]
        else:
            repeats = math.ceil(dim / embed_len)
            result  = np.tile(raw_embed, repeats)[:dim]
        _mu_cache[cache_key] = result
        return result
    except Exception:
        pass  # Ollama offline or not installed — fall through to hash

    h = hashlib.sha256(resource.encode()).digest()
    needed_bytes = dim * 4
    while len(h) < needed_bytes:
        h += hashlib.sha256(h).digest()
    raw_hash = np.frombuffer(h[:needed_bytes], dtype=np.uint32).astype(np.float64)
    max_val  = raw_hash.max()
    if max_val > 0:
        raw_hash = (raw_hash / max_val) * 4.0 - 2.0
    result = raw_hash[:dim]
    _mu_cache[cache_key] = result
    return result


def retract_agent_splats(agent_id: str, path: Optional[Path] = None) -> int:
    """Load field, retract ALL splats for agent_id, save.

    Call on crash or cleanup so orphaned splats don't permanently block others.
    Best-effort: returns 0 on load error.
    """
    try:
        cf      = CausalField.load(path)
        removed = cf.retract(agent_id)
        if removed > 0:
            cf.save(path)
        return removed
    except Exception as exc:
        log.warning("retract_agent_splats(%s) failed: %s", agent_id, exc)
        return 0


def score_tools_gaussian(
    intent_vec: np.ndarray,
    tool_vecs: dict[str, np.ndarray],
    sigma: float = 1.5,
) -> list[tuple[str, float]]:
    """Gaussian tool relevance scoring.

    Score each tool embedding against an intent vector using RBF kernel.
    Higher score = more semantically similar to the intent.
    """
    scores = []
    for name, t_vec in tool_vecs.items():
        diff    = intent_vec - t_vec
        dist_sq = np.sum(diff ** 2)
        score   = math.exp(-dist_sq / (2 * (sigma ** 2)))
        scores.append((name, score))
    return sorted(scores, key=lambda x: x[1], reverse=True)


def make_splat_from_keyword(
    agent_id: str,
    label: str,
    keyword: str,
    dim: int = DEFAULT_FIELD_DIM,
    alpha: float = 0.8,
    spread: float = 1.0,
) -> Splat:
    """Convenience: create a Splat from a plain keyword string."""
    return Splat(
        agent_id=agent_id,
        label=label,
        mu=_resource_to_mu(keyword, dim),
        sigma_diag=np.full(dim, spread),
        alpha=alpha,
    )



def contention_benchmark(
    n_agents: int = 8,
    n_rounds: int = 200,
    dim: int = 64,
    n_resources: int = 2,
) -> dict:
    """Real multi-threaded contention benchmark — agents compete concurrently.

    Unlike benchmark() which runs sequentially, this spawns actual threads
    competing for the same resources simultaneously. Shows true contention
    behavior: collision rates, latency under load, and throughput degradation.

    Example::

        from causal_field import contention_benchmark
        r = contention_benchmark(n_agents=8, n_resources=2)
        print(r["soft_lock"]["collision_rate"])   # 0.12 — 12% deferred
        print(r["soft_lock"]["p99_ms"])           # p99 acquire latency
        print(r["verdict"])
    """
    import threading
    import statistics

    resource_pool = [f"file_write:/data/shared_{i}.csv" for i in range(n_resources)]
    cf = CausalField(dim=dim)

    soft_latencies: list[float] = []
    soft_collisions = 0
    _lock = threading.Lock()

    def soft_worker(agent_id: str) -> None:
        nonlocal soft_collisions
        lats = []
        for _ in range(n_rounds):
            resource = resource_pool[hash(agent_id) % n_resources]
            t0 = time.perf_counter()
            result = cf.soft_acquire(agent_id, resource)
            lats.append((time.perf_counter() - t0) * 1000)
            if result["acquired"]:
                cf.soft_release(agent_id, resource)
            else:
                with _lock:
                    soft_collisions += 1
        with _lock:
            soft_latencies.extend(lats)

    threads = [threading.Thread(target=soft_worker, args=(f"agent-{i}",)) for i in range(n_agents)]
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    soft_elapsed = time.perf_counter() - t0

    # Baseline: non-blocking mutex trylock under same concurrency
    mutex = threading.Lock()
    mutex_latencies: list[float] = []
    mutex_collisions = 0
    _mlock = threading.Lock()

    def mutex_worker() -> None:
        nonlocal mutex_collisions
        lats = []
        for _ in range(n_rounds):
            t0 = time.perf_counter()
            acquired = mutex.acquire(blocking=False)
            lats.append((time.perf_counter() - t0) * 1000)
            if acquired:
                mutex.release()
            else:
                with _mlock:
                    mutex_collisions += 1
        with _mlock:
            mutex_latencies.extend(lats)

    threads2 = [threading.Thread(target=mutex_worker) for _ in range(n_agents)]
    t0 = time.perf_counter()
    for t in threads2: t.start()
    for t in threads2: t.join()
    mutex_elapsed = time.perf_counter() - t0

    total_ops = n_agents * n_rounds
    sl_lats_sorted = sorted(soft_latencies)
    mx_lats_sorted = sorted(mutex_latencies)

    soft_ops_sec  = int(total_ops / soft_elapsed)  if soft_elapsed  > 0 else 0
    mutex_ops_sec = int(total_ops / mutex_elapsed) if mutex_elapsed > 0 else 0

    soft_cr  = round(soft_collisions  / total_ops, 3)
    mutex_cr = round(mutex_collisions / total_ops, 3)

    verdict = (
        f"Under {n_agents}-agent contention on {n_resources} resource(s): "
        f"soft-lock deferred {soft_cr*100:.1f}% of requests probabilistically "
        f"vs mutex hard-blocked {mutex_cr*100:.1f}%. "
        + ("Soft-lock distributes load without starvation." if soft_cr <= mutex_cr
           else "Mutex had fewer collisions — increase n_resources for soft-lock advantage.")
    )

    return {
        "n_agents":    n_agents,
        "n_rounds":    n_rounds,
        "n_resources": n_resources,
        "dim":         dim,
        "soft_lock": {
            "ops_per_sec":     soft_ops_sec,
            "collisions":      soft_collisions,
            "collision_rate":  soft_cr,
            "p50_ms":  round(sl_lats_sorted[len(sl_lats_sorted)//2], 4),
            "p99_ms":  round(sl_lats_sorted[min(int(len(sl_lats_sorted)*0.99), len(sl_lats_sorted)-1)], 4),
        },
        "mutex": {
            "ops_per_sec":     mutex_ops_sec,
            "collisions":      mutex_collisions,
            "collision_rate":  mutex_cr,
            "p50_ms":  round(mx_lats_sorted[len(mx_lats_sorted)//2], 4),
            "p99_ms":  round(mx_lats_sorted[min(int(len(mx_lats_sorted)*0.99), len(mx_lats_sorted)-1)], 4),
        },
        "verdict": verdict,
    }

_global_field: "CausalField | None" = None


def get_global_field() -> "CausalField":
    """Return the process-wide singleton CausalField, loaded from disk on first call."""
    global _global_field
    if _global_field is None:
        _global_field = CausalField.load()
    return _global_field


# ─────────────────────────────────────────────────────────────────────────────
# ASCII field renderer — zero extra dependencies
# ─────────────────────────────────────────────────────────────────────────────

_ASCII_RAMP = " \u2591\u2592\u2593\u2588"  # ░▒▓█


def render_ascii(
    field: "CausalField",
    width: int = 60,
    height: int = 20,
    x_range: tuple = (-2.5, 2.5),
    y_range: tuple = (-2.5, 2.5),
) -> str:
    """Render a 2D ASCII heatmap of the field using the first two dimensions.

    Works with no extra dependencies — uses Unicode block characters.
    High-intensity zones appear as █; low as ░ or space.

    Example::

        from causal_field import CausalField, render_ascii
        field = CausalField(dim=8)
        field.soft_acquire("agent-A", "file_write:/data/reports.csv")
        field.soft_acquire("agent-B", "file_write:/data/reports.csv")
        print(render_ascii(field))
    """
    x0, x1 = x_range
    y0, y1 = y_range
    xs = [x0 + (x1 - x0) * c / (width - 1)  for c in range(width)]
    ys = [y0 + (y1 - y0) * r / (height - 1) for r in range(height)]

    grid: list[list[float]] = []
    for y in ys:
        row = []
        for x in xs:
            probe = np.zeros(field.dim)
            probe[0] = x
            if field.dim > 1:
                probe[1] = y
            row.append(field.intensity_at(probe))
        grid.append(row)

    max_val = max(max(r) for r in grid) if grid else 1.0
    if max_val < 1e-9:
        max_val = 1.0

    lines = ["+" + "-" * width + "+"]
    for row in reversed(grid):
        chars = ""
        for v in row:
            idx = int((v / max_val) * (len(_ASCII_RAMP) - 1))
            chars += _ASCII_RAMP[max(0, min(idx, len(_ASCII_RAMP) - 1))]
        lines.append("|" + chars + "|")
    lines.append("+" + "-" * width + "+")

    if field.active_splats:
        lines.append("")
        for s in field.active_splats:
            peak = s.effective_alpha
            bar  = "\u2588" * int(peak * 10) + "\u2591" * (10 - int(peak * 10))
            lines.append(f"  [{s.agent_id}] {s.label[:30]:<30} {bar} {peak:.2f}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# matplotlib visualizer — optional dependency
# ─────────────────────────────────────────────────────────────────────────────

def visualize(
    field: "CausalField",
    resolution: int = 80,
    title: str = "CausalField — Resource Contention",
) -> None:
    """Render an interactive matplotlib heatmap of the field.

    Requires: ``pip install causal-field[viz]``

    Shows the 2D interference landscape across the first two field dimensions.
    Splat centroids are marked with agent labels; active splats shown in red.

    Example::

        from causal_field import CausalField, visualize
        field = CausalField(dim=8)
        for i in range(4):
            field.soft_acquire(f"agent-{i}", f"gpu:slot_{i % 2}")
        visualize(field)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        raise ImportError("visualize() requires matplotlib: pip install causal-field[viz]") from None

    grid = field.render_grid(resolution=resolution)
    fig, ax = plt.subplots(figsize=(9, 7))
    img = ax.imshow(
        grid,
        origin="lower",
        extent=[-2, 2, -2, 2],
        cmap=cm.plasma,
        interpolation="bilinear",
        vmin=0,
    )
    plt.colorbar(img, ax=ax, label="Interference intensity")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Dim 0")
    ax.set_ylabel("Dim 1")

    for s in field.active_splats:
        if len(s.mu) >= 2:
            x, y  = float(s.mu[0]), float(s.mu[1])
            color = "red" if s.active else "white"
            ax.plot(x, y, "o", color=color, markersize=8, markeredgecolor="black", linewidth=1)
            ax.annotate(
                f"{s.agent_id}\n{s.label[:20]}",
                xy=(x, y), xytext=(x + 0.1, y + 0.1),
                fontsize=7, color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
            )

    zones = field.find_contention_zones()
    if zones:
        ax.set_xlabel(f"Dim 0  —  {len(zones)} contention zone(s) detected", color="red")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark: soft-lock vs threading.Lock
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(
    n_agents: int = 8,
    n_rounds: int = 500,
    dim: int = 64,
    resources: list[str] | None = None,
) -> dict:
    """Compare throughput of Gaussian soft-lock vs threading.Lock under contention.

    Simulates n_agents competing for the same resources over n_rounds of
    acquire/release cycles.

    Example::

        from causal_field import benchmark
        results = benchmark(n_agents=10, n_rounds=1000)
        print(results["note"])
        # "Soft-lock is 1.4x faster than threading.Lock (10 agents, 1000 rounds, dim=64)"

    Returns:
        Dict with soft_lock_ops_per_sec, mutex_ops_per_sec, speedup, note.
    """
    import threading

    resources = resources or [
        "file_write:/data/reports.csv",
        "db_write:mission_ledger",
        "gpu:inference_slot_0",
    ]

    cf = CausalField(dim=dim)
    t0 = time.perf_counter()
    for _r in range(n_rounds):
        for i in range(n_agents):
            agent = f"agent-{i}"
            res   = resources[i % len(resources)]
            cf.soft_acquire(agent, res)
            cf.soft_release(agent, res)
    soft_elapsed = time.perf_counter() - t0
    soft_ops     = n_agents * n_rounds

    lock = threading.Lock()
    t0   = time.perf_counter()
    for _r in range(n_rounds):
        for _i in range(n_agents):
            with lock:
                pass
    mutex_elapsed = time.perf_counter() - t0
    mutex_ops     = n_agents * n_rounds

    soft_ops_sec  = soft_ops  / soft_elapsed  if soft_elapsed  > 0 else float("inf")
    mutex_ops_sec = mutex_ops / mutex_elapsed if mutex_elapsed > 0 else float("inf")
    speedup       = round(soft_ops_sec / mutex_ops_sec, 2) if mutex_ops_sec > 0 else float("inf")

    return {
        "n_agents":              n_agents,
        "n_rounds":              n_rounds,
        "soft_lock_ops_per_sec": int(soft_ops_sec),
        "mutex_ops_per_sec":     int(mutex_ops_sec),
        "soft_lock_wins":        soft_ops_sec > mutex_ops_sec,
        "speedup":               speedup,
        "note": (
            f"Soft-lock is {speedup}x faster than threading.Lock "
            f"({n_agents} agents, {n_rounds} rounds, dim={dim})"
            if speedup >= 1.0
            else f"threading.Lock is {round(1/speedup, 2)}x faster "
                 f"(soft-lock overhead at dim={dim})"
        ),
    }
