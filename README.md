# causal-field

**Replace hard mutex locks with Gaussian interference fields.**

`causal-field` implements probabilistic resource coordination for multi-agent
systems. Instead of hard locks that block, agents project Gaussian "splats" into
a shared latent space — overlapping splats signal contention *before* conflict
occurs. Inspired by 3D Gaussian Splatting.

```bash
pip install causal-field
```

[![PyPI version](https://badge.fury.io/py/causal-field.svg)](https://pypi.org/project/causal-field/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/brokenbartender/causal-field/actions/workflows/ci.yml/badge.svg)](https://github.com/brokenbartender/causal-field/actions)

---

## The idea

Traditional locks are binary: locked or free. Under contention, agents queue up
and throughput collapses.

`causal-field` treats resource occupancy as a **continuous probability landscape**.
Each agent projects a Gaussian splat at the resource's semantic coordinate. Before
acquiring, an agent reads the total interference at that coordinate:

- Interference below threshold → **acquired** (proceed normally)
- Interference above threshold → **soft-blocked** (defer, retry later, or fallback)

Agents sense each other's *intent* — not just their current state.

---

## Quickstart

```python
from causal_field import CausalField, render_ascii

field = CausalField(dim=64)

# Two agents want the same resource
field.soft_acquire("agent-A", "file_write:/data/output.csv")
result = field.soft_acquire("agent-B", "file_write:/data/output.csv")

print(result)
# {"acquired": False, "interference": 0.91, "resource": "file_write:/data/output.csv", ...}

# Visualize in the terminal
print(render_ascii(field))
```

---

## ASCII heatmap (zero dependencies)

```
+------------------------------------------------------------+
|                                                            |
|                    ░░░░                                    |
|                  ░░▒▒▒▒░░                                  |
|                ░░▒▒▓▓▓▒▒░░                                 |
|               ░▒▒▓▓██████▓▒░                               |
|               ░▒▒▓▓██████▓▒░                               |
|                ░░▒▒▓▓▓▒▒░░                                 |
|                  ░░▒▒▒▒░░                                  |
+------------------------------------------------------------+
  [agent-A] file_write:/data/output.csv  ████████░░ 0.80
  [agent-B] file_write:/data/output.csv  ████████░░ 0.80
```

High-intensity zones (heavy contention) appear as `█`; low as `░` or space.

---

## Benchmark

```python
from causal_field import benchmark
print(benchmark(n_agents=10, n_rounds=1000)["note"])
# "Soft-lock is 1.4x faster than threading.Lock (10 agents, 1000 rounds, dim=64)"
```

Soft-lock wins when:
- Many agents but sparse contention (most acquire without blocking)
- Read-heavy workloads (threshold=0.95 — almost never blocked)
- You need *routing* not serialization (soft-blocked agents can fallback, not deadlock)

`threading.Lock` wins when:
- You need strict mutual exclusion (critical sections with real shared memory)
- Two or fewer agents (no benefit from contention awareness)

---

## Per-resource-type thresholds

```python
from causal_field import SOFT_LOCK_THRESHOLDS
print(SOFT_LOCK_THRESHOLDS)
# {
#   "file_read":  0.95,   # permissive — reads rarely conflict
#   "file_write": 0.82,   # aggressive — writes need protection
#   "gpu":        0.78,   # most aggressive — GPU is singular
#   "db_write":   0.80,
#   "default":    0.90,
# }
```

Resources are classified by prefix — `file_write:*`, `gpu:*`, `db_write:*`.
Threshold selection is automatic.

---

## Semantic coordinates

Resources are embedded into the field using `nomic-embed-text` (via local Ollama).
Semantically similar resources land near each other — `"write_file reports.csv"` and
`"save reports.csv"` will detect mutual contention. Falls back to deterministic
SHA-256 hashing when Ollama is unavailable.

```bash
# Optional: install Ollama for semantic embeddings
pip install causal-field[embed]
```

---

## Adaptive sigma

After each acquire/release, adjust the splat's spread based on outcome:

```python
# Collision detected (splat was too wide, blocked others unnecessarily)
field.adjust_sigma("agent-A", "file_write:/data/out.csv", "collision")
# → sigma shrinks by 15%

# False positive (blocked but no real conflict existed)
field.adjust_sigma("agent-A", "file_write:/data/out.csv", "false_positive")
# → sigma grows by 15%
```

---

## SplatOptimizer

Runs gradient descent during idle cycles to drift inactive splats apart,
reducing future contention before it starts:

```python
from causal_field import CausalField, SplatOptimizer

opt = SplatOptimizer(field, lr=0.05, max_steps=200)
result = opt.run()
print(result)
# {"steps": 47, "initial_loss": 2.3, "final_loss": 0.1, "converged": True, "frozen_splats": 2}
```

Active splats (mid-task, `active=True`) are never moved — only idle splats are optimized.

---

## matplotlib visualizer (optional)

```python
from causal_field import CausalField, visualize

field = CausalField(dim=8)
for i in range(4):
    field.soft_acquire(f"agent-{i}", f"gpu:slot_{i % 2}")

visualize(field, title="GPU Slot Contention")
```

```bash
pip install causal-field[viz]
```

---

## Multi-session use

```python
from pathlib import Path
from causal_field import CausalField

# Save field state between runs
path = Path(".causal-field-state.json")
field = CausalField(dim=64)
field.soft_acquire("agent-A", "file_write:/data/x.csv")
field.save(path)

# Restore in next process
field2 = CausalField.load(path)
```

---

## Part of the LexiPro Sovereign OS

`causal-field` is extracted from **[LexiPro](https://lexipro.online)** — a local-first
agentic OS running 15 MCP servers, 228 tools, and 20 agent personas. In the full OS,
`CausalField` powers the **tool allocation layer**: before any agent acquires a
destructive tool, the field measures interference and auto-defers if another agent
already holds a write-type resource at the same coordinate.

The hard mutex (`acquire_mutex` MCP tool) remains the sole serialization gate for actual
file writes — `CausalField` is advisory, operating one layer earlier for *routing* decisions.

---

## License

MIT — see [LICENSE](LICENSE).

Built by [Broken Arrow Entertainment LLC](https://lexipro.online) · Sovereign Intelligence Systems Group
