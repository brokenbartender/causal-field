# causal-field — Formal Specification (v1.0)

## 1. Abstract

`causal-field` implements a **Gaussian Interference Field** (GIF) for probabilistic
resource coordination in multi-agent systems. Each agent projects a Gaussian
"splat" into a shared latent space. Before acquiring a resource, an agent reads
the total field intensity at the resource's coordinate — if intensity exceeds a
configurable threshold, the acquire is soft-blocked.

This document defines the mathematical model, invariants, and behavioral contracts
that any conforming implementation must satisfy.

---

## 2. Definitions

| Symbol | Type | Description |
|--------|------|-------------|
| `D` | integer ≥ 2 | Field dimensionality (default: 64; 768 with Ollama embeddings) |
| `μ_r` | ℝᴰ | Semantic coordinate of resource `r` (unit-normalized) |
| `μ_a` | ℝᴰ | Centroid of splat projected by agent `a` |
| `σ_a` | ℝ⁺ | Spread (standard deviation) of splat for agent `a` (default: 1.0) |
| `w_a` | ℝ ∈ [0,1] | Weight of splat for agent `a` (default: 1.0) |
| `θ_r` | ℝ ∈ (0,1) | Acquisition threshold for resource type of `r` |
| `I(μ)` | ℝ ∈ [0,∞) | Total field intensity at coordinate `μ` |
| `A` | set | Set of all active splats in the field |

---

## 3. Gaussian Splat Model

Each active splat `s ∈ A` is characterized by `(μ_s, σ_s, w_s)`.

The **contribution** of splat `s` at query point `μ` is:

```
φ(s, μ) = w_s · exp(−‖μ − μ_s‖² / (2 · σ_s²))
```

The **total field intensity** at `μ` is:

```
I(μ) = Σ_{s ∈ A} φ(s, μ)
```

Where `‖·‖` is the Euclidean norm over the first `min(D_s, D_μ)` dimensions
(backward-compatible when field dimensionality changes between sessions).

---

## 4. Resource Coordinate Mapping

Resources are mapped to field coordinates via one of two mechanisms:

**Primary (semantic):** When Ollama + `nomic-embed-text` are available:

```
μ_r = normalize(embed("nomic-embed-text", resource_string))
```

Where `normalize` produces a unit vector in ℝ⁷⁶⁸.

**Fallback (deterministic hash):** When Ollama is unavailable:

```
μ_r[i] = int.from_bytes(SHA256(resource_string)[i*4:(i+1)*4]) / 2³²
    for i in 0..D-1
```

This produces a deterministic coordinate in [0,1]ᴰ. Coordinates computed via
hash fallback are NOT semantically comparable to embed-based coordinates — the
field should be rebuilt after Ollama becomes available.

---

## 5. Acquire Protocol

`soft_acquire(agent_id, resource)` follows this sequence:

1. Compute `μ_r = resource_to_mu(resource)`
2. Read `I_before = I(μ_r)` — pure read, no splat projected yet
3. Determine threshold `θ_r` by resource type prefix:
   - `"file_read:*"` → 0.95
   - `"file_write:*"` → 0.82
   - `"gpu:*"` → 0.78
   - `"db_write:*"` → 0.80
   - default → 0.90
4. If `I_before ≥ θ_r`: return `{acquired: False, interference: I_before, ...}`
5. Project splat: add `(μ_r, σ_agent, 1.0)` to `A` with `active=True`
6. Read `I_after = I(μ_r)` — includes own splat
7. Return `{acquired: True, interference: I_after, ...}`

**Race window:** Steps 2 and 5 are not atomic. Two agents can both pass step 4
before either completes step 5. This is a known advisory limitation — the hard
mutex (`threading.Lock` or equivalent) must be used for serialization-critical
sections.

---

## 6. Release Protocol

`soft_release(agent_id, resource)`:

1. Find all splats in `A` where `agent_id` and `resource` match
2. Set `active=False` on each matching splat
3. Splats with `active=False` remain in `A` for ongoing intensity contribution
   but are no longer considered "held" by the agent

**Retract:** `retract_agent_splats(agent_id)` removes ALL splats (active or not)
for a given agent from `A`. Call on agent termination or Airlock events.

---

## 7. Adaptive Sigma

After each resource interaction, the splat spread self-calibrates:

| Outcome | Rule | Formula |
|---------|------|---------|
| `"collision"` | Splat was too wide — blocked others unnecessarily | `σ_new = max(σ_min, σ × 0.85)` |
| `"false_positive"` | Soft-blocked but no real conflict existed | `σ_new = min(σ_max, σ × 1.15)` |
| `"success"` | No-op | `σ_new = σ` |

Bounds: `σ_min = 0.1`, `σ_max = 10.0`.

---

## 8. Conformance Requirements

A conforming implementation MUST:

1. Compute intensity via the Gaussian formula in §3 — no linear or step approximations
2. Apply threshold comparison as `I(μ) ≥ θ` (≥, not >)
3. Support `soft_acquire`, `soft_release`, `retract_agent_splats`, `intensity_at`
4. Return `acquired: bool` and `interference: float` in every `soft_acquire` response
5. Accept `σ ∈ (0, 10.0]` and `w ∈ (0, 1.0]` as splat parameters
6. Document the advisory-only nature of `soft_acquire` (§5 Race Window)

A conforming implementation SHOULD:

- Support `adjust_sigma(agent_id, resource, outcome)` per §7
- Support `save()` / `load()` for cross-session field persistence
- Provide a terminal ASCII heatmap visualization

---

## 9. Non-Guarantees (Explicit)

- **Atomicity:** `soft_acquire` is NOT atomic. Do not use as a substitute for a mutex.
- **Completeness:** The field does not guarantee all contention is detected. Low-sigma
  splats may be invisible to distant query points.
- **Distributed coordination:** The field is in-process only. Cross-process use
  requires external serialization (e.g., `save()` + shared filesystem).
- **Real-time:** `intensity_at()` is a point-in-time snapshot. The field can change
  between read and splat projection.

---

## 10. Reference Implementation

`causal_field/core.py` in [brokenbartender/causal-field](https://github.com/brokenbartender/causal-field)
is the reference implementation. All behavioral contracts in this document are
verified by `tests/test_core.py`.

---

*Specification version 1.0 — Broken Arrow Entertainment LLC*
