"""causal-field: Gaussian resource-contention fields for multi-agent coordination.

Replace hard mutex locks with soft probabilistic interference — agents sense
each other's intent through overlapping Gaussian splats in a shared latent space.

>>> from causal_field import CausalField
>>> field = CausalField(dim=64)
>>> field.soft_acquire("agent-A", "file_write:/data/output.csv")
>>> field.soft_acquire("agent-B", "file_write:/data/output.csv")
>>> from causal_field import render_ascii
>>> print(render_ascii(field))
"""
from .core import (
    CausalField,
    Splat,
    SplatOptimizer,
    render_ascii,
    visualize,
    benchmark,
    get_global_field,
    make_splat_from_keyword,
    score_tools_gaussian,
    retract_agent_splats,
    SOFT_LOCK_THRESHOLDS,
    EMBED_DIM,
    HASH_FALLBACK_DIM,
)

__all__ = [
    "CausalField",
    "Splat",
    "SplatOptimizer",
    "render_ascii",
    "visualize",
    "benchmark",
    "get_global_field",
    "make_splat_from_keyword",
    "score_tools_gaussian",
    "retract_agent_splats",
    "SOFT_LOCK_THRESHOLDS",
    "EMBED_DIM",
    "HASH_FALLBACK_DIM",
]

__version__ = "1.0.1"
