"""Build a `path_landscape.System` from a `SystemSpec`."""
from __future__ import annotations

from ..system import System
from .schemas import SystemSpec


def build_system_from_spec(spec: SystemSpec) -> System:
    """Construct a System from the LLM-generated spec.

    - Each `SpecUnit` becomes a `Unit` (scale and parent preserved).
    - Each `SpecInteraction` becomes an edge; recurrent edges are flagged so
      `System.unroll(T)` materializes them as forward edges across time.
    - Inputs and outputs are taken from unit `role`.
    """
    sys = System()
    # de-dup names defensively
    seen: set[str] = set()
    for u in spec.units:
        if u.name in seen:
            continue
        seen.add(u.name)
        sys.add_unit(
            u.name,
            scale=int(u.scale),
            parent=(u.parent if u.parent else None),
        )
    for it in spec.interactions:
        if it.source not in sys.units or it.target not in sys.units:
            # silently skip dangling references rather than erroring out
            continue
        sys.add_edge(
            it.source,
            it.target,
            weight=float(it.weight),
            recurrent=bool(it.recurrent),
        )

    inputs = [u.name for u in spec.units if u.role == "input" and u.name in sys.units]
    outputs = [u.name for u in spec.units if u.role == "output" and u.name in sys.units]
    if not inputs:
        raise ValueError(
            "spec has no units with role='input'; can't define path sources"
        )
    if not outputs:
        raise ValueError(
            "spec has no units with role='output'; can't define path sinks"
        )
    sys.set_input(*inputs)
    sys.set_output(*outputs)
    return sys
