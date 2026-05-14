"""Agentic pipeline: phenomenon → path-landscape analysis → mechanistic report.

Given a natural-language description of an emergent phenomenon, this module
runs the full thesis-framework pipeline:

  1. **specify** — call Claude with a structured-output tool to encode the
     phenomenon as a System: basic computing units, interactions (including
     feedback loops and multi-scale parent relationships), time-unrolling
     parameters.
  2. **build**   — convert the spec into a `path_landscape.System` object.
  3. **extract** — unroll feedback loops in time, coarsen scales, sample
     paths from inputs to outputs through the resulting static graph.
  4. **analyze** — cluster the paths into modes and compute the four
     cross-system landscape metrics (modes, size exponent, persistence,
     meta-graph connectivity).
  5. **interpret** — call Claude again with the spec + metrics to produce
     a focused mechanistic explanation of how the path structure produces
     (or fails to produce) emergence here.
  6. **report**  — save a markdown report, a multi-panel figure, and the
     raw spec JSON to a chosen output directory.

Entry point:

    from path_landscape.agent import analyze_emergence
    result = analyze_emergence("A flock of starlings turning as one")
"""
from .schemas import (
    SpecUnit,
    SpecInteraction,
    SpecParameter,
    SystemSpec,
)
from .pipeline import analyze_emergence, specify_system, run_analysis, interpret
from .builder import build_system_from_spec

__all__ = [
    "SpecUnit",
    "SpecInteraction",
    "SpecParameter",
    "SystemSpec",
    "analyze_emergence",
    "specify_system",
    "run_analysis",
    "interpret",
    "build_system_from_spec",
]
