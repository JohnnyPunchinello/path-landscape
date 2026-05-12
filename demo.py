"""Run the path-landscape pipeline on a few example systems and save a figure.

Usage:  python demo.py

Compares:
  (a) plain feedforward chain         -> few modes, all paths the same length
  (b) feedforward + skip connections  -> more modes (length diversity)
  (c) recurrent network unrolled T=4  -> modes by recurrence depth
  (d) two-module net micro vs macro   -> illustrates scale coarsening
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt

from path_landscape import PathLandscape, enumerate_paths
from path_landscape.examples import (
    feedforward_chain,
    feedforward_with_skip,
    mixture_of_experts,
    simple_rnn,
)

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.manifold._mds")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._mds")


def landscape_for(sys, T=1, max_paths=5000, eps=0.4, max_length=None):
    g = sys.unroll(T)
    paths = enumerate_paths(
        g,
        sys.unroll_sources(T),
        sys.unroll_sinks(T),
        max_paths=max_paths,
        max_length=max_length,
    )
    if not paths:
        raise RuntimeError("no paths found from inputs to outputs")
    L = PathLandscape(paths)
    L.cluster(eps=eps, min_samples=2)
    return L


def report(name, L):
    print(f"\n=== {name} ===")
    print(f"  {L.describe()}")
    for c in L.cluster_summary():
        print(f"  {c}")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sys1 = feedforward_chain(depth=5, width=3)
    print(sys1.summary())
    L1 = landscape_for(sys1, T=1, max_paths=5000, eps=0.45)
    report("feedforward chain (depth=5, width=3)", L1)
    L1.plot(ax=axes[0, 0], title="feedforward chain")

    sys2 = feedforward_with_skip(depth=5, width=3, skip_every=2)
    print(sys2.summary())
    L2 = landscape_for(sys2, T=1, max_paths=5000, eps=0.45)
    report("feedforward + skip (depth=5, width=3, skip=2)", L2)
    L2.plot(ax=axes[0, 1], title="feedforward + skip")

    sys3 = simple_rnn(n_units=3, n_input=1, n_output=1)
    print(sys3.summary())
    L3 = landscape_for(sys3, T=4, max_paths=4000, eps=0.5)
    report("RNN unrolled T=4 (n_units=3)", L3)
    L3.plot(ax=axes[1, 0], title="RNN unrolled (T=4)")

    sys4 = mixture_of_experts(n_experts=3, expert_depth=3, expert_width=2)
    print(sys4.summary())
    L4 = landscape_for(sys4, T=1, eps=0.55)
    report("mixture of experts (n_experts=3, depth=3, width=2)", L4)
    L4.plot(ax=axes[1, 1], title="mixture of experts (3 experts)")

    plt.tight_layout()
    out = "demo_landscape.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved figure to {out}")


if __name__ == "__main__":
    main()
