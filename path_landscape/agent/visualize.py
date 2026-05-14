"""Multi-panel visualization for the agent pipeline:
  1) original-system graph (with feedback/multiscale rendered),
  2) static path graph (after unrolling/coarsening),
  3) path-landscape MDS scatter colored by cluster,
  4) length-by-cluster bar plot,
  5) persistence diagram (H0 + H1).
"""
from __future__ import annotations

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ..system import System
from ..landscape import PathLandscape
from ..metrics import persistence_h0, persistence_h1
from .schemas import SystemSpec

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="sklearn.manifold._mds"
)
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="sklearn.manifold._mds"
)

INK = "#0c1220"
PAPER = "#fcfcfa"
AMBER = "#f0b65a"
CYAN = "#7dc8dc"
BRICK = "#c83c37"
GRAY = "#aab4c3"


# ----------------------------------------------------------------- helpers

def _node_color(role: str) -> str:
    return {"input": CYAN, "output": BRICK}.get(role, AMBER)


def _draw_system_graph(ax, sys: System, spec: SystemSpec) -> None:
    g = nx.DiGraph()
    for name in sys.units:
        g.add_node(name)
    for u, v, data in sys.graph.edges(data=True):
        g.add_edge(u, v, recurrent=data.get("recurrent", False))
    # role lookup
    roles = {u.name: u.role for u in spec.units}
    parents = {u.name: (u.parent or "_") for u in spec.units}

    # group nodes by parent for the layout
    try:
        pos = nx.spring_layout(g, seed=0, k=1.0 / max(1, len(g) ** 0.5))
    except Exception:
        pos = {n: (i, 0) for i, n in enumerate(g)}

    # edges
    rec_edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("recurrent")]
    fwd_edges = [
        (u, v) for u, v, d in g.edges(data=True) if not d.get("recurrent")
    ]
    nx.draw_networkx_edges(
        g, pos, edgelist=fwd_edges, ax=ax, edge_color=INK,
        alpha=0.45, width=0.7, arrows=True, arrowsize=8,
    )
    nx.draw_networkx_edges(
        g, pos, edgelist=rec_edges, ax=ax, edge_color=BRICK,
        alpha=0.7, width=1.0, arrows=True, arrowsize=8,
        style="dashed",
    )
    # nodes
    node_colors = [_node_color(roles.get(n, "internal")) for n in g.nodes]
    # encode multiscale via node size — units with a parent are smaller
    sizes = [(180 if parents.get(n) == "_" else 110) for n in g.nodes]
    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_color=node_colors, node_size=sizes,
        edgecolors=INK, linewidths=0.6,
    )
    # short label
    labels = {n: (n if len(n) <= 8 else n[:7] + "…") for n in g.nodes}
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax,
                            font_size=6, font_color=INK)
    ax.set_title(
        f"{spec.phenomenon_name} - system graph\n"
        f"({len(sys.units)} units; cyan=input, red=output; "
        f"dashed red = recurrent)",
        fontsize=10,
    )
    ax.set_axis_off()


def _draw_static_path_graph(ax, sys: System, T: int) -> None:
    g = sys.unroll(T)
    # too many nodes? thin out the labels
    n = g.number_of_nodes()
    pos = nx.kamada_kawai_layout(g) if n <= 200 else nx.spring_layout(g, seed=0)
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=INK, alpha=0.25, width=0.4,
                           arrows=False)
    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_color=GRAY, node_size=14,
        edgecolors=INK, linewidths=0.3,
    )
    if n <= 50:
        nx.draw_networkx_labels(g, pos, ax=ax, font_size=5, font_color=INK)
    ax.set_title(
        f"static path graph after unroll(T={T}) - "
        f"{n} nodes, {g.number_of_edges()} edges",
        fontsize=10,
    )
    ax.set_axis_off()


def _draw_persistence(ax, L: PathLandscape) -> None:
    h0 = persistence_h0(L, max_features=30)
    h1 = persistence_h1(L)
    if h0:
        for h in h0[:25]:
            ax.plot([0.0, h], [h, h], color=INK, linewidth=0.7, alpha=0.55)
        ax.scatter(
            [0.0] * len(h0[:25]), h0[:25],
            color=CYAN, s=18, zorder=4,
            label=f"H0 (top {min(25, len(h0))} of {len(h0)})",
        )
    if h1:
        births = [b for b, _ in h1]
        deaths = [d for _, d in h1]
        ax.scatter(
            births, deaths, color=BRICK, s=24, zorder=5,
            edgecolors=INK, linewidths=0.4, label=f"H1 ({len(h1)})",
        )
    lim = 1.0
    if h0:
        lim = max(lim, max(h0))
    if h1:
        lim = max(lim, max((d for _, d in h1), default=0))
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.4, alpha=0.4)
    ax.set_xlim(0, lim * 1.05)
    ax.set_ylim(0, lim * 1.05)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title("persistence diagram", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)


# ----------------------------------------------------------------- public

def render_figure(
    spec: SystemSpec, sys: System, L: PathLandscape, out_path: str
) -> None:
    """Render the multi-panel analysis figure and save to `out_path`."""
    fig = plt.figure(figsize=(15, 12), facecolor=PAPER)
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1.1, 1.0, 1.0],
        hspace=0.45, wspace=0.25,
        top=0.94, bottom=0.05, left=0.06, right=0.97,
    )

    ax_sys = fig.add_subplot(gs[0, 0])
    _draw_system_graph(ax_sys, sys, spec)

    ax_static = fig.add_subplot(gs[0, 1])
    _draw_static_path_graph(ax_static, sys, spec.time_steps)

    ax_land = fig.add_subplot(gs[1, 0])
    L.plot(ax=ax_land, show_legend=False,
           title=f"path landscape ({L.n_modes} modes, "
                 f"{len(L.paths)} paths)")

    ax_bars = fig.add_subplot(gs[1, 1])
    L.plot_length_by_cluster(
        ax=ax_bars, drop_noise=True, label_top_k=8,
        title="paths by cluster, length on y-axis",
    )

    ax_pers = fig.add_subplot(gs[2, 0])
    _draw_persistence(ax_pers, L)

    # banner with the phenomenon summary on the bottom-right
    ax_text = fig.add_subplot(gs[2, 1])
    ax_text.axis("off")
    ax_text.text(
        0.0, 1.0, spec.phenomenon_name,
        ha="left", va="top",
        fontsize=14, fontweight="bold", color=INK,
    )
    ax_text.text(
        0.0, 0.85, spec.phenomenon_summary,
        ha="left", va="top",
        fontsize=10, color=INK, wrap=True,
    )
    bullets = []
    lengths = [p.length for p in L.paths]
    bullets.append(
        f"path length range: {min(lengths)} - {max(lengths)} "
        f"(mean {np.mean(lengths):.2f})"
    )
    bullets.append(f"clusters: {L.n_modes} modes")
    h1 = persistence_h1(L)
    if h1 is not None:
        bullets.append(f"H1 features (compositional loops): {len(h1)}")
    bullets.append(
        f"system: {len(sys.units)} units, {sys.graph.number_of_edges()} edges"
    )
    for i, line in enumerate(bullets):
        ax_text.text(
            0.0, 0.55 - i * 0.10, "• " + line,
            ha="left", va="top", fontsize=9, color=INK,
        )

    plt.savefig(out_path, dpi=150)
    plt.close(fig)
