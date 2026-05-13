"""Extract *active* (functionally weighted) path systems from trained models.

Where `torch_bridge.py` builds a System that mirrors a model's structure,
this module captures what the model actually *does* on a batch of inputs:
edge weights are set to the empirical magnitude of signal flowing along
that edge during the forward pass, and weak edges are pruned away. The
resulting System represents the model's functional sub-graph for that
input distribution.

Two extractors:

  - `extract_mlp_flow(model, x)`  : sequential `nn.Linear` MLP with optional
                                    residual / skip blocks.
  - `extract_rnn_flow(cell, xs)`  : `nn.RNNCell` / `nn.GRUCell` style cell
                                    rolled out across a token sequence.

A helper `prune_system(sys, ...)` keeps only the strongest edges, which
makes the static path graph small enough for full path enumeration.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    raise ImportError("path_landscape.extract requires PyTorch.") from exc

from .system import System


# ----------------------------------------------------------- pruning utility

def prune_system(
    sys: System,
    top_k_in: Optional[int] = None,
    top_k_out: Optional[int] = None,
    weight_threshold: Optional[float] = None,
) -> System:
    """Return a copy of `sys` with weak edges removed.

    - `top_k_in`  : keep only the top-K incoming edges per node (by weight).
    - `top_k_out` : keep only the top-K outgoing edges per node.
    - `weight_threshold` : drop edges with weight below this absolute value.
    Combinations are intersection (an edge survives only if it passes all
    active filters).
    """
    g = sys.graph
    surviving: set[tuple[str, str, int]] = set()

    if top_k_out is not None:
        for u in g.nodes:
            edges = sorted(
                ((v, k, d.get("weight", 1.0))
                 for v, k, d in [(v, k, d)
                                 for v, ddict in g._adj[u].items()
                                 for k, d in ddict.items()]),
                key=lambda t: -t[2],
            )
            for v, k, _ in edges[:top_k_out]:
                surviving.add((u, v, k))
    if top_k_in is not None:
        for v in g.nodes:
            edges = []
            for u, ddict in g._pred[v].items():
                for k, d in ddict.items():
                    edges.append((u, k, d.get("weight", 1.0)))
            edges.sort(key=lambda t: -t[2])
            kept = {(u, k) for u, k, _ in edges[:top_k_in]}
            for u, k, _ in edges:
                if (u, k) in kept and (top_k_out is None or (u, v, k) in surviving):
                    surviving.add((u, v, k))
                elif top_k_out is None and (u, k) in kept:
                    surviving.add((u, v, k))
    if top_k_in is None and top_k_out is None:
        for u, v, k in g.edges(keys=True):
            surviving.add((u, v, k))

    if weight_threshold is not None:
        surviving = {
            (u, v, k) for (u, v, k) in surviving
            if g[u][v][k].get("weight", 1.0) >= weight_threshold
        }

    new = System()
    for n, u in sys.units.items():
        new.add_unit(n, scale=u.scale, parent=u.parent, op=u.op)
    for n in sys.inputs:
        new.set_input(n)
    for n in sys.outputs:
        new.set_output(n)
    for u, v, k in surviving:
        d = g[u][v][k]
        new.add_edge(u, v, weight=float(d.get("weight", 1.0)),
                     recurrent=bool(d.get("recurrent", False)))
    return new


# ----------------------------------------------------- MLP active-flow graph

def extract_mlp_flow(
    model: nn.Module,
    x: torch.Tensor,
    prefix: str = "L",
    aggregate: str = "mean",
) -> System:
    """Build a System where edges are weighted by the actual magnitude of
    activation flow along that edge: edge_weight = aggregate over batch of
    |W[j, i] * a_in[i]|.

    Detects `nn.Linear` modules in forward order via hooks (so this picks up
    skip connections / residual additions correctly when they're routed
    through Linears).
    """
    layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not layers:
        raise ValueError("no nn.Linear modules in model")

    activations: list[torch.Tensor] = []

    def make_hook():
        def hook(_mod, inputs, _output):
            activations.append(inputs[0].detach())
        return hook

    handles = [layer.register_forward_hook(make_hook()) for layer in layers]
    try:
        model.eval()
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()

    sys = System()
    in0 = layers[0].in_features
    for k in range(in0):
        sys.add_unit(f"{prefix}0_n{k}", scale=0)
    sys.set_input(*[f"{prefix}0_n{k}" for k in range(in0)])
    prev = in0
    for li, (layer, a_in) in enumerate(zip(layers, activations), start=1):
        out = layer.out_features
        for k in range(out):
            sys.add_unit(f"{prefix}{li}_n{k}", scale=0)
        W = layer.weight.detach().cpu().abs().numpy()              # (out, in)
        a = a_in.detach().cpu().abs().numpy().reshape(-1, prev)
        if aggregate == "mean":
            scale = a.mean(axis=0)
        elif aggregate == "max":
            scale = a.max(axis=0)
        else:
            raise ValueError(f"unknown aggregate: {aggregate}")
        flow = W * scale[None, :]                                   # (out, in)
        for j in range(out):
            for i in range(prev):
                w = float(flow[j, i])
                if w > 0:
                    sys.add_edge(
                        f"{prefix}{li - 1}_n{i}",
                        f"{prefix}{li}_n{j}",
                        weight=w,
                    )
        prev = out
    sys.set_output(*[f"{prefix}{len(layers)}_n{k}"
                    for k in range(layers[-1].out_features)])
    return sys


# ----------------------------------------------------- RNN active-flow graph

def extract_rnn_flow(
    cell: nn.Module,
    inputs: torch.Tensor,
    h0: Optional[torch.Tensor] = None,
    in_prefix: str = "in",
    h_prefix: str = "h",
    aggregate: str = "mean",
) -> System:
    """Roll out an RNN cell (with `weight_ih`, `weight_hh`) across a token
    sequence and build a System in which:

      - Input units are tagged 'in{k}' (no time index, lifted to t=0 by
        unroll).
      - Hidden units are 'h{k}' with recurrent edges between them
        (materialized by `unroll(T)`).
      - Hidden-to-hidden edge weight    = mean over (batch x time) of
                                          |W_hh * h_prev|.
      - Input-to-hidden edge weight     = mean over (batch x time) of
                                          |W_ih * x_t|.

    `inputs` shape: (B, T, I).  If `h0` is None, zeros are used.
    Sources for path enumeration are the input units; sinks are the hidden
    units at time T-1 (consult `unroll(T)`).
    """
    if not (hasattr(cell, "weight_ih") and hasattr(cell, "weight_hh")):
        raise TypeError("cell must expose weight_ih and weight_hh")
    # GRUCell / LSTMCell concatenate gates along axis 0:
    # weight_ih: (n_gates * H, I)   weight_hh: (n_gates * H, H)
    H = getattr(cell, "hidden_size", cell.weight_hh.shape[1])
    I = getattr(cell, "input_size", cell.weight_ih.shape[1])
    n_gates = cell.weight_ih.shape[0] // H
    W_ih = cell.weight_ih.detach().abs().view(n_gates, H, I).sum(dim=0)  # (H,I)
    W_hh = cell.weight_hh.detach().abs().view(n_gates, H, H).sum(dim=0)  # (H,H)
    B, T, _ = inputs.shape
    if h0 is None:
        h0 = torch.zeros(B, H, dtype=inputs.dtype, device=inputs.device)

    # roll-out and accumulate per-edge magnitudes
    cell.eval()
    h = h0
    flow_ih = torch.zeros_like(W_ih)
    flow_hh = torch.zeros_like(W_hh)
    n_steps = 0
    with torch.no_grad():
        for t in range(T):
            x_t = inputs[:, t, :]
            # contributions: |W_ih| * mean |x_t|, |W_hh| * mean |h|
            x_abs_mean = x_t.detach().abs().mean(dim=0)            # (I,)
            h_abs_mean = h.detach().abs().mean(dim=0)              # (H,)
            flow_ih += W_ih * x_abs_mean[None, :]
            flow_hh += W_hh * h_abs_mean[None, :]
            n_steps += 1
            # advance hidden state
            try:
                h = cell(x_t, h)
            except Exception:
                h = cell(x_t)
    if aggregate == "mean":
        flow_ih /= max(1, n_steps)
        flow_hh /= max(1, n_steps)
    elif aggregate == "sum":
        pass
    else:
        raise ValueError(f"unknown aggregate: {aggregate}")

    sys = System()
    for k in range(I):
        sys.add_unit(f"{in_prefix}{k}", scale=0)
    for k in range(H):
        sys.add_unit(f"{h_prefix}{k}", scale=0)
    sys.set_input(*[f"{in_prefix}{k}" for k in range(I)])
    sys.set_output(*[f"{h_prefix}{k}" for k in range(H)])

    fih = flow_ih.cpu().numpy()
    fhh = flow_hh.cpu().numpy()
    for j in range(H):
        for i in range(I):
            w = float(fih[j, i])
            if w > 0:
                sys.add_edge(f"{in_prefix}{i}", f"{h_prefix}{j}", weight=w)
    for j in range(H):
        for i in range(H):
            w = float(fhh[j, i])
            if w > 0:
                sys.add_edge(f"{h_prefix}{i}", f"{h_prefix}{j}",
                             weight=w, recurrent=True)
    return sys
