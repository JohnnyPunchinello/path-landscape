# path-landscape

Build the **path representation** of any information-processing system — biological circuits, recurrent networks, trained neural nets — and study its landscape of computational modes.

This is a working code companion to the thesis *The Shape of Emergence*, which proposes that the behavior of an information-processing system is shaped by **the routes information takes as it flows through it**. The package gives you the machinery to:

1. Specify a system: units, edges, inputs, outputs, recurrent edges, and a multiscale (hierarchical) structure.
2. Convert it into a **static path graph** by absorbing time (unrolling recurrence) and scale (coarsening).
3. Enumerate or sample paths from input to output.
4. Compare paths with a similarity kernel and cluster them — the **path landscape** is the resulting measured metric space.
5. Read off how many distinct "modes of computation" the system has, and how they are distributed.

---

## Install

```bash
git clone https://github.com/JohnnyPunchinello/path-landscape.git
cd path-landscape
pip install -r requirements.txt
```

`torch` is optional and only needed for the PyTorch bridge (`path_landscape.torch_bridge`) and `demo_torch.py`.

---

## Quick start — 12 lines

```python
from path_landscape import PathLandscape, enumerate_paths
from path_landscape.examples import mixture_of_experts

sys = mixture_of_experts(n_experts=3, expert_depth=3, expert_width=2)
g   = sys.unroll(T=1)                                    # static path graph
ps  = enumerate_paths(g, sys.unroll_sources(), sys.unroll_sinks())

L = PathLandscape(ps)
L.cluster(eps=0.55)
print(L.describe())                  # -> PathLandscape(n_paths=24, n_modes=3, ...)
for c in L.cluster_summary():
    print(c)                         # 3 clusters of 8 paths each, one per expert
```

Run the bundled demos:

```bash
python demo.py          # 4 structural examples; saves demo_landscape.png
python demo_torch.py    # trains a small MLP, plots structural vs functional landscapes
```

---

## Concepts

### System: units, edges, recurrence, scale

A `System` is a directed multigraph of computing units. Each `Unit` has a name, an integer `scale` level, an optional `parent` (its container one level up), and an optional `op` (the function it computes). Edges may be marked `recurrent=True`, meaning they only fire after one time step.

```python
from path_landscape import System
sys = System()
sys.add_unit("x")                                 # scalar input
sys.add_unit("h0", parent="hidden")               # hidden unit, in module "hidden"
sys.add_unit("h1", parent="hidden")
sys.add_unit("y")                                 # output
sys.add_edge("x", "h0")
sys.add_edge("x", "h1")
sys.add_edge("h0", "h1", recurrent=True)          # feedback loop
sys.add_edge("h0", "y")
sys.add_edge("h1", "y")
sys.set_input("x"); sys.set_output("y")
```

### Two derived static graphs

- **`sys.unroll(T)`** — time-unrolled DAG. Each unit `u` becomes `u@0, u@1, ..., u@(T-1)`. Non-recurrent edges stay within one step; recurrent edges become forward edges from `u@t` to `v@(t+1)`. With `T=1`, recurrent edges have nowhere to go and are dropped. Time disappears into the structure.
- **`sys.coarsen()`** — the next-scale-up `System`. Each unit collapses into its `parent`; within-parent edges are dropped; cross-parent edges are aggregated. Inputs/outputs lift to their parents. Coarsen repeatedly to climb the hierarchy.

### Paths

```python
from path_landscape import enumerate_paths, sample_paths

g = sys.unroll(T=4)
paths = enumerate_paths(g, sys.unroll_sources(T=4), sys.unroll_sinks(T=4),
                        max_paths=5000)
# or, for large/dense graphs:
paths = sample_paths(g, sys.unroll_sources(T=4), sys.unroll_sinks(T=4),
                     n_samples=2000)
```

Each `Path` carries an ordered tuple of node names and a `weight` (product of edge weights traversed).

### Similarity kernels

The default `composite_similarity` blends two intuitive signals:

- **edge-Jaccard**: what components a path uses.
- **ordered overlap (LCS)**: the order it uses them in.

Both lie in `[0, 1]`. The composite is `alpha * edge_jaccard + (1 - alpha) * ordered_overlap`, default `alpha=0.5`. Available kernels:

| kernel | what it captures |
| --- | --- |
| `jaccard_nodes` | shared nodes, no order |
| `jaccard_edges` | shared transitions, no order |
| `ordered_overlap` | longest common subsequence, normalized |
| `composite_similarity` | edges + order (default) |

Plug in your own with `PathLandscape(paths, kernel=my_kernel)`.

### Path landscape

```python
from path_landscape import PathLandscape

L = PathLandscape(paths)
L.cluster(eps=0.45, min_samples=2)         # DBSCAN on the distance matrix
L.n_modes                                  # number of clusters
L.cluster_summary()                        # size, total weight, representative path per cluster
L.embed_2d()                               # MDS coordinates
L.plot()                                   # scatter colored by cluster
```

Clusters are the **modes** of the landscape — the system's distinct ways of computing. A system with one wide cluster has one characteristic mode of computation. A system with multiple separated clusters has structurally distinct routes.

---

## Feedback loops and multiscale structure

### Feedback loops

Mark recurrent edges with `recurrent=True`. `unroll(T)` materializes a finite static DAG with one node per `(unit, time)` pair and forward edges across time steps.

```python
from path_landscape.examples import simple_rnn
sys = simple_rnn(n_units=3)                 # all-to-all recurrence + self-loops
g = sys.unroll(T=4)                         # static DAG over 4 time steps
# paths now traverse (input@0 -> h*@0 -> h*@1 -> ... -> output@3).
```

Time scales are **inputs** to the path-construction procedure: `T` controls how many recurrent unrollings are kept. Once you have the static graph, time has disappeared into the structure.

To convert an `nn.RNNCell` into a `System` directly:

```python
import torch.nn as nn
from path_landscape.torch_bridge import rnncell_to_system

cell = nn.RNNCell(input_size=8, hidden_size=16)
sys  = rnncell_to_system(cell)
g    = sys.unroll(T=10)                     # 10-step unrolled DAG, ready for paths
```

### Multiscale (hierarchical) structure

Give each unit a `parent` to declare which higher-scale entity contains it. `sys.coarsen()` returns the next-scale-up system; iterate to climb the hierarchy.

```python
from path_landscape.examples import hierarchical_mlp
sys   = hierarchical_mlp(n_modules=3, units_per_module=3, depth=3)
print(sys.summary())                        # micro: 27 units, dense
print(sys.coarsen().summary())              # macro: 3 modules, sparse cross-edges
```

Compute the landscape at the micro scale to see fine-grained modes; coarsen first to see how modes look at the modular level. Scale, like time, is an input to the procedure but absorbed into the static graph.

---

## PyTorch bridge — structural vs functional landscapes

The `torch_bridge` module turns a trained PyTorch model into a `System`. You can then re-weight edges by their **functional** importance and watch the landscape change:

```python
import torch.nn as nn
from path_landscape import PathLandscape, sample_paths
from path_landscape.torch_bridge import (
    mlp_to_system, reweight_by_pathways, reweight_by_grad, train_toy_mlp,
)

model, X, y = train_toy_mlp(hidden=[8, 8], n_classes=3)
sys = mlp_to_system(model)                  # edges weighted by |W|
reweight_by_pathways(sys, model, X)         # now edges weighted by mean |W * a_in| on X

g = sys.unroll(T=1)
paths = sample_paths(g, sys.unroll_sources(), sys.unroll_sinks(), n_samples=500)
L = PathLandscape(paths)                    # path weight = product of edge weights
print(L.cluster(eps=0.55), L.describe())
```

Three re-weightings are available:

| function | edge weight |
| --- | --- |
| `reweight_by_weight(sys, model)`  | `|W|` |
| `reweight_by_grad(sys, model, x, ...)`  | `|dL/dW|` on input `x` |
| `reweight_by_pathways(sys, model, x)`  | mean `|W * a_in|` on input `x` |

The structural landscape (`|W|`) reflects what the network *could* compute; the functional landscape reflects what it *does* compute under a given input distribution.

---

## What's in the package

```
path_landscape/
  system.py        # System, Unit, unroll(T), coarsen()
  paths.py         # Path, enumerate_paths, sample_paths
  similarity.py    # jaccard_*, ordered_overlap, composite_similarity
  landscape.py     # PathLandscape: cluster, summary, embed_2d, plot
  examples.py      # feedforward_chain, feedforward_with_skip, simple_rnn,
                   # mixture_of_experts, hierarchical_mlp, two_module_network
  torch_bridge.py  # mlp_to_system, rnncell_to_system, reweight_*, train_toy_mlp
demo.py            # structural examples; saves demo_landscape.png
demo_torch.py      # trained-MLP landscape; saves demo_torch_landscape.png
```

---

## License

MIT. See `LICENSE`.

## Citation

If you use this code in academic work, please cite the thesis:

> Johnny Jingze Li, *The Shape of Emergence: Paths in the Phenome Category*, PhD thesis, University of California San Diego, 2026.
