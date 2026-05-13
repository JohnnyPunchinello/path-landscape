"""Serious side-by-side comparison: tiny attention-based LM vs. Dale's-law
recurrent circuit, both trained on the same next-token-prediction task.

Pipeline
--------
1. Build a synthetic sequence task: 16-symbol alphabet, sequences generated
   by a 4-state HMM + emission noise. Predict the next symbol given a
   context window.
2. Train an "LM-like" model: a deep MLP (16 layers w/ skip connections)
   over an embedded context window. Stand-in for a transformer's residual
   stream, with the path-extraction story preserved.
3. Train a "circuit-like" model: a GRU cell whose recurrent matrix is
   constrained by Dale's law (each hidden unit is purely excitatory or
   purely inhibitory) and is sparsely connected. Trained on the same task.
4. For each trained model, run inference on a test batch and *extract the
   active flow graph* — a System whose edge weights are the empirical
   magnitude of signal on that edge during the forward pass. Prune to the
   strongest edges per node so paths are interpretable.
5. Sample paths through the active sub-graph, build a `PathLandscape`,
   compute the four cross-system metrics, and visualize with both the
   classic landscape plot and the new length-by-cluster plot.

Run:  python demo_compare_serious.py
Saves: demo_compare_serious.png  (and prints comparison table)
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from path_landscape import (
    PathLandscape,
    sample_paths,
    compare,
    format_comparison,
)
from path_landscape.extract import (
    extract_mlp_flow,
    extract_rnn_flow,
    prune_system,
)

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.manifold._mds")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._mds")
torch.manual_seed(0)
np.random.seed(0)


# ============================================================ task

class ModularAdditionTask:
    """Predict y = (x[ctx-1] + x[ctx-2]) mod vocab, given a context window.

    Distractors at the earlier positions force the model to learn that only
    the last two tokens matter — both architectures should learn this, and
    the resulting active path graph should reflect *which* positions the
    model actually uses.
    """
    def __init__(self, vocab=8, ctx=6, seed=0):
        self.vocab = vocab
        self.ctx = ctx
        self.rng = np.random.default_rng(seed)

    def make_dataset(self, n: int):
        rng = self.rng
        X = rng.integers(0, self.vocab, size=(n, self.ctx)).astype(np.int64)
        y = (X[:, -1] + X[:, -2]) % self.vocab
        return torch.from_numpy(X), torch.from_numpy(y.astype(np.int64))


# ============================================================ models

class TinyLM(nn.Module):
    """Small attention-style model: per-position embedding -> per-position
    Linear projection -> sum-pool across positions -> 2-layer MLP with skip
    -> readout.

    The per-position Linear acts as a 'value head'; the sum-pool is the
    simplest associative readout. Both blocks are sequences of nn.Linear,
    so `extract_mlp_flow` picks them up.
    """
    def __init__(self, vocab=8, ctx=6, d=16, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.ctx = ctx
        self.d = d
        # value head per position (one Linear over the flattened context).
        self.value = nn.Linear(ctx * d, hidden)
        # mixer block with skip
        self.mix1 = nn.Linear(hidden, hidden)
        self.mix2 = nn.Linear(hidden, hidden)
        self.skip = nn.Linear(hidden, hidden)
        self.readout = nn.Linear(hidden, vocab)

    def forward(self, x):
        h0 = self.embed(x).reshape(x.shape[0], -1)
        h = F.relu(self.value(h0))
        m = F.relu(self.mix1(h))
        m = self.mix2(m) + self.skip(h)
        m = F.relu(m)
        return self.readout(m)


class DalesLawGRU(nn.Module):
    """GRU cell whose recurrent matrix is masked into Dale's-law columns
    (each presynaptic unit is purely excitatory or purely inhibitory) and
    has a fixed sparse connectivity pattern.

    A small linear readout maps the final hidden state to logits.
    """
    def __init__(self, vocab=16, hidden=24, frac_excitatory=0.8, sparsity=0.4,
                 seed=0):
        super().__init__()
        self.embed = nn.Embedding(vocab, 8)
        self.cell = nn.GRUCell(8, hidden)
        self.readout = nn.Linear(hidden, vocab)
        self.hidden = hidden

        rng = np.random.default_rng(seed)
        # excitatory / inhibitory sign per presynaptic unit (column of W_hh)
        n_e = int(round(hidden * frac_excitatory))
        signs = np.array([1.0] * n_e + [-1.0] * (hidden - n_e))
        rng.shuffle(signs)
        self.register_buffer(
            "ei_sign",
            torch.tensor(signs, dtype=torch.float32).view(1, hidden),
        )
        # sparse connectivity mask: keep `sparsity` fraction of entries.
        mask = (rng.uniform(size=(3 * hidden, hidden)) < sparsity).astype(
            np.float32
        )
        self.register_buffer("rec_mask", torch.tensor(mask))

    def _project_recurrent_weights(self):
        """Apply Dale's-law sign and sparsity to weight_hh in-place. Each
        column shares the sign of its presynaptic unit; entries are
        |W| * sign * mask.
        """
        with torch.no_grad():
            W = self.cell.weight_hh.data
            absW = W.abs()
            # broadcast sign across the 3 GRU gates
            sign_block = self.ei_sign.expand(3 * self.hidden, self.hidden)
            self.cell.weight_hh.data = absW * sign_block * self.rec_mask

    def forward(self, x):
        # x: (B, T)
        B, T = x.shape
        h = torch.zeros(B, self.hidden, device=x.device)
        e = self.embed(x)                        # (B, T, 8)
        for t in range(T):
            h = self.cell(e[:, t, :], h)
        return self.readout(h)


# ============================================================ training

def train(model, X, y, epochs=300, lr=5e-3, dales=False, log_every=50):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        opt.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        if dales:
            model._project_recurrent_weights()
        if (ep + 1) % log_every == 0:
            with torch.no_grad():
                acc = (logits.argmax(dim=1) == y).float().mean().item()
                print(f"  epoch {ep+1:>4d}  loss={loss.item():.3f}  acc={acc:.3f}")
    with torch.no_grad():
        acc = (model(X).argmax(dim=1) == y).float().mean().item()
    return acc


# ============================================================ main

def main():
    print("=" * 64)
    print("Building task and models...")
    task = ModularAdditionTask(vocab=8, ctx=6, seed=1)
    X_train, y_train = task.make_dataset(8000)
    X_test,  y_test  = task.make_dataset(2000)
    print(f"  task: predict (x[-1] + x[-2]) mod {task.vocab};  ctx={task.ctx}")

    print("\nTraining LM-like (per-position value head + 2-layer mixer + skip)...")
    lm = TinyLM(vocab=task.vocab, ctx=task.ctx, d=16, hidden=32)
    lm_acc = train(lm, X_train, y_train, epochs=600, lr=3e-3)
    test_acc_lm = (lm(X_test).argmax(dim=1) == y_test).float().mean().item()
    print(f"  LM      :  train acc {lm_acc:.3f}  |  test acc {test_acc_lm:.3f}")

    print("\nTraining circuit-like (Dale's-law GRU, hidden=24, 80% E, 50% sparse)...")
    circ = DalesLawGRU(vocab=task.vocab, hidden=24, frac_excitatory=0.8,
                       sparsity=0.5, seed=2)
    circ._project_recurrent_weights()
    circ_acc = train(circ, X_train, y_train, epochs=800, lr=5e-3, dales=True)
    test_acc_circ = (circ(X_test).argmax(dim=1) == y_test).float().mean().item()
    print(f"  circuit :  train acc {circ_acc:.3f}  |  test acc {test_acc_circ:.3f}")

    # ----------------------------------------------------- extract paths
    print("\n" + "=" * 64)
    print("Extracting active path graphs from forward passes on the test set...")
    # LM: run a forward pass and build flow graph from Linear hooks.
    # Use a small batch to keep extraction fast.
    sub = X_test[:128]
    sys_lm = extract_mlp_flow(lm, sub.long())
    sys_lm = prune_system(sys_lm, top_k_out=3)   # keep the 3 strongest
                                                  # outgoing edges per node
    print(f"  LM      :  {sys_lm.summary()}")

    # Circuit: roll out the GRU cell and build its flow graph.
    embedded = circ.embed(sub)                   # (B, T, 8)
    sys_circ = extract_rnn_flow(circ.cell, embedded.detach())
    sys_circ = prune_system(sys_circ, top_k_out=3)
    print(f"  circuit :  {sys_circ.summary()}")

    # ----------------------------------------------------- landscapes
    print("\nBuilding path landscapes (this can take ~30s)...")
    g_lm = sys_lm.unroll(T=1)
    paths_lm = sample_paths(
        g_lm, sys_lm.unroll_sources(T=1), sys_lm.unroll_sinks(T=1),
        n_samples=900, max_length=32,
    )
    L_lm = PathLandscape(paths_lm)
    L_lm.cluster(eps=0.45, min_samples=3)
    print(f"  LM      :  {L_lm.describe()}")

    g_c = sys_circ.unroll(T=task.ctx)
    paths_c = sample_paths(
        g_c, sys_circ.unroll_sources(T=task.ctx),
        sys_circ.unroll_sinks(T=task.ctx),
        n_samples=900, max_length=32,
    )
    L_c = PathLandscape(paths_c)
    L_c.cluster(eps=0.50, min_samples=3)
    print(f"  circuit :  {L_c.describe()}")

    # ----------------------------------------------------- compare
    print("\nLandscape comparison")
    print("=" * 64)
    cmp = compare(L_lm, L_c, names=("LM-like", "circuit-like"))
    print(format_comparison(cmp))
    print("=" * 64)

    # ----------------------------------------------------- visualize
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1.0, 1.0, 1.05],
        hspace=0.55, wspace=0.22,
        top=0.95, bottom=0.05, left=0.06, right=0.97,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    L_lm.plot(ax=ax_a, show_legend=False,
              title=f"LM-like - landscape ({L_lm.n_modes} modes)")
    L_c.plot(ax=ax_b, show_legend=False,
             title=f"circuit-like - landscape ({L_c.n_modes} modes)")

    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    L_lm.plot_length_by_cluster(
        ax=ax_c, drop_noise=True, label_top_k=8,
        title="LM-like - paths grouped by cluster (noise hidden)",
    )
    L_c.plot_length_by_cluster(
        ax=ax_d, drop_noise=True, label_top_k=8,
        title="circuit-like - paths grouped by cluster (noise hidden)",
    )

    # comparison table as a text panel
    ax_e = fig.add_subplot(gs[2, :])
    ax_e.axis("off")
    table_text = format_comparison(cmp)
    ax_e.text(
        0.5, 0.95, "Comparison metrics",
        ha="center", va="top", fontsize=12, fontweight="bold",
    )
    ax_e.text(
        0.5, 0.83, table_text,
        ha="center", va="top", family="monospace", fontsize=10,
    )

    out = "demo_compare_serious.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved figure to {out}")


if __name__ == "__main__":
    main()
