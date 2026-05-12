"""PathLandscape: the measured metric space of paths and its cluster structure.

Given a list of paths (with weights) and a similarity kernel, build the
pairwise similarity matrix and expose:

  - `cluster()`         : group paths into modes (DBSCAN on the distance matrix).
  - `cluster_summary()` : per-cluster size, total weight, representative path.
  - `embed_2d()`        : 2D MDS embedding for visualization.
  - `plot()`            : scatter of the landscape colored by cluster.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .paths import Path
from .similarity import composite_similarity, similarity_matrix


@dataclass
class ClusterSummary:
    label: int
    size: int
    total_weight: float
    representative: Path

    def __repr__(self) -> str:
        return (
            f"Cluster(label={self.label}, size={self.size}, "
            f"total_weight={self.total_weight:.3g}, rep={self.representative})"
        )


class PathLandscape:
    def __init__(
        self,
        paths: list[Path],
        kernel: Callable[[Path, Path], float] = composite_similarity,
        similarity: Optional[np.ndarray] = None,
    ) -> None:
        if not paths:
            raise ValueError("PathLandscape requires at least one path.")
        self.paths = paths
        self.kernel = kernel
        self.S = similarity if similarity is not None else similarity_matrix(paths, kernel)
        self.D = np.clip(1.0 - self.S, 0.0, None)
        np.fill_diagonal(self.D, 0.0)
        self._labels: Optional[np.ndarray] = None

    # ------------------------------------------------------------- cluster

    def cluster(
        self,
        method: str = "dbscan",
        eps: float = 0.4,
        min_samples: int = 2,
        n_clusters: Optional[int] = None,
    ) -> np.ndarray:
        """Group paths into modes.

        method='dbscan'        : density-based; eps is the max distance for
                                 neighbours, min_samples the core-point threshold.
                                 Noise points get label -1.
        method='agglomerative' : hierarchical; needs `n_clusters`.
        """
        if method == "dbscan":
            from sklearn.cluster import DBSCAN

            db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
            self._labels = db.fit_predict(self.D)
        elif method == "agglomerative":
            if n_clusters is None:
                raise ValueError("agglomerative clustering needs n_clusters.")
            from sklearn.cluster import AgglomerativeClustering

            ac = AgglomerativeClustering(
                n_clusters=n_clusters, metric="precomputed", linkage="average"
            )
            self._labels = ac.fit_predict(self.D)
        else:
            raise ValueError(f"unknown method: {method}")
        return self._labels

    @property
    def labels(self) -> np.ndarray:
        if self._labels is None:
            self.cluster()
        return self._labels

    @property
    def n_modes(self) -> int:
        return len(set(self.labels) - {-1})

    def cluster_summary(self) -> list[ClusterSummary]:
        out: list[ClusterSummary] = []
        for c in sorted(set(self.labels) - {-1}):
            idx = np.where(self.labels == c)[0]
            sub = self.S[np.ix_(idx, idx)]
            mean_sim = sub.mean(axis=1)
            rep = self.paths[idx[int(np.argmax(mean_sim))]]
            total_w = float(sum(self.paths[i].weight for i in idx))
            out.append(
                ClusterSummary(
                    label=int(c),
                    size=int(len(idx)),
                    total_weight=total_w,
                    representative=rep,
                )
            )
        return out

    # ----------------------------------------------------------- embedding

    def embed_2d(self, method: str = "mds", random_state: int = 0) -> np.ndarray:
        """2D embedding of the distance matrix for visualization."""
        if method == "mds":
            from sklearn.manifold import MDS

            mds = MDS(
                n_components=2,
                dissimilarity="precomputed",
                random_state=random_state,
                normalized_stress="auto",
                n_init=4,
            )
            return mds.fit_transform(self.D)
        if method == "tsne":
            from sklearn.manifold import TSNE

            tsne = TSNE(
                n_components=2,
                metric="precomputed",
                init="random",
                random_state=random_state,
                perplexity=min(30, max(5, len(self.paths) // 4)),
            )
            return tsne.fit_transform(self.D)
        raise ValueError(f"unknown embedding method: {method}")

    # -------------------------------------------------------------- plot

    def plot(
        self,
        ax=None,
        embedding: Optional[np.ndarray] = None,
        show_legend: bool = True,
        title: Optional[str] = None,
    ):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))
        if embedding is None:
            embedding = self.embed_2d()
        labels = self.labels
        weights = np.array([p.weight for p in self.paths])
        sizes = 25 + 75 * (weights / (weights.max() + 1e-12))
        for c in sorted(set(labels)):
            mask = labels == c
            tag = "noise" if c == -1 else f"mode {c}"
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=sizes[mask],
                alpha=0.75,
                label=tag,
                edgecolors="k",
                linewidths=0.4,
            )
        ax.set_xlabel("MDS 1")
        ax.set_ylabel("MDS 2")
        ax.set_title(title or f"path landscape — {self.n_modes} mode(s), {len(self.paths)} paths")
        if show_legend:
            ax.legend(loc="best", fontsize=8, frameon=False)
        return ax

    # ------------------------------------------------------------ summary

    def describe(self) -> str:
        labels = self.labels
        n = len(self.paths)
        n_noise = int((labels == -1).sum())
        lengths = [p.length for p in self.paths]
        return (
            f"PathLandscape(n_paths={n}, n_modes={self.n_modes}, n_noise={n_noise}, "
            f"mean_length={np.mean(lengths):.2f}, length_range=[{min(lengths)},{max(lengths)}])"
        )

    def __repr__(self) -> str:
        return self.describe()
