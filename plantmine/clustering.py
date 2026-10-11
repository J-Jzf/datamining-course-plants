"""Cluster region floristic compositions with metric-consistent evaluation."""

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score


def region_distance(matrix, metric="jaccard"):
    """Return finite region distances; two empty regions have distance zero."""
    matrix = np.asarray(matrix, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Expected a non-empty plants-by-regions matrix")
    numeric = matrix.astype(np.float64)
    intersection = numeric.T @ numeric
    sizes = np.diag(intersection)
    if metric == "jaccard":
        denominator = sizes[:, None] + sizes[None, :] - intersection
    elif metric == "cosine":
        denominator = np.sqrt(sizes[:, None] * sizes[None, :])
    else:
        raise ValueError("metric must be 'jaccard' or 'cosine'")
    similarity = np.divide(intersection, denominator, out=np.zeros_like(intersection), where=denominator > 0)
    both_empty = (sizes[:, None] == 0) & (sizes[None, :] == 0)
    similarity[both_empty] = 1.0
    distance = np.clip(1.0 - similarity, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    return (distance + distance.T) / 2.0


def cluster_regions(matrix, k, metric="jaccard", method="average"):
    """Hierarchical clustering with a requested exact cluster count.

    Average/complete linkage accept Jaccard or cosine distances. Ward is not
    offered because its Euclidean variance objective is unsuitable here.
    """
    distance = region_distance(matrix, metric)
    n_regions = distance.shape[0]
    if not isinstance(k, (int, np.integer)) or not 2 <= k < n_regions:
        raise ValueError("k must be an integer from 2 to n_regions - 1")
    if method not in {"average", "complete"}:
        raise ValueError("method must be 'average' or 'complete'")
    tree = linkage(squareform(distance, checks=False), method=method, optimal_ordering=True)
    labels = cut_tree(tree, n_clusters=[int(k)]).reshape(-1).astype(int)
    silhouette = float(silhouette_score(distance, labels, metric="precomputed"))
    return {"distance": distance, "linkage": tree, "labels": labels,
            "silhouette": silhouette, "metric": metric, "method": method, "k": int(k)}

