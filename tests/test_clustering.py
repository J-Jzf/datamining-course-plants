import numpy as np
import pytest


def test_jaccard_distance_handles_two_empty_regions():
    from plantmine.clustering import region_distance
    matrix = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=bool)
    distance = region_distance(matrix)
    np.testing.assert_allclose(distance, [
        [0, 2 / 3, 1, 1], [2 / 3, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]
    ])
    cosine = region_distance(matrix, "cosine")
    assert cosine[0, 1] == pytest.approx(0.5)
    assert cosine[2, 3] == 0
    assert cosine[0, 2] == 1


def test_clustering_uses_requested_distance_and_recovers_exact_groups():
    from plantmine.clustering import cluster_regions
    matrix = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=bool)
    model = cluster_regions(matrix, k=2)
    labels = model["labels"]
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert set(labels) == {0, 1}
    assert model["silhouette"] == pytest.approx(1.0)
    assert model["linkage"].shape == (3, 4)
    with pytest.raises(ValueError):
        cluster_regions(matrix, k=2, method="ward")

