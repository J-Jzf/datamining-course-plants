import numpy as np
import pandas as pd
import json
import pytest

from plantmine.experiments import (split_indices, bh_adjust, spatial_permutation_test,
                                   _completion_experiment, ExperimentConfig, run_core)


def test_group_split_never_shares_a_genus_and_is_reproducible():
    genera = np.array(['abies'] * 3 + ['acer'] * 4 + ['quercus'] * 2 + ['poa'] * 3)
    train, test = split_indices(genera, seed=7, grouped=True)
    assert set(genera[train]).isdisjoint(set(genera[test]))
    assert sorted(np.r_[train, test].tolist()) == list(range(len(genera)))
    assert len(train) > 0 and len(test) > 0
    train2, test2 = split_indices(genera, seed=7, grouped=True)
    np.testing.assert_array_equal(train, train2)
    np.testing.assert_array_equal(test, test2)


def test_bh_adjust_controls_the_entire_family_and_preserves_order():
    adjusted = bh_adjust(np.array([.03, .001, .2, .01]))
    np.testing.assert_allclose(adjusted, [.04, .004, .2, .02])


def test_spatial_permutation_uses_region_labels_not_independent_pair_shuffling():
    regions = pd.DataFrame({'code': ['A', 'B', 'C', 'D'],
                            'longitude': [-100, -90, -80, -70],
                            'latitude': [40, 40, 40, 40]})
    distances = np.array([[0, .1, .3, .5], [.1, 0, .15, .35],
                          [.3, .15, 0, .2], [.5, .35, .2, 0]])
    pairs, stats = spatial_permutation_test(distances, regions, seed=1, n_permutations=49)
    assert len(pairs) == 6
    assert stats['spearman_r'] > .8
    assert 1 / 50 <= stats['permutation_p'] <= 1
    assert stats['n_permutations'] == 49


def test_record_completion_with_no_eligible_species_reports_zero_queries():
    validation = np.eye(3, dtype=bool)
    empty_rules = pd.DataFrame(columns=['antecedent', 'consequent', 'confidence', 'lift', 'support'])
    queries, summary = _completion_experiment(validation, empty_rules, ['A', 'B', 'C'],
                                               np.ones((10, 3), dtype=bool), ExperimentConfig())
    assert queries.empty
    assert 'method' in queries.columns
    assert summary.n_queries.tolist() == [0, 0, 0]
    assert summary.recall_at_5.isna().all()


def test_starting_a_new_fit_invalidates_the_previous_extension_manifest(tmp_path):
    results = tmp_path / 'outputs' / 'results'
    results.mkdir(parents=True)
    (results / 'run_summary.json').write_text(json.dumps({'extensions_complete': True}), encoding='utf-8')
    with pytest.raises(FileNotFoundError):
        run_core(tmp_path / 'missing_data', tmp_path / 'outputs')
    manifest = json.loads((results / 'run_summary.json').read_text(encoding='utf-8'))
    assert manifest['extensions_complete'] is False
    assert manifest['core_complete'] is False
