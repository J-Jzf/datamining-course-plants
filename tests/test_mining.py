import itertools

import numpy as np
import pytest


FIXTURE = np.array([
    [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
], dtype=bool)


@pytest.mark.parametrize("algorithm", ["apriori", "fpgrowth"])
def test_exact_itemsets_have_transaction_counts_not_path_counts(algorithm):
    from plantmine.mining import mine_itemsets
    actual = mine_itemsets(FIXTURE, 0.4, max_len=3, algorithm=algorithm)
    assert actual == {
        (0,): 4, (1,): 4, (2,): 4,
        (0, 1): 3, (0, 2): 3, (1, 2): 3, (0, 1, 2): 2,
    }
    assert mine_itemsets(FIXTURE, 0.61, max_len=3, algorithm=algorithm) == {
        (0,): 4, (1,): 4, (2,): 4,
    }
    assert all(len(key) <= 2 for key in mine_itemsets(FIXTURE, 0.4, 2, algorithm))


def test_algorithms_equal_independent_exhaustive_reference_on_random_data():
    from plantmine.mining import mine_itemsets
    rng = np.random.default_rng(431)
    matrix = rng.random((53, 8)) < 0.45
    threshold = 9
    expected = {}
    for size in range(1, 4):
        for items in itertools.combinations(range(matrix.shape[1]), size):
            count = int(matrix[:, items].all(axis=1).sum())
            if count >= threshold:
                expected[items] = count
    for algo in ["fpgrowth", "apriori"]:
        assert mine_itemsets(matrix, threshold / len(matrix), 3, algo) == expected


def test_rule_metrics_and_independent_holdout_evaluation():
    from plantmine.mining import build_rules, evaluate_rules, mine_itemsets
    itemsets = mine_itemsets(FIXTURE, 0.4)
    rules = build_rules(itemsets, len(FIXTURE), min_confidence=0, min_lift=0)
    selected = rules[(rules.antecedent == (0,)) & (rules.consequent == 1)]
    assert len(selected) == 1
    rule = selected.iloc[0]
    assert rule["count"] == 3
    assert rule["confidence"] == pytest.approx(0.75)
    assert rule["support"] == pytest.approx(0.6)
    assert rule["lift"] == pytest.approx(0.9375)
    assert rule["leverage"] == pytest.approx(-0.04)
    assert rule["conviction"] == pytest.approx(0.8)
    heldout = np.array([[0, 1, 1], [0, 0, 1]], dtype=bool)
    assessed = evaluate_rules(selected, heldout).iloc[0]
    assert assessed["count"] == 0
    assert assessed["antecedent_count"] == 0
    assert assessed["support"] == 0
    assert np.isnan(assessed["confidence"])
    assert assessed["antecedent"] == (0,)
    assert assessed["consequent"] == 1


def test_no_empty_or_invalid_transactions_are_silently_accepted():
    from plantmine.mining import mine_itemsets
    with pytest.raises(ValueError):
        mine_itemsets(np.zeros((0, 3), dtype=bool), 0.1)
    with pytest.raises(ValueError):
        mine_itemsets(FIXTURE, 0)
    with pytest.raises(ValueError):
        mine_itemsets(FIXTURE, 0.2, algorithm="fake")


def test_rules_keep_short_antecedents_even_when_given_longer_itemsets():
    # A longer mining run must not silently change the rule model's scope.
    from plantmine.mining import build_rules, mine_itemsets
    matrix = np.ones((4, 4), dtype=bool)
    itemsets = mine_itemsets(matrix, 0.1, max_len=4)
    rules = build_rules(itemsets, 4, min_confidence=0, min_lift=0)
    assert set(rules.antecedent.map(len)) == {1, 2}

