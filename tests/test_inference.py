import pandas as pd
import pytest

from plantmine.inference import rank_regions


def test_two_region_antecedent_requires_both_regions_and_excludes_observed():
    rules = pd.DataFrame([
        {'antecedent': 'CA|OR', 'consequent': 'WA', 'confidence': .95, 'lift': 2., 'support': .1},
        {'antecedent': 'CA', 'consequent': 'NV', 'confidence': .8, 'lift': 1.5, 'support': .2},
        {'antecedent': 'CA', 'consequent': 'OR', 'confidence': .7, 'lift': 1.3, 'support': .3},
    ])
    one = rank_regions(['CA'], rules, ['CA', 'OR', 'WA', 'NV'])
    assert one['region'].tolist() == ['NV', 'OR']
    two = rank_regions(['CA', 'OR'], rules, ['CA', 'OR', 'WA', 'NV'])
    assert two['region'].tolist() == ['WA', 'NV']
    assert two.iloc[0]['score'] == .95
    assert two.iloc[0]['matched_antecedent'] == 'CA|OR'


def test_unknown_region_is_rejected_instead_of_silently_dropped():
    rules = pd.DataFrame(columns=['antecedent', 'consequent', 'confidence', 'lift', 'support'])
    with pytest.raises(ValueError, match='XX'):
        rank_regions(['XX'], rules, ['CA'])
