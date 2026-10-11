"""Inference from discovered distribution rules; scores are conditional frequencies."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


PREDICTION_COLUMNS = ['region', 'score', 'lift', 'support', 'matched_antecedent', 'matched_rules']


def rank_regions(observed, rules, regions, top_k=None):
    """Require every antecedent item, rank unseen consequents by strongest confidence.

    Multiple rules are not independent evidence, so their probabilities are never
    multiplied or added. This is record completion, not confirmed ecological presence.
    """
    observed = {str(code).strip().upper() for code in observed if str(code).strip()}
    unknown = observed.difference(regions)
    if unknown:
        raise ValueError('Unknown/out-of-scope region code(s): ' + ', '.join(sorted(unknown)))
    candidates = {}
    for row in rules.itertuples(index=False):
        consequent = str(row.consequent)
        antecedent = set(str(row.antecedent).split('|'))
        if consequent in observed or not antecedent.issubset(observed):
            continue
        score = float(row.confidence)
        if not np.isfinite(score):
            continue
        item = {'region': consequent, 'score': score, 'lift': float(row.lift),
                'support': float(row.support), 'matched_antecedent': str(row.antecedent),
                'matched_rules': 1}
        previous = candidates.get(consequent)
        if previous is None:
            candidates[consequent] = item
        else:
            number = previous['matched_rules'] + 1
            if (score, item['lift'], item['support']) > (previous['score'], previous['lift'], previous['support']):
                candidates[consequent] = item
            candidates[consequent]['matched_rules'] = number
    frame = pd.DataFrame(candidates.values(), columns=PREDICTION_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(['score', 'lift', 'support', 'region'],
                                  ascending=[False, False, False, True]).reset_index(drop=True)
    return frame.head(top_k) if top_k is not None else frame


def load_model(model_dir):
    model_dir = Path(model_dir)
    with (model_dir / 'model.json').open(encoding='utf-8') as stream:
        metadata = json.load(stream)
    rules = pd.read_csv(model_dir / 'rules.csv')
    return metadata, rules
