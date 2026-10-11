"""Real weighted FP-growth and level-wise Apriori for region transactions.

FP-growth builds prefix trees and mines conditional trees. Apriori uses its
join/prune candidate generation and integer transaction bitsets for counting.
Both return integer transaction counts and obey an identical support cutoff.
"""

from collections import Counter
from itertools import combinations
import math

import numpy as np
import pandas as pd


RULE_COLUMNS = ["antecedent", "consequent", "count", "antecedent_count",
                "consequent_count", "support", "confidence", "lift",
                "leverage", "conviction"]


def _validate_matrix(matrix, allow_empty=False):
    matrix = np.asarray(matrix, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[1] == 0 or (matrix.shape[0] == 0 and not allow_empty):
        raise ValueError("Expected a non-empty two-dimensional transaction matrix")
    return matrix


def _column_bitsets(matrix):
    packed = np.packbits(matrix.T, axis=1, bitorder="little")
    return [int.from_bytes(row.tobytes(), "little") for row in packed]


def _apriori(matrix, threshold, max_len):
    bits = _column_bitsets(matrix)
    level = {(i,): value for i, value in enumerate(bits) if value.bit_count() >= threshold}
    result = {items: mask.bit_count() for items, mask in level.items()}
    for size in range(2, max_len + 1):
        keys = sorted(level)
        candidates = {}
        previous = set(keys)
        for offset, left in enumerate(keys):
            for right in keys[offset + 1:]:
                if left[:-1] != right[:-1]:
                    continue
                items = left + (right[-1],)
                if not all(subset in previous for subset in combinations(items, size - 1)):
                    continue
                mask = level[left] & bits[right[-1]]
                count = mask.bit_count()
                if count >= threshold:
                    candidates[items] = mask
                    result[items] = count
        level = candidates
        if not level:
            break
    return result


class _FPNode:
    __slots__ = ("item", "count", "parent", "children", "next")

    def __init__(self, item, count=0, parent=None):
        self.item, self.count, self.parent = item, count, parent
        self.children = {}
        self.next = None


def _build_tree(paths, threshold):
    frequencies = Counter()
    for path, weight in paths:
        for item in path:
            frequencies[item] += weight
    frequencies = {item: count for item, count in frequencies.items() if count >= threshold}
    if not frequencies:
        return None, {}
    order = {item: rank for rank, item in enumerate(sorted(frequencies, key=lambda item: (-frequencies[item], item)))}
    headers = {item: [count, None, None] for item, count in frequencies.items()}
    root = _FPNode(None)
    for path, weight in paths:
        filtered = sorted((item for item in path if item in order), key=order.__getitem__)
        current = root
        for item in filtered:
            child = current.children.get(item)
            if child is None:
                child = _FPNode(item, weight, current)
                current.children[item] = child
                if headers[item][1] is None:
                    headers[item][1] = child
                else:
                    headers[item][2].next = child
                headers[item][2] = child
            else:
                child.count += weight
            current = child
    return root, headers


def _fpgrowth(matrix, threshold, max_len):
    transaction_counts = Counter(tuple(np.flatnonzero(row).tolist()) for row in matrix)
    paths = list(transaction_counts.items())
    _, headers = _build_tree(paths, threshold)
    result = {}

    def grow(header_table, suffix):
        for item in sorted(header_table, key=lambda value: (header_table[value][0], value)):
            count, node, _ = header_table[item]
            pattern = suffix + (item,)
            result[tuple(sorted(pattern))] = int(count)
            if len(pattern) >= max_len:
                continue
            conditional = []
            while node is not None:
                path = []
                parent = node.parent
                while parent is not None and parent.item is not None:
                    path.append(parent.item)
                    parent = parent.parent
                if path:
                    conditional.append((tuple(path), node.count))
                node = node.next
            if conditional:
                if len(pattern) == max_len - 1:
                    # No deeper recursion is possible: the conditional-base
                    # frequencies are already the final extension counts.
                    # Avoid allocating a tree which would only read headers.
                    final_counts = Counter()
                    for path, weight in conditional:
                        for extension in path:
                            final_counts[extension] += weight
                    for extension, extension_count in final_counts.items():
                        if extension_count >= threshold:
                            result[tuple(sorted(pattern + (extension,)))] = int(extension_count)
                else:
                    _, conditional_headers = _build_tree(conditional, threshold)
                    grow(conditional_headers, pattern)

    grow(headers, ())
    return result


def mine_itemsets(matrix, min_support, max_len=3, algorithm="fpgrowth"):
    """Mine supports with ceil(min_support * n), max_len up to region count."""
    matrix = _validate_matrix(matrix)
    if not 0 < min_support <= 1:
        raise ValueError("min_support must be in (0, 1]")
    if not isinstance(max_len, (int, np.integer)) or max_len < 1:
        raise ValueError("max_len must be a positive integer")
    threshold = max(1, math.ceil(float(min_support) * matrix.shape[0] - 1e-12))
    max_len = min(int(max_len), matrix.shape[1])
    if algorithm == "apriori":
        result = _apriori(matrix, threshold, max_len)
    elif algorithm == "fpgrowth":
        result = _fpgrowth(matrix, threshold, max_len)
    else:
        raise ValueError("algorithm must be 'fpgrowth' or 'apriori'")
    return dict(sorted(result.items(), key=lambda pair: (len(pair[0]), pair[0])))


def _metrics(antecedent, consequent, count, antecedent_count, consequent_count, n):
    support = count / n if n else np.nan
    antecedent_support = antecedent_count / n if n else np.nan
    consequent_support = consequent_count / n if n else np.nan
    confidence = count / antecedent_count if antecedent_count else np.nan
    lift = confidence / consequent_support if consequent_support else np.nan
    leverage = support - antecedent_support * consequent_support
    if np.isnan(confidence):
        conviction = np.nan
    elif confidence == 1:
        conviction = np.inf if consequent_support < 1 else np.nan
    else:
        conviction = (1 - consequent_support) / (1 - confidence)
    return (tuple(antecedent), int(consequent), int(count), int(antecedent_count),
            int(consequent_count), support, confidence, lift, leverage, conviction)


def build_rules(itemsets, n_samples, min_confidence=0.75, min_lift=1.15):
    """Emit one/two-region antecedents -> one-region consequent rules."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if not 0 <= min_confidence <= 1 or min_lift < 0:
        raise ValueError("Invalid rule thresholds")
    rows = []
    for items, count in itemsets.items():
        if not 2 <= len(items) <= 3:
            continue
        for consequent in items:
            antecedent = tuple(item for item in items if item != consequent)
            antecedent_count = itemsets[antecedent]
            consequent_count = itemsets[(consequent,)]
            metrics = _metrics(antecedent, consequent, count, antecedent_count, consequent_count, n_samples)
            if metrics[6] >= min_confidence and metrics[7] >= min_lift:
                rows.append(metrics)
    return pd.DataFrame(rows, columns=RULE_COLUMNS)


def evaluate_rules(rules, matrix):
    """Assess frozen rule keys on independent transactions, without filtering."""
    matrix = _validate_matrix(matrix, allow_empty=True)
    bits = _column_bitsets(matrix)
    all_mask = (1 << matrix.shape[0]) - 1
    rows = []
    for rule in rules.itertuples(index=False):
        antecedent = tuple(rule.antecedent)
        consequent = int(rule.consequent)
        if not antecedent or consequent in antecedent:
            raise ValueError("Rules require non-empty antecedents and distinct consequent")
        mask = all_mask
        for item in antecedent:
            mask &= bits[int(item)]
        rows.append(_metrics(antecedent, consequent, (mask & bits[consequent]).bit_count(),
                             mask.bit_count(), bits[consequent].bit_count(), matrix.shape[0]))
    return pd.DataFrame(rows, columns=RULE_COLUMNS)
