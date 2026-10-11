"""Reproducible core and extension experiments; all parameters live in Python/CLI."""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from time import perf_counter
import hashlib
import json
import platform
import sys
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import fisher_exact, rankdata, spearmanr
from sklearn.metrics import adjusted_rand_score

from .data import load_plants
from .mining import mine_itemsets, build_rules, evaluate_rules
from .clustering import cluster_regions
from .inference import rank_regions


@dataclass
class ExperimentConfig:
    seed: int = 2026
    min_support: float = .03
    min_confidence: float = .75
    min_lift: float = 1.15
    max_len: int = 3
    test_fraction: float = .2
    clusters: int | None = None
    bootstrap_repeats: int = 30
    benchmark_repeats: int = 2
    n_permutations: int = 999
    max_completion_queries: int = 2000
    broad_fraction: float = .05


def split_indices(genera, seed=2026, grouped=False, test_fraction=.2):
    if not 0 < test_fraction < 1:
        raise ValueError('test_fraction must be between 0 and 1.')
    genera = np.asarray(genera)
    rng = np.random.default_rng(seed)
    units = np.unique(genera) if grouped else np.arange(len(genera))
    if len(units) < 2:
        raise ValueError('At least two split units are required.')
    order = rng.permutation(units)
    n_test = min(len(units) - 1, max(1, int(round(len(units) * test_fraction))))
    if grouped:
        test_mask = np.isin(genera, order[:n_test])
        return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
    return np.sort(order[n_test:]), np.sort(order[:n_test])


def bh_adjust(p_values):
    """Benjamini-Hochberg adjustment over the complete supplied hypothesis family."""
    values = np.asarray(p_values, dtype=float)
    if not len(values):
        return values.copy()
    values = np.where(np.isfinite(values), values, 1.)
    order = np.argsort(values, kind='stable')
    adjusted = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.)
    return result


def spatial_permutation_test(distance, regions, seed=2026, n_permutations=999):
    """Spearman Mantel-style test: permute region labels jointly in rows/columns."""
    latitude = np.radians(regions.latitude.to_numpy(float))
    longitude = np.radians(regions.longitude.to_numpy(float))
    dlat = latitude[:, None] - latitude[None, :]
    dlon = longitude[:, None] - longitude[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(latitude[:, None]) * np.cos(latitude[None, :]) * np.sin(dlon / 2) ** 2
    geography = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    i, j = np.triu_indices(len(regions), 1)
    geo_rank = rankdata(geography[i, j])
    flora_rank = rankdata(distance[i, j])
    correlation = float(np.corrcoef(geo_rank, flora_rank)[0, 1])
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_permutations):
        permutation = rng.permutation(len(regions))
        r = np.corrcoef(geo_rank, rankdata(distance[permutation[i], permutation[j]]))[0, 1]
        extreme += abs(r) >= abs(correlation)
    frame = pd.DataFrame({'code_a': regions.code.to_numpy()[i], 'code_b': regions.code.to_numpy()[j],
                          'geo_distance_km': geography[i, j], 'floristic_distance': distance[i, j]})
    stats = {'spearman_r': correlation, 'permutation_p': (extreme + 1) / (n_permutations + 1),
             'n_permutations': int(n_permutations), 'alternative': 'two-sided',
             'permutation_unit': 'region labels, jointly permuted matrix rows and columns',
             'geography': 'great-circle distance between approximate administrative representative points'}
    return frame, stats


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def _write_csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def _encode_rules(rules, regions):
    frame = rules.copy()
    frame['antecedent_size'] = frame.antecedent.map(len)
    frame['antecedent'] = frame.antecedent.map(lambda items: '|'.join(regions[int(i)] for i in items))
    frame['consequent'] = frame.consequent.map(lambda i: regions[int(i)])
    frame.insert(0, 'id', frame.antecedent + '->' + frame.consequent)
    return frame


def _validated_rules(rules, test_matrix, regions, config):
    heldout = evaluate_rules(rules, test_matrix)
    frame = _encode_rules(rules, regions)
    for column in ['count', 'antecedent_count', 'consequent_count', 'support', 'confidence', 'lift']:
        frame['train_' + column] = rules[column].to_numpy()
        frame['validation_' + column] = heldout[column].to_numpy()
    frame['validation_joint_count'] = heldout['count'].to_numpy()
    p_values = []
    n = len(test_matrix)
    for row in heldout.itertuples(index=False):
        joint, ante, cons = int(row.count), int(row.antecedent_count), int(row.consequent_count)
        table = [[joint, ante - joint], [cons - joint, n - ante - cons + joint]]
        p_values.append(float(fisher_exact(table, alternative='greater').pvalue) if ante and cons else 1.)
    frame['validation_p'] = p_values
    frame['validation_q'] = bh_adjust(p_values)
    # Independent held-out inference over the frozen training-rule family.
    frame['validation_pass'] = ((frame.validation_antecedent_count >= 20) &
                                (frame.validation_support >= config.min_support) &
                                (frame.validation_confidence >= config.min_confidence) &
                                (frame.validation_lift >= config.min_lift) &
                                (frame.validation_q <= .05))
    return frame


def _discover(matrix, config, min_support=None, algorithm='fpgrowth'):
    itemsets = mine_itemsets(matrix, min_support=min_support or config.min_support,
                            max_len=config.max_len, algorithm=algorithm)
    rules = build_rules(itemsets, len(matrix), min_confidence=config.min_confidence, min_lift=config.min_lift)
    if rules.empty:
        raise RuntimeError('No rules passed the training thresholds. Reduce --min-confidence or --min-lift.')
    return itemsets, rules


def _save_dataset(dataset, results_dir):
    np.savez_compressed(results_dir / 'dataset.npz', matrix=dataset.matrix, names=dataset.names,
                        genera=dataset.genera, regions=np.asarray(dataset.regions))
    _write_json(results_dir / 'data_audit.json', dataset.audit)
    if hasattr(dataset, 'lineage'):
        _write_csv(dataset.lineage, results_dir / 'taxonomy_lineage.csv')
    _write_csv(pd.DataFrame({'species': dataset.names, 'genus': dataset.genera,
                            'recorded_regions': ['|'.join(np.asarray(dataset.regions)[row]) for row in dataset.matrix],
                            'n_regions': dataset.matrix.sum(axis=1)}), results_dir / 'cleaned_species.csv')


def _completion_experiment(test_matrix, encoded_rules, regions, train_matrix, config):
    """Hide one observed region per eligible held-out species, never use unobserved negatives."""
    rng = np.random.default_rng(config.seed + 401)
    eligible = np.flatnonzero(test_matrix.sum(axis=1) >= 2)
    selected = rng.choice(eligible, min(config.max_completion_queries, len(eligible)), replace=False)
    popularity = np.argsort(-train_matrix.sum(axis=0), kind='stable')
    region_array = np.asarray(regions)
    # Compile rule matching once; per-query calls through pandas would be needlessly expensive.
    code_to_index = {code: i for i, code in enumerate(regions)}
    lookup = {}
    for row in encoded_rules.itertuples(index=False):
        key = tuple(sorted(code_to_index[c] for c in row.antecedent.split('|')))
        lookup.setdefault(key, []).append((code_to_index[row.consequent], float(row.confidence), float(row.lift), float(row.support)))
    records = []
    from itertools import combinations
    for species_index in selected:
        positive = np.flatnonzero(test_matrix[species_index])
        hidden = int(rng.choice(positive))
        observed = sorted(int(i) for i in positive if i != hidden)
        seen = set(observed)
        scores = {}
        for length in range(1, min(config.max_len - 1, len(observed)) + 1):
            for antecedent in combinations(observed, length):
                for consequent, confidence, lift, support in lookup.get(antecedent, []):
                    if consequent not in seen:
                        score = (confidence, lift, support, -consequent)
                        if consequent not in scores or score > scores[consequent]:
                            scores[consequent] = score
        ranked_rules = sorted(scores, key=scores.get, reverse=True)
        ranked_popularity = [int(i) for i in popularity if int(i) not in seen]
        ranked_random = rng.permutation([i for i in range(len(regions)) if i not in seen]).tolist()
        for method, ranking in [('Association rules', ranked_rules), ('Region popularity', ranked_popularity), ('Random', ranked_random)]:
            records.append({'species_index_in_test': int(species_index), 'method': method,
                            'hidden_region': regions[hidden], 'n_observed': len(observed),
                            'covered': bool(ranking), 'hit_at_1': hidden in ranking[:1],
                            'hit_at_3': hidden in ranking[:3], 'hit_at_5': hidden in ranking[:5],
                            'top5': '|'.join(region_array[ranking[:5]])})
    details = pd.DataFrame(records, columns=['species_index_in_test', 'method', 'hidden_region',
                                             'n_observed', 'covered', 'hit_at_1', 'hit_at_3', 'hit_at_5', 'top5'])
    if details.empty:
        summary = pd.DataFrame([{'method': method, 'n_queries': 0, 'coverage': np.nan,
                                 'recall_at_1': np.nan, 'recall_at_3': np.nan, 'recall_at_5': np.nan}
                                for method in ['Association rules', 'Region popularity', 'Random']])
        return details, summary
    summary = []
    for method, frame in details.groupby('method', sort=False):
        summary.append({'method': method, 'n_queries': len(frame), 'coverage': float(frame.covered.mean()),
                        'recall_at_1': float(frame.hit_at_1.mean()), 'recall_at_3': float(frame.hit_at_3.mean()),
                        'recall_at_5': float(frame.hit_at_5.mean())})
    return details, pd.DataFrame(summary)


def run_core(data_dir, output_dir, config=None):
    config = config or ExperimentConfig()
    started = perf_counter()
    output_dir = Path(output_dir)
    results_dir, model_dir = output_dir / 'results', output_dir / 'model'
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    # A new fit invalidates old extension files even if this run stops midway.
    _write_json(results_dir / 'run_summary.json', {'config': asdict(config),
                                                 'core_complete': False, 'extensions_complete': False})
    print('[1/8] Cleaning taxonomy and region records...', flush=True)
    dataset = load_plants(data_dir)
    _save_dataset(dataset, results_dir)
    train_index, validation_index = split_indices(dataset.genera, seed=config.seed, test_fraction=config.test_fraction)
    np.savez_compressed(results_dir / 'splits.npz', train=train_index, validation=validation_index)
    train_matrix, test_matrix = dataset.matrix[train_index], dataset.matrix[validation_index]
    print(f'      {len(dataset.names):,} species, {len(dataset.regions)} regions; train={len(train_index):,}, held-out={len(validation_index):,}', flush=True)
    print('[2/8] FP-growth, frozen-rule held-out validation, and region clustering...', flush=True)
    mining_started = perf_counter()
    itemsets, rules = _discover(train_matrix, config)
    mining_seconds = perf_counter() - mining_started
    encoded = _validated_rules(rules, test_matrix, dataset.regions, config)
    _write_csv(encoded, results_dir / 'core_rules.csv')
    _write_csv(encoded.sort_values(['validation_pass', 'lift', 'confidence', 'support'], ascending=False), results_dir / 'rules_ranked.csv')
    _write_csv(_encode_rules(rules, dataset.regions), model_dir / 'rules.csv')
    selection = []
    candidates = {}
    for k in range(2, min(11, len(dataset.regions))):
        fit = cluster_regions(train_matrix, k=k)
        candidates[k] = fit
        selection.append({'k': k, 'silhouette': fit['silhouette'], 'eligible': 4 <= k <= 9})
    if config.clusters is not None:
        k = config.clusters
        if k not in candidates:
            raise ValueError('--clusters must be between 2 and 10.')
    else:
        # A priori 4..9 range gives useful regional resolution without fitting geography.
        k = max((x for x in candidates if 4 <= x <= 9), key=lambda x: candidates[x]['silhouette'])
    fit = candidates[k]
    selection_frame = pd.DataFrame(selection)
    selection_frame['selected'] = selection_frame.k == k
    _write_csv(selection_frame, results_dir / 'cluster_selection.csv')
    np.save(results_dir / 'region_distance.npy', fit['distance'])
    np.save(results_dir / 'region_linkage.npy', fit['linkage'])
    metadata = dataset.region_info.copy().set_index('code').loc[dataset.regions].reset_index()
    metadata['richness'] = dataset.matrix.sum(axis=0)
    metadata['train_richness'] = train_matrix.sum(axis=0)
    metadata['cluster'] = fit['labels']
    _write_csv(metadata, results_dir / 'regions.csv')
    _write_csv(metadata, model_dir / 'regions.csv')
    np.save(model_dir / 'region_distance.npy', fit['distance'])
    np.save(model_dir / 'region_linkage.npy', fit['linkage'])
    matrix_path = Path(data_dir) / 'plants.data'
    summary = {'core_complete': True, 'extensions_complete': False,
               'n_species': len(dataset.names), 'n_regions': len(dataset.regions),
               'n_train': len(train_index), 'n_validation': len(validation_index),
               'n_itemsets': len(itemsets), 'n_rules': len(rules),
               'n_validated_rules': int(encoded.validation_pass.sum()),
               'cluster_k': int(k), 'silhouette': float(fit['silhouette']),
               'mining_seconds': mining_seconds, 'config': asdict(config),
               'data_sha256': hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
               'python': sys.version.split()[0], 'platform': platform.platform(),
               'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
               'interpretation': 'Recorded presence; zero is unrecorded, not verified absence.',
               'selection': 'All rules and cluster selection fitted on training species only; full-data richness is descriptive.',
               'validation': 'One-sided Fisher tests and BH q-values across the frozen training-rule family; antecedent count >=20.',
               'cluster_selection': 'Highest training Jaccard silhouette within a priori k=4..9 unless --clusters supplied.'}
    _write_json(model_dir / 'model.json', {'regions': dataset.regions, 'config': asdict(config),
                                         'n_train': len(train_matrix), 'cluster_k': k,
                                         'score': 'Maximum training confidence among matching rules; not calibrated ecological probability.',
                                         'data_sha256': summary['data_sha256']})
    details, completion = _completion_experiment(test_matrix, _encode_rules(rules, dataset.regions), dataset.regions, train_matrix, config)
    _write_csv(details, results_dir / 'missing_region_queries.csv')
    _write_csv(completion, results_dir / 'missing_region_summary.csv')
    summary['core_seconds'] = perf_counter() - started
    _write_json(results_dir / 'run_summary.json', summary)
    print(f'      {len(rules):,} training rules, {summary["n_validated_rules"]:,} held-out replications; k={k}, silhouette={fit["silhouette"]:.3f}', flush=True)
    return dataset, train_matrix, test_matrix, rules, encoded, fit, summary


def run_extensions(dataset, train_matrix, test_matrix, rules, encoded, fit, summary, output_dir, config):
    results_dir = Path(output_dir) / 'results'
    rng = np.random.default_rng(config.seed + 100)
    print('[3/8] Bootstrap rule recurrence and region co-assignment...', flush=True)
    confidence, lift, retained = [], [], []
    coassignment = np.zeros((len(dataset.regions), len(dataset.regions)), dtype=float)
    bootstrap_cluster_rows = []
    for repeat in range(config.bootstrap_repeats):
        resampled = train_matrix[rng.integers(0, len(train_matrix), len(train_matrix))]
        metrics = evaluate_rules(rules, resampled)
        confidence.append(metrics.confidence.to_numpy())
        lift.append(metrics.lift.to_numpy())
        retained.append(((metrics.support >= config.min_support) & (metrics.confidence >= config.min_confidence) &
                         (metrics.lift >= config.min_lift)).to_numpy())
        current = cluster_regions(resampled, k=summary['cluster_k'])
        coassignment += current['labels'][:, None] == current['labels'][None, :]
        bootstrap_cluster_rows.append({'repeat': repeat, 'ari_to_baseline': adjusted_rand_score(fit['labels'], current['labels']),
                                       'silhouette': current['silhouette']})
        if (repeat + 1) % 10 == 0 or repeat + 1 == config.bootstrap_repeats:
            print(f'      bootstrap {repeat + 1}/{config.bootstrap_repeats}', flush=True)
    bootstrap = pd.DataFrame({'id': encoded.id, 'recurrence': np.mean(retained, axis=0)})
    for name, array in [('confidence', confidence), ('lift', lift)]:
        interval = np.nanpercentile(np.asarray(array), [2.5, 50, 97.5], axis=0)
        bootstrap[name + '_low'], bootstrap[name + '_median'], bootstrap[name + '_high'] = interval
    _write_csv(bootstrap, results_dir / 'bootstrap_rules.csv')
    np.save(results_dir / 'bootstrap_coassignment.npy', coassignment / config.bootstrap_repeats)
    _write_csv(pd.DataFrame(bootstrap_cluster_rows), results_dir / 'bootstrap_clusters.csv')
    summary['bootstrap_mean_ari'] = float(np.mean([row['ari_to_baseline'] for row in bootstrap_cluster_rows]))
    print('[4/8] Independent genus-grouped discovery and validation...', flush=True)
    grouped_train, grouped_test = split_indices(dataset.genera, seed=config.seed, grouped=True, test_fraction=config.test_fraction)
    np.savez_compressed(results_dir / 'grouped_splits.npz', train=grouped_train, validation=grouped_test)
    _, grouped_rules = _discover(dataset.matrix[grouped_train], config)
    grouped_encoded = _validated_rules(grouped_rules, dataset.matrix[grouped_test], dataset.regions, config)
    _write_csv(grouped_encoded, results_dir / 'grouped_validation_rules.csv')
    matched = encoded[['id', 'validation_confidence', 'validation_lift']].merge(
        grouped_encoded[['id', 'validation_confidence', 'validation_lift']], on='id', suffixes=('_random', '_grouped'))
    _write_csv(matched, results_dir / 'matched_split_rules.csv')
    summary['grouped'] = {'n_train': len(grouped_train), 'n_validation': len(grouped_test),
                          'n_train_genera': len(np.unique(dataset.genera[grouped_train])),
                          'n_validation_genera': len(np.unique(dataset.genera[grouped_test])),
                          'n_rules': len(grouped_rules), 'n_validated_rules': int(grouped_encoded.validation_pass.sum()),
                          'matched_random_rules': len(matched)}
    print('[5/8] Broad-species and distance/linkage sensitivity...', flush=True)
    breadth = train_matrix.sum(axis=1)
    number = max(1, int(np.ceil(config.broad_fraction * len(train_matrix))))
    # Exactly top 5%, ties broken deterministically by species index, no validation information.
    broad_indices = np.argsort(-breadth.astype(np.int64), kind='stable')[:number]
    keep = np.ones(len(train_matrix), dtype=bool)
    keep[broad_indices] = False
    filtered = train_matrix[keep]
    filtered_metrics = evaluate_rules(rules, filtered)
    filtered_fit = cluster_regions(filtered, k=summary['cluster_k'])
    broad = pd.DataFrame({'id': encoded.id, 'baseline_lift': rules.lift, 'filtered_lift': filtered_metrics.lift,
                          'baseline_confidence': rules.confidence, 'filtered_confidence': filtered_metrics.confidence,
                          'baseline_support': rules.support, 'filtered_support': filtered_metrics.support})
    broad['retained'] = ((broad.filtered_lift >= config.min_lift) & (broad.filtered_confidence >= config.min_confidence) &
                         (broad.filtered_support >= config.min_support))
    _write_csv(broad, results_dir / 'broad_species_rules.csv')
    broad_stats = {'baseline_k': summary['cluster_k'], 'filtered_k': summary['cluster_k'],
                   'ari': float(adjusted_rand_score(fit['labels'], filtered_fit['labels'])), 'removed_n': number,
                   'removed_fraction': number / len(train_matrix), 'minimum_removed_breadth': int(breadth[broad_indices].min()),
                   'rule_retention_fraction': float(broad.retained.mean()),
                   'tie_policy': 'Exactly top fraction; ties use deterministic training-row order.'}
    _write_json(results_dir / 'broad_species_summary.json', broad_stats)
    np.save(results_dir / 'broad_species_labels.npy', filtered_fit['labels'])
    sensitivity = []
    for metric in ['jaccard', 'cosine']:
        for method in ['average', 'complete']:
            candidate = cluster_regions(train_matrix, k=summary['cluster_k'], metric=metric, method=method)
            sensitivity.append({'setting': metric + ' / ' + method, 'k': summary['cluster_k'],
                                'silhouette': candidate['silhouette'],
                                'ari_to_baseline': adjusted_rand_score(fit['labels'], candidate['labels'])})
    _write_csv(pd.DataFrame(sensitivity), results_dir / 'metric_sensitivity.csv')
    print('[6/8] Apriori / FP-growth equality and runtime benchmark...', flush=True)
    benchmark_rows, support_rows = [], []
    sample_order = np.random.default_rng(config.seed + 300).permutation(len(train_matrix))
    sizes = sorted(set([min(5000, len(train_matrix)), min(10000, len(train_matrix)), len(train_matrix)]))
    supports = sorted(set([config.min_support, .06, .10]))
    for size in sizes:
        subset = train_matrix[sample_order[:size]]
        for support in supports:
            results = {}
            for repeat in range(config.benchmark_repeats):
                # Alternate ordering to reduce systematic warm-up/order bias.
                algorithms = ['apriori', 'fpgrowth'] if repeat % 2 == 0 else ['fpgrowth', 'apriori']
                for algorithm in algorithms:
                    start = perf_counter()
                    patterns = mine_itemsets(subset, min_support=support, max_len=config.max_len, algorithm=algorithm)
                    seconds = perf_counter() - start
                    results[algorithm] = patterns
                    benchmark_rows.append({'algorithm': algorithm, 'min_support': support, 'n_plants': size,
                                           'repeat': repeat, 'seconds': seconds, 'n_itemsets': len(patterns), 'equivalent': True})
                if results['apriori'] != results['fpgrowth']:
                    raise AssertionError(f'Apriori/FP-growth results differ at n={size}, support={support}.')
            print(f'      n={size:,}, support={support:g}: {len(results["fpgrowth"]):,} equal itemsets', flush=True)
            if size == len(train_matrix):
                alternative = build_rules(results['fpgrowth'], len(subset), min_confidence=config.min_confidence, min_lift=config.min_lift)
                validation = _validated_rules(alternative, test_matrix, dataset.regions, replace(config, min_support=support))
                support_rows.append({'min_support': support, 'n_itemsets': len(results['fpgrowth']),
                                     'n_rules': len(alternative), 'validated_fraction': float(validation.validation_pass.mean()) if len(validation) else 0.,
                                     'mean_validation_confidence': float(validation.validation_confidence.mean()) if len(validation) else 0.})
    _write_csv(pd.DataFrame(benchmark_rows), results_dir / 'algorithm_benchmark.csv')
    _write_csv(pd.DataFrame(support_rows), results_dir / 'support_sensitivity.csv')
    print('[7/8] Geography / floristic-distance permutation experiment...', flush=True)
    metadata = pd.read_csv(results_dir / 'regions.csv')
    pairs, spatial = spatial_permutation_test(fit['distance'], metadata, seed=config.seed + 200, n_permutations=config.n_permutations)
    _write_csv(pairs, results_dir / 'spatial_pairs.csv')
    _write_json(results_dir / 'spatial_test.json', spatial)
    summary['spatial'] = spatial
    summary['broad_species'] = broad_stats
    _write_json(results_dir / 'run_summary.json', summary)
    return summary


def run_experiments(data_dir, output_dir, config=None, core_only=False):
    config = config or ExperimentConfig()
    if config.bootstrap_repeats < 2 or config.benchmark_repeats < 1 or config.n_permutations < 1 or config.max_completion_queries < 1:
        raise ValueError('Need bootstrap>=2, benchmark>=1, permutations>=1 and queries>=1.')
    if not 0 < config.broad_fraction < 1:
        raise ValueError('broad_fraction must be between 0 and 1.')
    start = perf_counter()
    dataset, train, test, rules, encoded, fit, summary = run_core(data_dir, output_dir, config)
    if not core_only:
        summary = run_extensions(dataset, train, test, rules, encoded, fit, summary, output_dir, config)
    summary['all_experiments_seconds'] = perf_counter() - start
    summary['extensions_complete'] = not core_only
    _write_json(Path(output_dir) / 'results' / 'run_summary.json', summary)
    return summary
