"""Shared command-line options. No configuration files are needed."""
from pathlib import Path
import argparse
from .experiments import ExperimentConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def experiment_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'plantsdata')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'outputs')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--min-support', type=float, default=.03)
    parser.add_argument('--min-confidence', type=float, default=.75)
    parser.add_argument('--min-lift', type=float, default=1.15)
    parser.add_argument('--clusters', type=int, default=None, help='Fixed k (2..10); default: best silhouette within k=4..9.')
    parser.add_argument('--bootstrap-repeats', type=int, default=30)
    parser.add_argument('--benchmark-repeats', type=int, default=2)
    parser.add_argument('--permutations', type=int, default=999)
    parser.add_argument('--completion-queries', type=int, default=2000)
    parser.add_argument('--core-only', action='store_true', help='Run discovery/validation/clustering/completion, omit extension experiments.')
    return parser


def config_from_args(args):
    return ExperimentConfig(seed=args.seed, min_support=args.min_support,
                            min_confidence=args.min_confidence, min_lift=args.min_lift,
                            clusters=args.clusters, bootstrap_repeats=args.bootstrap_repeats,
                            benchmark_repeats=args.benchmark_repeats, n_permutations=args.permutations,
                            max_completion_queries=args.completion_queries)
