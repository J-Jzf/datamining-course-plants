"""Fit core/extended experiments and save reusable rule/region models."""
import os
for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_name, '1')

from plantmine.cli import experiment_parser, config_from_args
from plantmine.experiments import run_experiments


def main():
    parser = experiment_parser('Fit plant distribution rules and regional clustering, with extension experiments.')
    args = parser.parse_args()
    summary = run_experiments(args.data_dir, args.output_dir, config_from_args(args), core_only=args.core_only)
    print(f'Saved {summary["n_rules"]:,} rules and {summary["cluster_k"]} clusters to {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
