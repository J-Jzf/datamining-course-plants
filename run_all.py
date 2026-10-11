"""One command: preprocessing, discovery, inference evaluation, extensions, figures."""
import os
for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_name, '1')

from time import perf_counter
from plantmine.cli import experiment_parser, config_from_args
from plantmine.experiments import run_experiments


def main():
    parser = experiment_parser('Run every plant-mining experiment and render publication figures.')
    parser.add_argument('--no-plots', action='store_true', help='Save experiments and models only.')
    args = parser.parse_args()
    start = perf_counter()
    summary = run_experiments(args.data_dir, args.output_dir, config_from_args(args), core_only=args.core_only)
    if not args.no_plots:
        from plantmine.plots import generate_figures
        from plantmine.diagrams import draw_diagrams
        print('[8/8] Rendering maps, experiment figures, framework and topic overview...', flush=True)
        figures = generate_figures(args.output_dir / 'results', args.output_dir / 'figures')
        figures += draw_diagrams(args.output_dir / 'figures', args.output_dir / 'results')
        from plantmine.gallery import write_gallery
        write_gallery(figures, args.output_dir)
        print(f'      {len(figures)} figure files saved.', flush=True)
    print(f'Finished in {(perf_counter() - start) / 60:.1f} min. Outputs: {args.output_dir.resolve()}', flush=True)
    print(f'Rules: {summary["n_rules"]:,}; held-out replications: {summary["n_validated_rules"]:,}; clusters: {summary["cluster_k"]}', flush=True)


if __name__ == '__main__':
    main()
