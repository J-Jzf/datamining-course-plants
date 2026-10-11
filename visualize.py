"""Regenerate all figures from saved tables without re-running experiments."""
from pathlib import Path
import argparse
from plantmine.plots import generate_figures
from plantmine.diagrams import draw_diagrams


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=project / 'outputs' / 'results')
    parser.add_argument('--figures-dir', type=Path, default=project / 'outputs' / 'figures')
    args = parser.parse_args()
    paths = generate_figures(args.results_dir, args.figures_dir)
    paths += draw_diagrams(args.figures_dir, args.results_dir)
    # Gallery links are relative to the output directory, including custom subdirectories.
    if args.figures_dir.resolve().parent == args.results_dir.resolve().parent:
        from plantmine.gallery import write_gallery
        write_gallery(paths, args.figures_dir.resolve().parent)
    print(f'Saved {len(paths)} figure files to {args.figures_dir.resolve()}')


if __name__ == '__main__':
    main()
