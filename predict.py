"""Complete a plant's region records using the frozen training rules."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from plantmine.inference import load_model, rank_regions


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=project / 'outputs' / 'model')
    parser.add_argument('--regions', nargs='+', default=None, help='Observed uppercase region codes, e.g. CA OR (comma separators also accepted).')
    parser.add_argument('--species', default=None, help='Look up an exact cleaned Latin species name; all its recorded regions become observed.')
    parser.add_argument('--dataset', type=Path, default=project / 'outputs' / 'results' / 'dataset.npz')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--output', type=Path, default=None, help='Optional path for prediction CSV.')
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error('--top-k must be positive.')
    metadata, rules = load_model(args.model_dir)
    if args.species and args.regions:
        parser.error('Choose --species or --regions.')
    if args.species:
        with np.load(args.dataset, allow_pickle=False) as dataset:
            matches = np.flatnonzero(dataset['names'] == args.species.strip().lower())
            if not len(matches):
                parser.error('Species not found; see outputs/results/cleaned_species.csv.')
            observed = dataset['regions'][dataset['matrix'][matches[0]]].tolist()
    else:
        observed = [part.upper() for item in (args.regions or ['OR', 'WA']) for part in item.replace(',', ' ').split()]
    try:
        prediction = rank_regions(observed, rules, metadata['regions'], top_k=args.top_k)
    except ValueError as error:
        parser.error(str(error))
    print('Observed:', ', '.join(sorted(set(observed))))
    print('Score = strongest matching training-rule confidence; candidate record completion only.')
    if prediction.empty:
        print('No unseen region matched a discovered rule at the current thresholds.')
    else:
        print(prediction.to_string(index=False, float_format=lambda value: f'{value:.4f}'))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        prediction.to_csv(args.output, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
