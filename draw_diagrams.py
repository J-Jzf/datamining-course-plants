"""Generate the study framework and topic overview as editable figures."""
from __future__ import annotations

import argparse
from pathlib import Path

from plantmine.diagrams import draw_diagrams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    args = parser.parse_args()
    for path in draw_diagrams(args.output_dir, args.results_dir):
        print(path.resolve())


if __name__ == "__main__":
    main()
