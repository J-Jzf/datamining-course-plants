"""Read the historical UCI Plants transactions and audit every transformation.

The input contains genus summaries and infraspecific records.  We remove the
former and union the latter at species level; this is name normalization, not
taxonomic synonym resolution.  Zero means no recorded distribution here.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


# Approximate geographic centres, used for visualization and spatial analyses.
# The original stateabbr.txt swaps AL/AB and omits PE: do not parse that file.
_US = [
    ("AL", "Alabama", -86.8, 32.8), ("AK", "Alaska", -152.0, 64.0),
    ("AZ", "Arizona", -111.9, 34.3), ("AR", "Arkansas", -92.4, 34.9),
    ("CA", "California", -119.5, 37.2), ("CO", "Colorado", -105.5, 39.0),
    ("CT", "Connecticut", -72.7, 41.6), ("DE", "Delaware", -75.4, 39.0),
    ("DC", "District of Columbia", -77.0, 38.9), ("FL", "Florida", -82.0, 28.1),
    ("GA", "Georgia", -83.4, 32.7), ("HI", "Hawaii", -157.5, 20.8),
    ("ID", "Idaho", -114.5, 44.3), ("IL", "Illinois", -89.0, 40.0),
    ("IN", "Indiana", -86.2, 40.0), ("IA", "Iowa", -93.5, 42.0),
    ("KS", "Kansas", -98.3, 38.5), ("KY", "Kentucky", -85.3, 37.8),
    ("LA", "Louisiana", -91.9, 31.0), ("ME", "Maine", -69.1, 45.4),
    ("MD", "Maryland", -76.7, 39.0), ("MA", "Massachusetts", -71.8, 42.3),
    ("MI", "Michigan", -85.5, 44.3), ("MN", "Minnesota", -94.5, 46.0),
    ("MS", "Mississippi", -89.7, 32.8), ("MO", "Missouri", -92.5, 38.5),
    ("MT", "Montana", -110.4, 47.0), ("NE", "Nebraska", -99.8, 41.5),
    ("NV", "Nevada", -116.9, 39.3), ("NH", "New Hampshire", -71.6, 43.8),
    ("NJ", "New Jersey", -74.5, 40.1), ("NM", "New Mexico", -106.0, 34.5),
    ("NY", "New York", -75.5, 43.0), ("NC", "North Carolina", -79.8, 35.7),
    ("ND", "North Dakota", -100.5, 47.5), ("OH", "Ohio", -82.8, 40.4),
    ("OK", "Oklahoma", -97.3, 35.6), ("OR", "Oregon", -120.6, 44.0),
    ("PA", "Pennsylvania", -77.8, 40.9), ("RI", "Rhode Island", -71.5, 41.7),
    ("SC", "South Carolina", -80.9, 33.9), ("SD", "South Dakota", -100.0, 44.5),
    ("TN", "Tennessee", -86.0, 35.8), ("TX", "Texas", -99.0, 31.5),
    ("UT", "Utah", -111.7, 39.3), ("VT", "Vermont", -72.7, 44.0),
    ("VA", "Virginia", -79.4, 37.5), ("WA", "Washington", -120.7, 47.4),
    ("WV", "West Virginia", -80.7, 38.6), ("WI", "Wisconsin", -89.8, 44.7),
    ("WY", "Wyoming", -107.5, 43.0),
]
_CANADA = [
    ("AB", "Alberta", -114.5, 54.5), ("BC", "British Columbia", -125.0, 54.0),
    ("MB", "Manitoba", -98.5, 55.0), ("NB", "New Brunswick", -66.5, 46.5),
    ("NL", "Newfoundland and Labrador", -60.0, 54.0),
    ("NS", "Nova Scotia", -63.5, 45.0), ("NT", "Northwest Territories", -120.0, 65.0),
    ("NU", "Nunavut", -96.0, 69.0), ("ON", "Ontario", -85.0, 50.0),
    ("PE", "Prince Edward Island", -63.2, 46.4), ("QC", "Quebec", -72.0, 53.0),
    ("SK", "Saskatchewan", -106.0, 54.0), ("YT", "Yukon", -136.0, 64.0),
]


@dataclass
class PlantDataset:
    names: np.ndarray
    genera: np.ndarray
    regions: list[str]
    matrix: np.ndarray
    region_info: pd.DataFrame
    audit: dict
    lineage: pd.DataFrame


def _species_name(raw_name):
    """Preserve named hybrids and collapse only documented rank suffixes."""
    tokens = raw_name.lower().split()
    if len(tokens) < 2:
        return None, "genus_removed"
    base_len = 3 if tokens[1] in ("x", "×") else 2
    if len(tokens) < base_len or any(t in ("x", "×") for t in tokens[:1]):
        return None, "malformed_name"
    # UCI names have var./ssp./f. rank suffixes; unknown extra tokens are audited.
    if len(tokens) > base_len and tokens[base_len] not in {"var.", "ssp.", "subsp.", "f."}:
        return None, "malformed_name"
    species = " ".join(tokens[:base_len])
    return species, "collapsed_infraspecific" if len(tokens) > base_len else "retained_species"


def load_plants(data_dir) -> PlantDataset:
    """Return species-by-region presence for US 50 states + DC + Canada 13.

    NF and LB are unioned into NL. PR, VI, GL, DENGL and FRASPM are outside
    this explicit scope; their original codes/counts remain in the audit.
    """
    data_dir = Path(data_dir)
    path = data_dir / "plants.data"
    region_info = pd.DataFrame(
        [(code, name, "USA", lon, lat) for code, name, lon, lat in _US]
        + [(code, name, "CAN", lon, lat) for code, name, lon, lat in _CANADA],
        columns=["code", "name", "country", "longitude", "latitude"],
    ).sort_values("code").reset_index(drop=True)
    regions = region_info.code.tolist()
    region_index = {code: index for index, code in enumerate(regions)}
    species_regions = defaultdict(set)
    excluded = Counter()
    raw_counts = Counter()
    actions = Counter()
    lineage = []
    nonempty_lines = 0
    for line_number, line in enumerate(path.read_text(encoding="latin1").splitlines(), 1):
        if not line.strip():
            continue
        nonempty_lines += 1
        fields = [field.strip().lower() for field in line.split(",")]
        raw_name, codes = fields[0], [code for code in fields[1:] if code]
        raw_counts.update(codes)
        species, action = _species_name(raw_name)
        normalized_codes = set()
        for code in codes:
            mapped = "NL" if code in {"nf", "lb"} else code.upper()
            if mapped in region_index:
                normalized_codes.add(mapped)
            else:
                excluded[code] += 1
        if species is not None:
            species_regions[species].update(normalized_codes)
            if not normalized_codes:
                lineage_action = "excluded_no_scope"
            else:
                lineage_action = action
        else:
            lineage_action = action
        actions[action] += 1
        lineage.append({
            "line_number": line_number, "raw_name": raw_name,
            "species": species or "", "action": lineage_action,
            "original_regions": ",".join(codes),
            "normalized_regions": ",".join(sorted(normalized_codes)),
        })
    names = np.array(sorted(name for name, codes in species_regions.items() if codes), dtype=str)
    if names.size == 0:
        raise ValueError(f"No species with in-scope distribution were found in {path}")
    matrix = np.zeros((len(names), len(regions)), dtype=bool)
    for row, name in enumerate(names):
        matrix[row, [region_index[code] for code in species_regions[name]]] = True
    metadata_count = None
    metadata_path = data_dir / "plants.names"
    if metadata_path.exists():
        match = re.search(r"Number of Instances:\s*(\d+)", metadata_path.read_text(encoding="latin1"))
        if match:
            metadata_count = int(match.group(1))
    audit = {
        "input_file": str(path.resolve()), "raw_rows": nonempty_lines,
        "metadata_declared_instances": metadata_count,
        "metadata_count_discrepancy": metadata_count is not None and metadata_count != nonempty_lines,
        "genus_rows_removed": actions["genus_removed"],
        "malformed_name_rows_removed": actions["malformed_name"],
        "infraspecific_rows_collapsed": actions["collapsed_infraspecific"],
        "species_records_before_union": actions["retained_species"] + actions["collapsed_infraspecific"],
        "unique_species_before_scope_filter": len(species_regions),
        "species_without_in_scope_presence": sum(not codes for codes in species_regions.values()),
        "clean_species": len(names), "clean_regions": len(regions),
        "recorded_presences": int(matrix.sum()),
        "matrix_density": float(matrix.mean()),
        "raw_region_code_counts": dict(sorted(raw_counts.items())),
        "excluded_region_codes": dict(sorted(excluded.items())),
        "scope": "US 50 states + District of Columbia + Canada 13 provinces/territories",
        "corrections": ["AL=Alabama; AB=Alberta", "PE=Prince Edward Island", "NF and LB merged to NL"],
        "semantics": "1=recorded presence; 0=unrecorded, not confirmed absence",
        "taxonomic_policy": "Drop genus summaries; union rank records at species names; preserve named hybrids; no synonym resolution",
    }
    return PlantDataset(names, np.array([name.split()[0] for name in names], dtype=str),
                        regions, matrix, region_info, audit, pd.DataFrame(lineage))

