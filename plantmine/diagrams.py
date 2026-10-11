"""Editable, code-drawn diagrams for the plant distribution mining study.

The overview uses real distribution records when they are available. Diagram
glyphs illustrate the analysis and are never presented as fitted model output.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np


FOREST = "#2f6b57"
OCHRE = "#c28e39"
BLUE = "#527a94"
INK = "#243b36"
MUTED = "#62736c"
PALE = "#edf3ef"
PAPER = "#ffffff"
EXAMPLES = ("abies amabilis", "quercus rubra", "opuntia basilaris")
EXAMPLE_REGIONS = ("wa", "or", "ca", "az", "ut", "on", "ny", "qc")


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": PAPER,
    })


def _canvas(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, width), ylim=(0, height), aspect="equal")
    ax.axis("off")
    return fig, ax


def _box(ax, x, y, width, height, color=FOREST, fill=PALE, radius=.15):
    patch = FancyBboxPatch((x, y), width, height,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          linewidth=1.1, edgecolor=color, facecolor=fill)
    ax.add_patch(patch)
    return patch


def _text(ax, x, y, label, size=10, color=INK, weight="normal", **kwargs):
    return ax.text(x, y, label, fontsize=size, color=color, weight=weight,
                   va="center", **kwargs)


def _arrow(ax, start, end, color=MUTED, lw=1.65, dashed=False,
           connection="arc3,rad=0", head=12):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                mutation_scale=head, linewidth=lw,
                                color=color, linestyle="--" if dashed else "-",
                                connectionstyle=connection, shrinkA=0, shrinkB=0))


def _leaf(ax, x, y, scale=1.0, color=FOREST, kind="oak"):
    """Small botanical glyph; its form identifies a conifer, oak or cactus."""
    if kind == "conifer":
        ax.plot([x, x + .30 * scale], [y - .45 * scale, y + .48 * scale],
                color=color, lw=1.6)
        for k, t in enumerate(np.linspace(0.0, .8, 7)):
            px, py = x + .30 * scale * t, y - .40 * scale + .85 * scale * t
            length = (.28 - .12 * t) * scale
            ax.plot([px, px - length], [py, py + .13 * scale], color=color, lw=1.7)
            ax.plot([px, px + length], [py, py + .02 * scale], color=color, lw=1.7)
    elif kind == "cactus":
        for dx, dy, sx, sy, angle in ((0, -.12, .35, .68, -6),
                                      (-.23, .23, .31, .49, -23),
                                      (.23, .20, .32, .47, 25)):
            ax.add_patch(Ellipse((x + dx * scale, y + dy * scale),
                                  sx * scale, sy * scale, angle=angle,
                                  facecolor=color, edgecolor=color, lw=1))
        for dx, dy in ((-.2, .24), (.17, .2), (-.05, -.17), (.03, .06)):
            ax.plot([x + dx * scale, x + (dx + .04) * scale],
                    [y + dy * scale, y + (dy + .02) * scale], color=PALE, lw=.9)
        ax.plot([x, x], [y - .45 * scale, y - .65 * scale], color=color, lw=1.5)
    else:
        outline = np.array([
            [0, -.48], [-.11, -.29], [-.28, -.20], [-.20, -.08],
            [-.38, .03], [-.23, .15], [-.36, .28], [-.17, .33],
            [-.14, .47], [0, .60], [.14, .47], [.17, .33],
            [.36, .28], [.23, .15], [.38, .03], [.20, -.08],
            [.28, -.20], [.11, -.29],
        ])
        ax.add_patch(Polygon(outline * scale + np.array([x, y]),
                              closed=True, facecolor=color, edgecolor=color, lw=1))
        ax.plot([x, x], [y - .64 * scale, y + .44 * scale], color=PALE, lw=1.15)
        for t in (-.16, .10, .30):
            ax.plot([x, x - .20 * scale], [y + t * scale, y + (t + .09) * scale],
                    color=PALE, lw=.7)
            ax.plot([x, x + .20 * scale], [y + t * scale, y + (t + .09) * scale],
                    color=PALE, lw=.7)


def _matrix(ax, x, y, values, cell=.22, colors=None, labels=None):
    values = np.asarray(values)
    colors = colors or {0: "#e7ebe7", 1: FOREST}
    nrows, ncols = values.shape
    for i in range(nrows):
        for j in range(ncols):
            ax.add_patch(Polygon([
                (x + j * cell, y - i * cell),
                (x + (j + .91) * cell, y - i * cell),
                (x + (j + .91) * cell, y - (i + .91) * cell),
                (x + j * cell, y - (i + .91) * cell),
            ], facecolor=colors[int(values[i, j])], edgecolor="none"))
    if labels:
        for j, label in enumerate(labels):
            _text(ax, x + (j + .45) * cell, y + .17, label, size=8.4, ha="center")


def _rules_glyph(ax, x, y, scale=1.0):
    nodes = [(x - .45 * scale, y + .36 * scale),
             (x - .45 * scale, y - .36 * scale), (x + .45 * scale, y)]
    for point, label in zip(nodes, ("A", "B", "C")):
        ax.add_patch(Circle(point, .19 * scale, facecolor=PAPER,
                            edgecolor=FOREST, lw=1.5))
        _text(ax, *point, label, size=8.3 * scale, ha="center", color=FOREST)
    _arrow(ax, (nodes[0][0] + .19 * scale, nodes[0][1]),
           (nodes[2][0] - .19 * scale, nodes[2][1] + .04 * scale), color=FOREST)
    _arrow(ax, (nodes[1][0] + .19 * scale, nodes[1][1]),
           (nodes[2][0] - .19 * scale, nodes[2][1] - .04 * scale), color=FOREST)


def _dendrogram_glyph(ax, x, y, scale=1.0):
    leaf_x = x + np.array([-.5, -.2, .2, .5]) * scale
    for xx in leaf_x[:2]:
        ax.plot([xx, xx], [y - .3 * scale, y + .0 * scale], color=BLUE, lw=1.6)
    for xx in leaf_x[2:]:
        ax.plot([xx, xx], [y - .3 * scale, y + .12 * scale], color=OCHRE, lw=1.6)
    ax.plot(leaf_x[:2], [y, y], color=BLUE, lw=1.6)
    ax.plot(leaf_x[2:], [y + .12 * scale, y + .12 * scale], color=OCHRE, lw=1.6)
    ax.plot([x - .35 * scale, x - .35 * scale], [y, y + .45 * scale], color=MUTED, lw=1.6)
    ax.plot([x + .35 * scale, x + .35 * scale], [y + .12 * scale, y + .45 * scale], color=MUTED, lw=1.6)
    ax.plot([x - .35 * scale, x + .35 * scale], [y + .45 * scale] * 2, color=MUTED, lw=1.6)
    for j, xx in enumerate(leaf_x):
        ax.add_patch(Circle((xx, y - .40 * scale), .055 * scale,
                            facecolor=BLUE if j < 2 else OCHRE, edgecolor="none"))


def _save(fig, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for extension in ("png", "svg", "pdf"):
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=300, facecolor=PAPER, bbox_inches=None)
        files.append(path)
    plt.close(fig)
    return files


def _framework(output_dir: Path) -> list[Path]:
    fig, ax = _canvas(15.6, 9.1)
    # Four explicit preparation steps lead to two complementary mining views.
    upper_y, upper_h = 6.91, 1.77
    x_positions = [.38, 4.21, 8.04, 11.87]
    width = 3.35
    for x in x_positions:
        _box(ax, x, upper_y, width, upper_h, color="#bbc9c1", fill="#f6f8f5")
    _text(ax, .64, 8.40, "Historical USDA records", 12, weight="bold")
    _leaf(ax, .97, 7.60, .52)
    _text(ax, 1.45, 7.88, "taxon, region, region, ...", 9.8)
    _text(ax, 1.45, 7.52, "recorded plant presence", 9.4, color=MUTED)
    _text(ax, 1.45, 7.20, "US states + Canadian provinces", 8.8, color=MUTED)

    _text(ax, 4.47, 8.40, "Normalize rank & region codes", 11.3, weight="bold")
    for yy, line in zip((7.97, 7.59, 7.21),
                        ("Remove genus summary rows",
                         "Merge infraspecific records",
                         "Correct codes; merge NF + LB")):
        ax.add_patch(Circle((4.53, yy), .035, facecolor=FOREST, edgecolor="none"))
        _text(ax, 4.66, yy, line, 9.2)

    _text(ax, 8.30, 8.40, "Species × 64-region matrix", 11.5, weight="bold")
    miniature = np.array([[1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0],
                          [0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 0, 0]])
    _matrix(ax, 8.35, 7.99, miniature, cell=.225)
    _text(ax, 9.94, 7.81, "1  recorded", 8.9, color=FOREST)
    _text(ax, 9.94, 7.45, "0  unrecorded", 8.9, color=MUTED)

    _text(ax, 12.13, 8.40, "Freeze discovery / validation", 11.1, weight="bold")
    for i in range(10):
        color = FOREST if i < 8 else OCHRE
        _box(ax, 12.18 + i * .263, 7.72, .20, .26, color=color, fill=color, radius=.035)
    _text(ax, 12.18, 7.38, "80% training", 9.5, color=FOREST, weight="bold")
    _text(ax, 13.72, 7.38, "20% hold-out", 9.5, color=OCHRE, weight="bold")
    for a, b in zip(x_positions[:-1], x_positions[1:]):
        _arrow(ax, (a + width + .05, 7.80), (b - .07, 7.80))

    branch_y, branch_h = 3.91, 2.11
    _box(ax, .80, branch_y, 6.53, branch_h, color=FOREST, fill="#edf4ef")
    _box(ax, 8.28, branch_y, 6.53, branch_h, color=BLUE, fill="#edf2f5")
    # Arrows originate from training only; held-out plants remain untouched.
    ax.plot([12.65, 12.65, 4.05], [6.91, 6.53, 6.53], color=FOREST, lw=1.5)
    _arrow(ax, (4.05, 6.53), (4.05, 6.09), color=FOREST)
    _arrow(ax, (11.52, 6.53), (11.52, 6.09), color=FOREST)
    _text(ax, 7.60, 6.60, "Training species", 9.0, ha="center", color=FOREST,
          bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 2})

    _text(ax, 1.09, 5.68, "Region association mining", 13, weight="bold", color=FOREST)
    _text(ax, 1.09, 5.25, "FP-growth  |  itemset length ≤ 3", 10.2)
    _text(ax, 1.09, 4.85, "Filter by support, confidence and lift", 10.0)
    _text(ax, 1.09, 4.32, "Single-region and two-region antecedents", 9.1, color=MUTED)
    _rules_glyph(ax, 6.39, 4.91, .96)

    _text(ax, 8.56, 5.68, "Regional composition clustering", 12.5, weight="bold", color=BLUE)
    _text(ax, 8.56, 5.25, "Jaccard distance  |  average linkage", 10.2)
    _text(ax, 8.56, 4.85, "Choose k by silhouette (k = 4–9)", 10.0)
    _text(ax, 8.56, 4.32, "Groups of regions with shared recorded flora", 9.1, color=MUTED)
    _dendrogram_glyph(ax, 13.89, 4.93, .97)

    _arrow(ax, (4.05, 3.85), (4.05, 3.23), color=FOREST)
    _arrow(ax, (11.52, 3.85), (11.52, 3.23), color=BLUE)
    _box(ax, .38, .42, 14.84, 2.72, color="#bbc9c1", fill="#f8f9f6")
    _text(ax, .71, 2.80, "Reliability checks and extension experiments", 12, weight="bold")
    checks = [
        ("Replication & record completion", "Random / genus splits; hide one recorded region", FOREST),
        ("Bootstrap stability", "Rule recurrence + cluster co-assignment", BLUE),
        ("Broad-distribution sensitivity", "Remove the broadest 5% of species", OCHRE),
        ("Algorithm agreement & efficiency", "Apriori vs FP-growth: same itemsets", FOREST),
        ("Distance / linkage sensitivity", "Jaccard vs cosine; average vs complete", BLUE),
        ("Spatial permutation test", "Composition distance vs geographic distance", OCHRE),
    ]
    for i, (label, detail, color) in enumerate(checks):
        col, row = i % 3, i // 3
        bx, by = .70 + col * 4.78, 1.60 - row * 1.01
        _box(ax, bx, by, 4.47, .82, color="#dbe3db", fill=PAPER, radius=.09)
        ax.plot([bx + .14, bx + .14], [by + .19, by + .64], color=color, lw=2.6)
        _text(ax, bx + .32, by + .56, label, 10.1, weight="bold", color=color)
        _text(ax, bx + .32, by + .24, detail, 8.8, color=MUTED)
    return _save(fig, output_dir, "method_framework")


def _example_records(results_dir: Path | None):
    """Get genuine presence values without depending on preprocessing modules."""
    records: dict[str, set[str]] = {}
    if results_dir is not None:
        dataset_file = results_dir / "dataset.npz"
        if dataset_file.exists():
            with np.load(dataset_file, allow_pickle=False) as dataset:
                names = np.char.lower(dataset["names"].astype(str))
                regions = np.char.lower(dataset["regions"].astype(str))
                matrix = dataset["matrix"]
                for name in EXAMPLES:
                    ids = np.flatnonzero(names == name)
                    if len(ids):
                        present = np.asarray(matrix[ids[0]], dtype=bool)
                        records[name] = set(regions[present].tolist())
    if len(records) < len(EXAMPLES):
        raw_path = Path(__file__).resolve().parents[1] / "plantsdata" / "plants.data"
        if raw_path.exists():
            for row in raw_path.read_text(encoding="latin-1").splitlines():
                fields = row.strip().split(",")
                if fields[0] in EXAMPLES and fields[0] not in records:
                    records[fields[0]] = set(fields[1:])
    if len(records) < len(EXAMPLES):
        return records, None
    values = np.array([[int(region in records[name]) for region in EXAMPLE_REGIONS]
                       for name in EXAMPLES], dtype=np.int8)
    return records, values


def _schematic_geography(ax, x, y, width, height):
    """Geographic concept glyph, deliberately marked schematic by caller."""
    # Stylized mainland silhouette, deliberately without inferred boundaries.
    shape = np.array([
        [.10, .83], [.20, .91], [.40, .85], [.46, .97], [.57, .89],
        [.66, .95], [.82, .80], [.91, .69], [.86, .53], [.75, .45],
        [.85, .38], [.69, .34], [.73, .12], [.67, .19], [.59, .28],
        [.43, .25], [.34, .18], [.27, .39], [.17, .42], [.13, .64],
    ])
    ax.add_patch(Polygon(shape * np.array([width, height]) + np.array([x, y]),
                          closed=True, facecolor="#e9eeea", edgecolor="#bdcabe", lw=1))
    groups = [(.28, .66, FOREST), (.34, .39, OCHRE), (.72, .53, BLUE)]
    for gx, gy, color in groups:
        for dx, dy in ((-.035, .02), (.025, .04), (.02, -.04), (-.035, -.05)):
            ax.add_patch(Circle((x + (gx + dx) * width, y + (gy + dy) * height),
                                .05 * min(width, height), facecolor=color,
                                edgecolor=PAPER, lw=.7, alpha=.90))


def _lambert(coordinates):
    """Lambert conformal conic coordinates, standard parallels 33 / 62 deg."""
    coordinates = np.asarray(coordinates, dtype=float)
    lon = np.deg2rad(coordinates[:, 0])
    lat = np.deg2rad(coordinates[:, 1])
    phi1, phi2, phi0 = np.deg2rad([33., 62., 45.])
    n = np.log(np.cos(phi1) / np.cos(phi2)) / np.log(
        np.tan(np.pi / 4 + phi2 / 2) / np.tan(np.pi / 4 + phi1 / 2))
    scale = np.cos(phi1) * np.tan(np.pi / 4 + phi1 / 2) ** n / n
    rho = scale / np.tan(np.pi / 4 + lat / 2) ** n
    rho0 = scale / np.tan(np.pi / 4 + phi0 / 2) ** n
    angle = n * (lon - np.deg2rad(-100.))
    return np.column_stack((rho * np.sin(angle), rho0 - rho * np.cos(angle)))


def _actual_geography(ax, x, y, width, height, results_dir):
    """Use the bundled Natural Earth polygons and fitted cluster assignments."""
    assets = Path(__file__).resolve().parents[1] / "assets"
    candidates = sorted(assets.glob("**/*.geojson"))
    if not candidates:
        return None
    features = json.loads(candidates[0].read_text(encoding="utf-8"))["features"]
    clusters = {}
    if results_dir is not None and (results_dir / "regions.csv").exists():
        with (results_dir / "regions.csv").open(encoding="utf-8-sig", newline="") as stream:
            clusters = {row["code"].upper(): int(row["cluster"])
                        for row in csv.DictReader(stream) if "cluster" in row}
    from .maps import CLUSTER_COLORS
    palette = CLUSTER_COLORS
    shapes = []
    for feature in features:
        properties = feature.get("properties", {})
        country = properties.get("adm0_a3", properties.get("iso_a2", ""))
        if country not in ("USA", "CAN", "US", "CA"):
            continue
        iso = str(properties.get("iso_3166_2", ""))
        code = iso.rsplit("-", 1)[-1] if "-" in iso else properties.get("postal", "")
        code = str(code).upper()
        if clusters and code not in clusters:
            continue
        geometry = feature.get("geometry") or {}
        polygons = geometry.get("coordinates", [])
        if geometry.get("type") == "Polygon":
            polygons = [polygons]
        elif geometry.get("type") != "MultiPolygon":
            continue
        for rings in polygons:
            if not rings:
                continue
            shapes.append((code, [_lambert(ring) for ring in rings]))
    mainland = [(code, rings) for code, rings in shapes if code not in ("AK", "HI")]
    if not mainland:
        return None
    points = np.concatenate([rings[0] for _, rings in mainland])
    low, high = points.min(axis=0), points.max(axis=0)
    padding = .10
    scale = min((width - 2 * padding) / (high[0] - low[0]),
                (height - 2 * padding) / (high[1] - low[1]))
    drawn = (high - low) * scale
    origin = np.array([x + (width - drawn[0]) / 2, y + (height - drawn[1]) / 2])
    for code, rings in mainland:
        color = palette[clusters.get(code, 0) % len(palette)] if clusters else "#d2ded3"
        for i, ring in enumerate(rings):
            ax.add_patch(Polygon((ring - low) * scale + origin, closed=True,
                                  facecolor=color if i == 0 else "#f7f9f5",
                                  edgecolor=PAPER if i == 0 else "none", lw=.30))
    # Compact inset symbols preserve the two noncontiguous US state memberships.
    for j, code in enumerate(("AK", "HI")):
        color = palette[clusters.get(code, 0) % len(palette)] if clusters else FOREST
        bx = x + .05 + j * .59
        _box(ax, bx, y + .02, .49, .26, color=color, fill=PAPER, radius=.035)
        ax.add_patch(Circle((bx + .11, y + .15), .042, facecolor=color, edgecolor="none"))
        _text(ax, bx + .29, y + .15, code, 7.4, color=color, ha="center")
    return "fitted" if clusters else "outline"


def _topic_overview(output_dir: Path, results_dir: Path | None) -> list[Path]:
    fig, ax = _canvas(15.6, 7.8)
    panels = [(.38, 4.52, FOREST), (5.54, 4.52, BLUE), (10.70, 4.52, OCHRE)]
    for x, width, color in panels:
        _box(ax, x, .94, width, 6.48, color="#ccd8ce", fill="#f7f9f5")
        ax.plot([x + .26, x + 1.1], [6.89, 6.89], color=color, lw=3)
    _text(ax, .69, 7.13, "Observed plant distributions", 12.4, weight="bold", color=FOREST)
    _text(ax, 5.85, 7.13, "Compare regional flora", 12.5, weight="bold", color=BLUE)
    _text(ax, 11.01, 7.13, "Discover geographic patterns", 12.0, weight="bold", color=OCHRE)
    _arrow(ax, (4.98, 4.07), (5.42, 4.07), color=MUTED)
    _arrow(ax, (10.14, 4.07), (10.58, 4.07), color=MUTED)
    records, values = _example_records(results_dir)
    kinds = ("conifer", "oak", "cactus")
    colors = (FOREST, BLUE, OCHRE)
    region_labels = ["Pacific coast", "Eastern North America", "Southwestern states"]
    for i, (name, kind, color, description) in enumerate(zip(EXAMPLES, kinds, colors, region_labels)):
        cy = 5.83 - 1.68 * i
        _box(ax, .66, cy - .70, 3.94, 1.38, color="#dce5dc", fill=PAPER, radius=.12)
        _leaf(ax, 1.14, cy - .02, .57, color=color, kind=kind)
        _text(ax, 1.68, cy + .33, name.capitalize(), 10.3, weight="medium", style="italic")
        _text(ax, 1.68, cy - .02, description, 8.9, color=MUTED)
        badges = sorted(records.get(name, set()))
        selected = [r for r in EXAMPLE_REGIONS if r in badges][:5]
        if len(selected) < 3:
            selected = badges[:4]
        for j, code in enumerate(selected):
            bx = 1.68 + j * .46
            _box(ax, bx, cy - .46, .38, .23, color=color, fill="#f0f4ef", radius=.035)
            _text(ax, bx + .19, cy - .34, code.upper(), 7.6, color=color, ha="center")
    _text(ax, .83, 1.30, "Real example records; selected region codes shown", 8.5, color=MUTED)

    _text(ax, 5.85, 6.46, "Shared species connect regions", 10.0, color=MUTED)
    _text(ax, 5.85, 5.95, "Example presence matrix", 10.6, weight="bold", color=BLUE)
    if values is not None:
        _matrix(ax, 6.05, 5.46, values, cell=.425,
                colors={0: "#e1e8e1", 1: FOREST}, labels=[r.upper() for r in EXAMPLE_REGIONS])
        for i, label in enumerate(("Fir", "Oak", "Cactus")):
            _text(ax, 6.04, 5.26 - i * .425, label, 8.2, ha="right")
    else:
        _text(ax, 6.05, 5.0, "Supply plants.data to display real records", 9.2, color=MUTED)
    ax.add_patch(Circle((6.15, 3.71), .06, facecolor=FOREST, edgecolor="none"))
    _text(ax, 6.33, 3.71, "recorded", 9.0)
    ax.add_patch(Circle((7.71, 3.71), .06, facecolor="#e1e8e1", edgecolor="none"))
    _text(ax, 7.89, 3.71, "unrecorded", 9.0)
    _arrow(ax, (7.80, 3.38), (7.80, 2.95), color=BLUE)
    # Pairwise overlap is a genuine local illustration, not a fitted cluster.
    _text(ax, 5.85, 2.67, "Regional composition similarity", 10.5, weight="bold", color=BLUE)
    for i, yy in enumerate((2.19, 1.77)):
        left_codes = ("WA", "AZ")
        right_codes = ("OR", "UT")
        color = FOREST if i == 0 else OCHRE
        for xx, label in ((6.48, left_codes[i]), (9.02, right_codes[i])):
            _box(ax, xx - .34, yy - .15, .68, .30, color=color, fill=PAPER, radius=.055)
            _text(ax, xx, yy, label, 9.2, ha="center", color=color)
        ax.plot([6.88, 8.62], [yy, yy], color=color, lw=1.7)
        _text(ax, 7.75, yy + .12, "shared records", 8.3, ha="center", color=MUTED,
              bbox={"facecolor": "#f7f9f5", "edgecolor": "none", "pad": 1})
    _text(ax, 5.85, 1.30, "Similarity uses the retained training species", 8.4, color=MUTED)

    _text(ax, 11.02, 6.44, "Region association rules", 10.8, weight="bold", color=FOREST)
    _rules_glyph(ax, 12.97, 5.58, 1.17)
    _text(ax, 12.95, 4.83, "A → C     {A, B} → C", 10.2, ha="center", color=FOREST)
    _text(ax, 11.02, 4.34, "Regions with similar composition", 10.5, weight="bold", color=BLUE)
    actual_map = _actual_geography(ax, 11.00, 1.64, 3.91, 2.42, results_dir)
    if not actual_map:
        _schematic_geography(ax, 11.00, 1.64, 3.91, 2.42)
    if actual_map == "fitted":
        label = "Example regional groups: contiguous US + Canada"
    elif actual_map:
        label = "US + Canada geography (no fitted groups loaded)"
    else:
        label = "Schematic view of regional groups"
    _text(ax, 12.98, 1.44, label, 8.3, color=MUTED, ha="center")
    _text(ax, .55, .43,
          "Historical recorded presence supports interpretable association rules and regional groupings.",
          10.0, color=MUTED)
    return _save(fig, output_dir, "plant_topic_overview")


def draw_diagrams(output_dir: Path = Path("outputs/figures"),
                  results_dir: Path | None = Path("outputs/results")) -> list[Path]:
    """Generate the framework and study overview, each in PNG, SVG and PDF."""
    _style()
    output_dir = Path(output_dir)
    results_dir = Path(results_dir) if results_dir is not None else None
    return _framework(output_dir) + _topic_overview(output_dir, results_dir)
