"""Publication figures for optional experiments, read only from saved results.

The completion flag is checked before any artifact is read: files from an older
run must never be presented as extensions of an incomplete current run.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
from scipy.cluster.hierarchy import leaves_list

from .maps import CLUSTER_COLORS

GREEN, BLUE, OCHRE, MAUVE = "#2f6b57", "#527a94", "#c28e39", "#926d91"
PALETTE = CLUSTER_COLORS
DENSITY = LinearSegmentedColormap.from_list("plant_density", ["#f1f6f3", "#90b1a2", GREEN])


def _read(results: Path, name: str) -> pd.DataFrame | None:
    path = results / name
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else None


def _json(results: Path, name: str) -> dict:
    path = results / name
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}


def _truth(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin(["true", "1"])


def _panel(ax, label: str):
    ax.text(0, 1.035, label, transform=ax.transAxes, fontweight="bold", fontsize=11, color="#263a31")


def _save(fig, out: Path, name: str) -> list[Path]:
    # Local import avoids a cycle when the main plots module invokes this module.
    from .plots import save_figure

    return save_figure(fig, out, name)


def _violin(ax, values, color: str, ylabel: str, panel: str):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return
    if len(values) > 1 and np.ptp(values) > 0:
        parts = ax.violinplot(values, positions=[0], widths=.65, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(.22)
    jitter = np.random.default_rng(2026).uniform(-.12, .12, len(values))
    ax.scatter(jitter, values, s=20, alpha=.7, color=color, linewidths=.3, edgecolors="white", zorder=3)
    median = float(np.median(values))
    ax.plot([-.28, .28], [median, median], color=color, linewidth=2, zorder=4)
    ax.set_xlim(-.58, .58)
    ax.set_xticks([])
    ax.set_ylabel(ylabel)
    ax.yaxis.set_label_position("right")
    ax.text(.97, .04, f"Median {median:.3f}\nn = {len(values)} bootstraps", ha="right", va="bottom", transform=ax.transAxes, fontsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="y", alpha=.15)
    _panel(ax, panel)


def _bootstrap(results: Path, out: Path) -> list[Path]:
    required = ["bootstrap_coassignment.npy", "regions.csv", "region_linkage.npy"]
    if not all((results / name).exists() for name in required):
        return []
    regions = _read(results, "regions.csv")
    coassignment = np.load(results / required[0], allow_pickle=False)
    tree = np.load(results / required[2], allow_pickle=False)
    if coassignment.shape != (len(regions), len(regions)):
        raise ValueError("Bootstrap co-assignment does not match region metadata.")
    order = leaves_list(tree)
    ordered = regions.iloc[order]
    stability = _read(results, "bootstrap_clusters.csv")
    fig = plt.figure(figsize=(11.1, 7.1))
    grid = fig.add_gridspec(3, 3, width_ratios=[7.0, .25, 2.7], height_ratios=[.20, 3.0, 3.0], wspace=.43, hspace=.4)
    strip = fig.add_subplot(grid[0, 0])
    clusters = ordered.cluster.to_numpy(dtype=int)
    strip.imshow((clusters % len(PALETTE))[None, :], aspect="auto", cmap=ListedColormap(PALETTE), vmin=-.5, vmax=len(PALETTE)-.5)
    strip.set_xticks([])
    strip.set_yticks([])
    for spine in strip.spines.values():
        spine.set_visible(False)
    strip.text(0, 1.7, "a", fontweight="bold", fontsize=11, transform=strip.transAxes, color="#263a31")
    ax = fig.add_subplot(grid[1:, 0])
    im = ax.imshow(coassignment[np.ix_(order, order)], cmap=DENSITY, vmin=0, vmax=1, interpolation="none", aspect="equal")
    ax.set_xticks(np.arange(len(ordered)), ordered.code, rotation=90, fontsize=5.8)
    ax.set_yticks(np.arange(len(ordered)), ordered.code, fontsize=5.8)
    ax.tick_params(length=0, pad=2)
    ax.set_xlabel("Region · baseline dendrogram order")
    ax.set_ylabel("Region · baseline dendrogram order")
    cax = fig.add_subplot(grid[1:, 1])
    fig.colorbar(im, cax=cax, label="Bootstrap co-assignment probability")
    handles = [Patch(facecolor=PALETTE[int(k) % len(PALETTE)], label=f"Baseline group {int(k)+1}") for k in sorted(regions.cluster.unique())]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.10, .015), frameon=False, ncol=min(4, len(handles)), fontsize=8)
    if stability is not None and len(stability):
        _violin(fig.add_subplot(grid[:2, 2]), stability.ari_to_baseline, BLUE, "Adjusted Rand index", "b")
        _violin(fig.add_subplot(grid[2, 2]), stability.silhouette, OCHRE, "Jaccard silhouette", "c")
    fig.subplots_adjust(left=.065, right=.96, top=.93, bottom=.13)
    fig.canvas.draw()
    heatmap_position = ax.get_position()
    strip.set_position([heatmap_position.x0, heatmap_position.y1+.014, heatmap_position.width, .019])
    return _save(fig, out, "08_bootstrap_stability")


def _rule_stability(results: Path, out: Path) -> list[Path]:
    bootstrap = _read(results, "bootstrap_rules.csv")
    core = _read(results, "core_rules.csv")
    if bootstrap is None or core is None or bootstrap.empty or core.empty:
        return []
    candidates = core.loc[core.antecedent_size.eq(1), ["id", "antecedent", "consequent", "support", "lift", "validation_pass"]]
    candidates = candidates.loc[_truth(candidates.validation_pass)].merge(bootstrap, on="id", validate="one_to_one")
    candidates = candidates.loc[candidates.recurrence.ge(.8)].sort_values(["lift", "support"], ascending=False)
    selected, pairs, destinations = [], set(), {}
    for _, row in candidates.iterrows():
        pair = tuple(sorted((row.antecedent, row.consequent)))
        if pair in pairs or destinations.get(row.consequent, 0) >= 2:
            continue
        selected.append(row)
        pairs.add(pair)
        destinations[row.consequent] = destinations.get(row.consequent, 0) + 1
        if len(selected) == 14:
            break
    if not selected:
        return []
    data = pd.DataFrame(selected).iloc[::-1].reset_index(drop=True)
    y = np.arange(len(data))
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 6.2), sharey=True, gridspec_kw={"width_ratios": [3.6, 3.1, 1.7], "wspace": .18})
    for ax, metric, color, letter in zip(axes[:2], ["confidence", "lift"], [GREEN, BLUE], ["a", "b"]):
        median = data[f"{metric}_median"].to_numpy()
        low = data[f"{metric}_low"].to_numpy()
        high = data[f"{metric}_high"].to_numpy()
        ax.errorbar(median, y, xerr=np.vstack([np.maximum(median-low, 0), np.maximum(high-median, 0)]), fmt="o", color=color, markersize=4.2, capsize=2, linewidth=1.25)
        ax.set_xlabel(f"Bootstrap {metric}")
        ax.grid(axis="x", alpha=.16)
        ax.set_ylim(-.65, len(data)-.35)
        _panel(ax, letter)
    axes[0].set_yticks(y, [f"{row.antecedent} → {row.consequent}" for row in data.itertuples()], fontsize=9)
    axes[0].set_xlim(max(0, float(data.confidence_low.min())-.04), 1.01)
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].tick_params(axis="y", left=False)
    axes[2].barh(y, data.recurrence, height=.5, color=OCHRE, alpha=.8)
    for yi, value in zip(y, data.recurrence):
        axes[2].text(value+.025, yi, f"{value:.0%}", va="center", fontsize=8)
    axes[2].set_xlim(0, 1.3)
    axes[2].set_xticks([0, .5, 1], ["0", "50", "100"])
    axes[2].set_xlabel("Recurrence (%)")
    axes[2].tick_params(axis="y", left=False)
    _panel(axes[2], "c")
    fig.text(.12, .04, "Single-antecedent rules passing held-out validation; recurrence ≥80%.\nRanked by baseline training lift, then support; points: median, intervals: bootstrap 2.5–97.5 percentiles.", fontsize=8, color="#63766c")
    fig.subplots_adjust(left=.12, right=.98, top=.93, bottom=.17)
    return _save(fig, out, "09_rule_stability")


def _algorithm(results: Path, out: Path) -> list[Path]:
    data = _read(results, "algorithm_benchmark.csv")
    if data is None or data.empty:
        return []
    supports = sorted(data.min_support.unique())
    fig, axes = plt.subplots(1, len(supports), figsize=(10.7, 3.8), squeeze=False, sharey=True)
    axes = axes.ravel()
    algorithm_labels = {"apriori": "Apriori", "fpgrowth": "FP-growth"}
    for index, (ax, support) in enumerate(zip(axes, supports)):
        subset = data.loc[data.min_support.eq(support)]
        for algorithm, color, marker in [("apriori", GREEN, "o"), ("fpgrowth", BLUE, "s")]:
            current = subset.loc[subset.algorithm.eq(algorithm)]
            if current.empty:
                continue
            stats = current.groupby("n_plants").seconds.agg(["median", "min", "max"])
            x = stats.index.to_numpy()
            mid = stats["median"].to_numpy()
            ax.errorbar(x, mid, yerr=np.vstack([mid-stats["min"], stats["max"]-mid]), color=color, marker=marker, markersize=4, capsize=3, linewidth=1.5, label=algorithm_labels[algorithm])
        ax.set_yscale("log")
        ax.set_xticks(sorted(subset.n_plants.unique()))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:g}k"))
        ax.set_xlabel("Training species")
        ax.grid(axis="y", which="both", alpha=.16)
        ax.text(.05, .96, f"Minimum support {support:.0%}", transform=ax.transAxes, ha="left", va="top", fontsize=9)
        _panel(ax, chr(97+index))
    axes[0].set_ylabel("Runtime (s; logarithmic scale)")
    axes[-1].legend(loc="center right", frameon=False, fontsize=8)
    equivalent = bool(_truth(data.equivalent).all())
    repeats = data.groupby(["algorithm", "min_support", "n_plants"]).size()
    repeat_note = f"{int(repeats.iloc[0])} repeats per setting" if repeats.nunique() == 1 else "repeats vary by setting"
    fig.text(.075, .03, f"Markers: median; bars: observed repeat range ({repeat_note}). " + ("Identical itemsets at every setting." if equivalent else "Itemset equivalence is not satisfied at every setting."), fontsize=8, color="#63766c")
    fig.subplots_adjust(left=.075, right=.98, top=.91, bottom=.22, wspace=.2)
    return _save(fig, out, "10_algorithm_efficiency")


def _hexbin(ax, x, y, xlabel: str, ylabel: str, panel: str, *, identity=False, change=False):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if not len(x):
        return None
    artist = ax.hexbin(x, y, gridsize=42, mincnt=1, bins="log", cmap=DENSITY, linewidths=0, rasterized=True)
    if identity:
        bounds = [min(x.min(), y.min()), max(x.max(), y.max())]
        margin = max((bounds[1]-bounds[0])*.025, .01)
        limits = (bounds[0]-margin, bounds[1]+margin)
        ax.plot(limits, limits, linestyle="--", linewidth=1, color=MAUVE)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
    if change:
        ax.axhline(0, color=MAUVE, linestyle="--", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _panel(ax, panel)
    return artist


def _genus_validation(results: Path, out: Path) -> list[Path]:
    matched = _read(results, "matched_split_rules.csv")
    if matched is None or matched.empty:
        return []
    grouped = _read(results, "grouped_validation_rules.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.6))
    for ax, metric, letter in zip(axes, ["confidence", "lift"], ["a", "b"]):
        image = _hexbin(ax, matched[f"validation_{metric}_random"], matched[f"validation_{metric}_grouped"], f"Random-split validation {metric}", f"Genus-split validation {metric}", letter, identity=True)
        if image is not None:
            fig.colorbar(image, ax=ax, fraction=.045, pad=.025, label="Rules per hexagon")
        delta = matched[f"validation_{metric}_grouped"]-matched[f"validation_{metric}_random"]
        ax.text(.04, .96, f"Median paired change {delta.median():+.3f}", transform=ax.transAxes, va="top", fontsize=8)
    family_note = f"Matched rule intersection: n = {len(matched):,}; dashed line: identical validation metrics."
    if grouped is not None:
        passed = int(_truth(grouped.validation_pass).sum()) if "validation_pass" in grouped else None
        family_note += f"\nIndependent genus-split discovery: {len(grouped):,} rules" + (f", {passed:,} passing validation." if passed is not None else ".")
    fig.text(.075, .035, family_note, fontsize=8, color="#63766c")
    fig.subplots_adjust(left=.075, right=.96, top=.93, bottom=.20, wspace=.38)
    return _save(fig, out, "11_genus_validation")


def _broad_species(results: Path, out: Path) -> list[Path]:
    data = _read(results, "broad_species_rules.csv")
    if data is None or data.empty:
        return []
    summary = _json(results, "broad_species_summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.5))
    for ax, metric, letter in zip(axes, ["confidence", "lift"], ["a", "b"]):
        delta = data[f"filtered_{metric}"]-data[f"baseline_{metric}"]
        image = _hexbin(ax, data[f"baseline_{metric}"], delta, f"Baseline training {metric}", f"Change in training {metric}", letter, change=True)
        if image is not None:
            fig.colorbar(image, ax=ax, fraction=.045, pad=.025, label="Rules per hexagon")
        ax.text(.04, .96, f"Median change {delta.median():+.3f}", transform=ax.transAxes, va="top", fontsize=8)
    retention = summary.get("rule_retention_fraction", float(_truth(data.retained).mean()))
    note = f"Fixed baseline rule family: n = {len(data):,}; {retention:.1%} remain above all discovery thresholds."
    if "removed_fraction" in summary:
        note += f"\nRemoved broadest {summary['removed_fraction']:.0%} of training species"
        if "removed_n" in summary:
            note += f" (n = {summary['removed_n']:,})"
        if "ari" in summary:
            note += f"; cluster ARI = {summary['ari']:.3f}"
        note += ". Dashed line: no metric change."
    fig.text(.075, .035, note, fontsize=8, color="#63766c")
    fig.subplots_adjust(left=.075, right=.96, top=.93, bottom=.22, wspace=.40)
    return _save(fig, out, "12_broad_species_sensitivity")


def _metric_support(results: Path, out: Path) -> list[Path]:
    metric = _read(results, "metric_sensitivity.csv")
    support = _read(results, "support_sensitivity.csv")
    has_metric = metric is not None and not metric.empty
    has_support = support is not None and not support.empty
    if not has_metric and not has_support:
        return []
    rows = 2 if has_metric and has_support else 1
    fig, axes = plt.subplots(rows, 2, figsize=(10.4, 6.9 if rows == 2 else 3.9), squeeze=False)
    panel_index = 0
    row = 0
    if has_metric:
        labels = [f"{entry.setting.replace(' / ', ' · ').capitalize()}  (k={entry.k})" for entry in metric.itertuples()]
        y = np.arange(len(metric))[::-1]
        for ax, column, xlabel in zip(axes[0], ["ari_to_baseline", "silhouette"], ["Adjusted Rand index to baseline", "Silhouette in the specified distance"]):
            for index, (yi, value) in enumerate(zip(y, metric[column])):
                ax.axhline(yi, color="#dce5df", linewidth=.8, zorder=0)
                ax.scatter(value, yi, color=PALETTE[index % len(PALETTE)], s=50, marker="s" if index == 0 else "o", zorder=3)
                ax.annotate(f"{value:.3f}", (value, yi), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8)
            ax.set_yticks(y, labels if column == "ari_to_baseline" else [""]*len(y), fontsize=8)
            ax.set_ylim(-.65, len(metric)-.35)
            ax.set_xlim(min(0, metric[column].min()-.1), max(1.12 if column == "ari_to_baseline" else .40, metric[column].max()*1.18))
            ax.set_xlabel(xlabel)
            ax.tick_params(axis="y", length=0)
            _panel(ax, chr(97+panel_index))
            panel_index += 1
        row = 1
    if has_support:
        support = support.sort_values("min_support")
        x = support.min_support.to_numpy()
        for column, label, color, marker in [("n_itemsets", "Frequent itemsets", GREEN, "o"), ("n_rules", "Discovered rules", BLUE, "s")]:
            axes[row if rows == 2 else 0, 0].plot(x, support[column], color=color, marker=marker, linewidth=1.5, markersize=4, label=label)
        count_ax = axes[row if rows == 2 else 0, 0]
        count_ax.set_yscale("log")
        count_ax.set_ylabel("Pattern count (logarithmic scale)")
        count_ax.legend(frameon=False, fontsize=8, loc="upper right")
        validation_ax = axes[row if rows == 2 else 0, 1]
        for column, label, color, marker in [("validated_fraction", "Rules passing validation", GREEN, "o"), ("mean_validation_confidence", "Mean validation confidence", OCHRE, "s")]:
            validation_ax.plot(x, support[column], color=color, marker=marker, linewidth=1.5, markersize=4, label=label)
            for xi, value in zip(x, support[column]):
                validation_ax.annotate(f"{value:.1%}", (xi, value), xytext=(0, 7 if column == "validated_fraction" else -14), textcoords="offset points", ha="center", fontsize=7.5, color=color)
        validation_ax.set_ylim(max(0, float(support[["validated_fraction", "mean_validation_confidence"]].min().min())-.07), 1.025)
        validation_ax.yaxis.set_major_formatter(PercentFormatter(1))
        validation_ax.set_ylabel("Held-out validation metric")
        validation_ax.legend(frameon=False, fontsize=8, loc="lower right")
        for ax in (count_ax, validation_ax):
            ax.set_xlabel("Minimum support")
            ax.set_xticks(x)
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis="y", alpha=.16)
            _panel(ax, chr(97+panel_index))
            panel_index += 1
    fig.text(.25, .025, "Silhouettes use each setting's own distance; square in the upper row marks the baseline." if has_metric else "Discovery confidence and lift thresholds are held fixed while minimum support changes.", fontsize=8, color="#63766c")
    fig.subplots_adjust(left=.25 if has_metric else .08, right=.96, top=.94, bottom=.11 if rows == 2 else .19, wspace=.45, hspace=.5)
    return _save(fig, out, "13_metric_support_sensitivity")


def _spatial(results: Path, out: Path) -> list[Path]:
    pairs = _read(results, "spatial_pairs.csv")
    if pairs is None or pairs.empty:
        return []
    stats = _json(results, "spatial_test.json")
    fig, ax = plt.subplots(figsize=(7.3, 5.0))
    image = _hexbin(ax, pairs.geo_distance_km, pairs.floristic_distance, "Geographic distance between region reference points (km)", "Floristic Jaccard dissimilarity", "a")
    if image is not None:
        fig.colorbar(image, ax=ax, fraction=.045, pad=.035, label="Region pairs per hexagon")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, max(pairs.geo_distance_km)*1.02)
    annotation = f"n = {len(pairs):,} region pairs"
    if "spearman_r" in stats:
        annotation += f"\nSpearman ρ = {stats['spearman_r']:.3f}"
    if "permutation_p" in stats:
        annotation += f"\nRegion-label permutation P = {stats['permutation_p']:.3g}"
    if "n_permutations" in stats:
        annotation += f"\n{stats['n_permutations']:,} permutations"
    ax.text(.035, .96, annotation, transform=ax.transAxes, va="top", fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 4})
    fig.text(.12, .025, "Permutation preserves dependence among region pairs by jointly permuting region labels.\nApproximate administrative reference points represent geography.", fontsize=8, color="#63766c")
    fig.subplots_adjust(left=.12, right=.91, top=.93, bottom=.19)
    return _save(fig, out, "14_spatial_relation")


def generate_extension_figures(results_dir: Path, figures_dir: Path) -> list[Path]:
    """Save optional experiment figures in PNG (300 dpi), SVG and PDF formats."""
    results_dir, figures_dir = Path(results_dir), Path(figures_dir)
    if not _json(results_dir, "run_summary.json").get("extensions_complete", False):
        return []
    from .plots import setup_style

    setup_style()
    paths: list[Path] = []
    for plot in (_bootstrap, _rule_stability, _algorithm, _genus_validation, _broad_species, _metric_support, _spatial):
        paths.extend(plot(results_dir, figures_dir))
    return paths
