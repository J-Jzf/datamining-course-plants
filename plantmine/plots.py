"""Publication figures from saved experiment artifacts, without figure titles."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from scipy.cluster.hierarchy import dendrogram, leaves_list

from .maps import CLUSTER_COLORS, map_with_insets

GREEN, BLUE, OCHRE = CLUSTER_COLORS[:3]


def setup_style():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.labelsize":10,
        "axes.spines.top":False, "axes.spines.right":False, "axes.edgecolor":"#87978f",
        "axes.labelcolor":"#263a31", "xtick.color":"#46584e", "ytick.color":"#46584e",
        "figure.facecolor":"white", "axes.facecolor":"white", "svg.fonttype":"none",
        "pdf.fonttype":42, "savefig.facecolor":"white"})


def save_figure(fig, figures_dir, stem):
    figures_dir = Path(figures_dir); figures_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = figures_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.08)
        paths.append(path)
    plt.close(fig)
    return paths


def _panel(ax, label):
    ax.text(0, 1.025, label, transform=ax.transAxes, fontsize=10, fontweight="bold", color="#263a31")


def _select_rules(rules, limit=16, stable=None, require_validation=True):
    selected = rules[rules.antecedent_size.eq(1)].copy()
    if require_validation and "validation_pass" in selected:
        selected = selected[selected.validation_pass.astype(str).str.lower().eq("true")]
    if stable is not None:
        selected = selected.merge(stable[["id", "recurrence"]], on="id", how="left")
        selected = selected[selected.recurrence.ge(.8)]
    selected = selected.sort_values(["lift", "support"], ascending=False)
    # Freeze displayed examples from training statistics; cap destination repetition.
    rows, used_pairs, endpoints = [], set(), {}
    for _, row in selected.iterrows():
        pair = tuple(sorted((row.antecedent, row.consequent)))
        if pair in used_pairs or endpoints.get(row.consequent, 0) >= 2:
            continue
        rows.append(row); used_pairs.add(pair)
        endpoints[row.consequent] = endpoints.get(row.consequent, 0) + 1
        if len(rows) >= limit:
            break
    return pd.DataFrame(rows) if rows else selected.iloc[:0].copy()


def _maps(regions, rules, stable, out):
    paths = []
    fig = plt.figure(figsize=(11, 8.5))
    map_with_insets(fig, dict(zip(regions.code, regions.cluster)), regions, categorical=True)
    handles = [Patch(color=CLUSTER_COLORS[int(k)], label=f"Group {int(k)+1}") for k in sorted(regions.cluster.unique())]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(.5,.08))
    fig.text(.04,.035,"Average linkage · Jaccard distance · training species",fontsize=8,color="#63766c")
    paths += save_figure(fig, out, "01_region_clusters")
    fig = plt.figure(figsize=(11,8.5)); cmap=plt.get_cmap("YlGnBu"); norm=colors.Normalize(0, regions.richness.max())
    map_with_insets(fig, dict(zip(regions.code,regions.richness)),regions,cmap,norm)
    cax=fig.add_axes([.31,.115,.38,.018]); cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation="horizontal")
    cb.set_label("Recorded plant species (full cleaned dataset)")
    paths += save_figure(fig,out,"02_recorded_richness")
    fig=plt.figure(figsize=(11,8.5)); ax=map_with_insets(fig,dict(zip(regions.code,regions.cluster)),regions,categorical=True,labels=False)
    lookup=regions.set_index("code"); chosen=_select_rules(rules,limit=11,stable=stable)
    confnorm=colors.Normalize(.75,1); confcmap=plt.get_cmap("viridis")
    shown=[]
    for _, row in chosen.iterrows():
        a,b=lookup.loc[row.antecedent],lookup.loc[row.consequent]
        # Alaska/Hawaii arrows use a separate rule-network visualization below.
        if row.antecedent in {"AK","HI"} or row.consequent in {"AK","HI"}: continue
        arrow=FancyArrowPatch((a.longitude,a.latitude),(b.longitude,b.latitude),arrowstyle="-|>",mutation_scale=10,
            connectionstyle="arc3,rad=.14",lw=.7+min(float(row.lift),15)/8,color=confcmap(confnorm(row.confidence)),alpha=.9,zorder=8)
        ax.add_patch(arrow); shown.append(row)
        for code,pt in [(row.antecedent,a),(row.consequent,b)]:
            ax.scatter(pt.longitude,pt.latitude,s=18,c="white",edgecolor="#30463b",lw=.6,zorder=9)
            ax.annotate(code,(pt.longitude,pt.latitude),xytext=(3,4),textcoords="offset points",fontsize=7,zorder=10)
    cax=fig.add_axes([.30,.12,.37,.019]); cb=fig.colorbar(plt.cm.ScalarMappable(norm=confnorm,cmap=confcmap),cax=cax,orientation="horizontal")
    cb.set_label("Training confidence (arrow width scales with lift)")
    note="Single-region rules · held-out validation passed"
    if stable is not None: note += " · bootstrap recurrence ≥ 0.80"
    fig.text(.04,.05,note,fontsize=8,color="#63766c")
    paths += save_figure(fig,out,"03_stable_rule_geography")
    return paths


def _structure(regions, distance, linkage, selection, out):
    paths=[]; order=leaves_list(linkage); labels=regions.code.to_numpy()[order]
    fig=plt.figure(figsize=(12,10)); gs=fig.add_gridspec(3,2,height_ratios=[1.2,.16,6],width_ratios=[8,2],hspace=.035,wspace=.3)
    dax=fig.add_subplot(gs[0,0]); dendrogram(linkage,ax=dax,no_labels=True,color_threshold=0,above_threshold_color=GREEN); dax.set_ylabel("Jaccard\ndistance"); dax.spines[["top","right","bottom"]].set_visible(False); dax.set_xticks([])
    strip=fig.add_subplot(gs[1,0]); strip.imshow(np.array([colors.to_rgb(CLUSTER_COLORS[int(c)]) for c in regions.cluster.to_numpy()[order]])[None,:,:],aspect="auto"); strip.set_axis_off()
    ax=fig.add_subplot(gs[2,0]); im=ax.imshow(1-distance[np.ix_(order,order)],cmap="YlGnBu",vmin=0,vmax=1,aspect="equal")
    ax.set_xticks(range(len(labels)),labels,rotation=90,fontsize=5.8); ax.set_yticks(range(len(labels)),labels,fontsize=5.8)
    ax.set_xlabel("Regions (hierarchical order)"); ax.set_ylabel("Regions (hierarchical order)")
    cax=fig.add_subplot(gs[2,1]); cax.set_axis_off(); fig.colorbar(im,ax=cax,fraction=.4,shrink=.52,label="Jaccard similarity")
    sax=fig.add_subplot(gs[0,1]); sax.plot(selection.k,selection.silhouette,"o-",color=GREEN,ms=4)
    sax.axvspan(3.8,9.2,color=GREEN,alpha=.07,zorder=0)
    k=regions.cluster.nunique(); selected_score=selection.loc[selection.k.eq(k),"silhouette"]
    if len(selected_score): sax.scatter([k],[selected_score.iloc[0]],s=43,c=OCHRE,zorder=5,edgecolor="white",lw=.5)
    sax.set_xlabel("Groups (selection: 4–9)",fontsize=8); sax.set_ylabel("Silhouette",fontsize=8); sax.tick_params(labelsize=7)
    # Equal-aspect heatmaps shrink inside their GridSpec slot. Match the tree and
    # color strip to its final physical width so every leaf aligns with a column.
    fig.canvas.draw(); heatpos=ax.get_position()
    for topax in (dax,strip):
        old=topax.get_position(); topax.set_position([heatpos.x0,old.y0,heatpos.width,old.height])
    paths += save_figure(fig,out,"04_region_similarity_dendrogram")
    # Classical MDS is deterministic; coordinates describe the distance geometry, not geography.
    n=len(distance); center=np.eye(n)-np.ones((n,n))/n; gram=-.5*center@(distance**2)@center
    vals,vecs=np.linalg.eigh(gram); ids=np.argsort(vals)[::-1][:2]; coords=vecs[:,ids]*np.sqrt(np.maximum(vals[ids],0))
    fig,ax=plt.subplots(figsize=(10,7))
    for k in sorted(regions.cluster.unique()):
        take=regions.cluster.to_numpy()==k; ax.scatter(coords[take,0],coords[take,1],s=28+regions.richness.to_numpy()[take]/95,
            c=CLUSTER_COLORS[int(k)],edgecolors="white",lw=.5,alpha=.9,label=f"Group {int(k)+1}")
    # Deterministic force adjustment in display-normalized coordinates reduces label collisions.
    low=coords.min(axis=0); span=np.maximum(coords.max(axis=0)-low,1e-6)
    anchors=(coords-low)/span; positions=anchors+np.array([.012,.014])
    for _ in range(90):
        shifts=(anchors+np.array([.012,.014])-positions)*.018
        for i in range(n):
            for j in range(i):
                delta=positions[i]-positions[j]
                if abs(delta[0]) < .042 and abs(delta[1]) < .034:
                    push=np.array([.001 if delta[0]>=0 else -.001,.0018 if delta[1]>=0 else -.0018])
                    shifts[i]+=push; shifts[j]-=push
        positions+=shifts
    for i,code in enumerate(regions.code):
        target=low+positions[i]*span
        ax.plot([coords[i,0],target[0]],[coords[i,1],target[1]],c="#b0bdb5",lw=.4,zorder=1)
        ax.text(*target,code,fontsize=7,ha="center",va="center",zorder=5)
    ax.axhline(0,color="#e5ebe7",lw=.7,zorder=0); ax.axvline(0,color="#e5ebe7",lw=.7,zorder=0)
    ax.set_xlabel("Floristic composition · coordinate 1"); ax.set_ylabel("Floristic composition · coordinate 2")
    ax.legend(frameon=False,loc="best"); ax.text(.02,.02,"Classical MDS of training Jaccard distances\nPoint area scales with full-data recorded richness",transform=ax.transAxes,fontsize=8,color="#63766c")
    paths += save_figure(fig,out,"05_floristic_embedding")
    return paths


def _rules(rules, out):
    if rules.empty:
        paths=[]
        for stem in ("06_rule_support_confidence","07_rule_heldout_validation"):
            fig,ax=plt.subplots(figsize=(8,4)); ax.set_axis_off()
            ax.text(.5,.5,"No association rules satisfy the configured training thresholds.",
                    ha="center",va="center",transform=ax.transAxes,color="#63766c")
            paths+=save_figure(fig,out,stem)
        return paths
    paths=[]; fig,axes=plt.subplots(1,2,figsize=(12,5),sharex=True,sharey=True)
    vmin=max(1e-6,float(rules.lift.min())); vmax=max(vmin*1.001,float(np.quantile(rules.lift,.98)))
    norm=colors.LogNorm(vmin=vmin,vmax=vmax)
    for ax,size,label in zip(axes,[1,2],["(a) One-region antecedent","(b) Two-region antecedent"]):
        tab=rules[rules.antecedent_size.eq(size)]; tab=tab.sample(min(len(tab),4500),random_state=2026)
        sc=ax.scatter(tab.support,tab.confidence,c=np.clip(tab.lift,None,vmax),s=7+90*tab.support,alpha=.35,cmap="viridis",norm=norm,rasterized=True,edgecolors="none")
        ax.set_xlabel("Training support"); _panel(ax,label); ax.grid(alpha=.15)
    axes[0].set_ylabel("Training confidence"); fig.colorbar(sc,ax=axes.tolist(),label=f"Training lift (clipped at 98th percentile: {vmax:.1f})",fraction=.026,pad=.025)
    fig.subplots_adjust(top=.90,bottom=.15,wspace=.12)
    paths+=save_figure(fig,out,"06_rule_support_confidence")
    chosen=_select_rules(rules,16,require_validation=False).sort_values("confidence")
    fig,axes=plt.subplots(1,2,figsize=(11,6),sharey=True); yy=np.arange(len(chosen))
    for ax,metric,label in zip(axes,["confidence","lift"],["Confidence","Lift"]):
        train=chosen[f"train_{metric}"].to_numpy(); valid=chosen[f"validation_{metric}"].to_numpy()
        for y,a,b in zip(yy,train,valid): ax.plot([a,b],[y,y],color="#c4d0c8",lw=2)
        ax.scatter(train,yy,c=GREEN,s=30,label="Training",zorder=3); ax.scatter(valid,yy,c=OCHRE,s=34,marker="D",label="Held-out species",zorder=3)
        ax.set_xlabel(label); ax.grid(axis="x",alpha=.15)
    axes[0].set_yticks(yy,[f"{r.antecedent} → {r.consequent}" for _,r in chosen.iterrows()]); axes[0].legend(frameon=False,loc="lower right")
    fig.tight_layout(); paths+=save_figure(fig,out,"07_rule_heldout_validation")
    return paths


def _completion(summary, out):
    if summary.empty or not (summary.n_queries > 0).any(): return []
    fig,ax=plt.subplots(figsize=(7,4.5)); xs=[1,3,5]
    for i,row in summary.iterrows():
        ys=[row[f"recall_at_{k}"] for k in xs]
        ax.plot(xs,ys,"o-",c=[GREEN,BLUE,OCHRE][i%3],lw=2,label=row.method)
    ax.set_xticks(xs); ax.set_ylim(0,1); ax.set_xlabel("Number of candidate regions (k)"); ax.set_ylabel("Hit rate of the masked recorded region")
    ax.legend(frameon=False); ax.grid(alpha=.15); fig.tight_layout()
    return save_figure(fig,out,"16_masked_region_completion")


def generate_figures(results_dir:Path, figures_dir:Path)->list[Path]:
    """Regenerate figures from an already fitted, frozen result directory."""
    setup_style(); results_dir=Path(results_dir); figures_dir=Path(figures_dir)
    regions=pd.read_csv(results_dir/"regions.csv"); rules=pd.read_csv(results_dir/"core_rules.csv")
    summary=json.loads((results_dir/"run_summary.json").read_text(encoding="utf-8"))
    if summary.get("core_complete") is False:
        raise RuntimeError("The core experiment is still running; generate figures after training completes.")
    stable=pd.read_csv(results_dir/"bootstrap_rules.csv") if summary.get("extensions_complete") and (results_dir/"bootstrap_rules.csv").exists() else None
    paths=_maps(regions,rules,stable,figures_dir)
    paths+=_structure(regions,np.load(results_dir/"region_distance.npy"),np.load(results_dir/"region_linkage.npy"),pd.read_csv(results_dir/"cluster_selection.csv"),figures_dir)
    paths+=_rules(rules,figures_dir)
    if (results_dir/"missing_region_summary.csv").exists(): paths+=_completion(pd.read_csv(results_dir/"missing_region_summary.csv"),figures_dir)
    from .extension_plots import generate_extension_figures
    paths+=generate_extension_figures(results_dir,figures_dir)
    return paths
