"""Stage 7 figure generation for the Quantum Encoding Atlas.

This module generates all publication-ready figures for the tradeoff
analysis. Each plot function imports matplotlib locally to avoid slow
import at module load time.

Can be used standalone::

    python experiments/plotting.py --results-dir experiments/results/raw/stage7_tradeoff/

Or called programmatically from :func:`experiments.tradeoff.run_tradeoff_analysis`.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────

FIGURE_SIZE_STANDARD = (10, 6)
FIGURE_SIZE_WIDE = (12, 6)
FIGURE_SIZE_SQUARE = (8, 8)
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 10
FONT_SIZE_TITLE = 14
DPI = 300


# ── Internal Helpers ───────────────────────────────────────────────

def _save_figure(fig: Any, figure_dir: str, name: str) -> list[str]:
    """Save figure as both PNG (300 DPI) and PDF, then close."""
    import matplotlib.pyplot as plt
    os.makedirs(figure_dir, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = os.path.join(figure_dir, f"{name}.{ext}")
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = DPI
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def _abbreviate(name: str) -> str:
    """Abbreviate encoding name for plot labels."""
    abbrevs = {
        "angle": "Angle", "amplitude": "Amp", "basis": "Basis",
        "iqp": "IQP", "zz_feature_map": "ZZ", "pauli_feature_map": "Pauli",
        "data_reuploading": "DRU", "hardware_efficient": "HWE",
        "higher_order_angle": "HOA", "qaoa_encoding": "QAOA",
        "hamiltonian_encoding": "Ham", "symmetry_inspired": "Sym",
        "trainable_encoding": "Train", "so2_equivariant": "SO2",
        "cyclic_equivariant": "Cyc", "swap_equivariant": "Swap",
    }
    return abbrevs.get(name, name[:6])


# ── Plot Functions ─────────────────────────────────────────────────

def plot_accuracy_vs_depth(result_7_1: dict, profiles: dict, figure_dir: str) -> None:
    """7.1 scatter: x=depth, y=accuracy, labeled points."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overall = result_7_1.get("overall", {})
    if overall.get("status") == "insufficient_data":
        logger.warning("Skipping accuracy_vs_depth plot: insufficient data")
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    for name, p in profiles.items():
        depth = p.get("depth")
        vqc_acc = p.get("vqc_accuracy")
        if depth is None or not vqc_acc:
            continue
        mean_acc = float(np.mean(list(vqc_acc.values())))
        ci = p.get("vqc_ci")
        yerr = None
        if ci:
            lows = [c[0] for c in ci.values()]
            highs = [c[1] for c in ci.values()]
            yerr = [[mean_acc - np.mean(lows)], [np.mean(highs) - mean_acc]]
        ax.errorbar(depth, mean_acc, yerr=yerr, fmt="o", capsize=3, markersize=6)
        ax.annotate(_abbreviate(name), (depth, mean_acc), fontsize=8,
                     xytext=(4, 4), textcoords="offset points")

    rho = overall.get("rho", 0)
    p_val = overall.get("p_value", 1)
    ax.annotate(f"Spearman rho={rho:.3f}, p={p_val:.3f}",
                xy=(0.02, 0.98), xycoords="axes fraction",
                fontsize=10, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Circuit Depth", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Mean VQC Test Accuracy", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.set_title("Accuracy vs Circuit Depth", fontsize=FONT_SIZE_TITLE)
    plt.tight_layout()
    _save_figure(fig, figure_dir, "accuracy_vs_depth")


def plot_accuracy_vs_trainability(result_7_2: dict, profiles: dict, figure_dir: str) -> None:
    """7.2 scatter: x=trainability, y=accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    for name, p in profiles.items():
        train = p.get("trainability_estimate")
        vqc_acc = p.get("vqc_accuracy")
        if train is None or not vqc_acc:
            continue
        mean_acc = float(np.mean(list(vqc_acc.values())))
        ax.plot(train, mean_acc, "o", markersize=6)
        ax.annotate(_abbreviate(name), (train, mean_acc), fontsize=8,
                     xytext=(4, 4), textcoords="offset points")

    overall = result_7_2.get("overall", {})
    rho = overall.get("rho", 0)
    partial = overall.get("partial_rho", 0)
    ax.annotate(f"rho={rho:.3f}, partial_rho={partial:.3f}",
                xy=(0.02, 0.98), xycoords="axes fraction",
                fontsize=10, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Trainability Estimate", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Mean VQC Test Accuracy", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.set_title("Accuracy vs Trainability", fontsize=FONT_SIZE_TITLE)
    plt.tight_layout()
    _save_figure(fig, figure_dir, "accuracy_vs_trainability")


def plot_expressibility_vs_accuracy(result_7_3: dict, profiles: dict, figure_dir: str) -> None:
    """7.3 scatter: x=expressibility, y=accuracy with regression line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    exprs, accs = [], []
    for name, p in profiles.items():
        if not p.get("is_entangling"):
            continue
        expr = p.get("expressibility")
        vqc_acc = p.get("vqc_accuracy")
        if expr is None or not vqc_acc:
            continue
        mean_acc = float(np.mean(list(vqc_acc.values())))
        exprs.append(expr)
        accs.append(mean_acc)
        ax.plot(expr, mean_acc, "o", markersize=6)
        ax.annotate(_abbreviate(name), (expr, mean_acc), fontsize=8,
                     xytext=(4, 4), textcoords="offset points")

    if len(exprs) >= 2:
        z = np.polyfit(exprs, accs, 1)
        x_line = np.linspace(min(exprs), max(exprs), 50)
        ax.plot(x_line, np.polyval(z, x_line), "--", color="gray", alpha=0.7)

    overall = result_7_3.get("overall", {})
    rho = overall.get("rho", 0)
    p_val = overall.get("p_value", 1)
    ax.annotate(f"rho={rho:.3f}, p={p_val:.3f}",
                xy=(0.02, 0.98), xycoords="axes fraction", fontsize=10, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Expressibility", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Mean VQC Test Accuracy", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.set_title("Expressibility vs Accuracy (Entangling Only)", fontsize=FONT_SIZE_TITLE)
    plt.tight_layout()
    _save_figure(fig, figure_dir, "expressibility_vs_accuracy")


def plot_equivariant_comparison(result_7_4: dict, figure_dir: str) -> None:
    """7.4 grouped bar: equivariant vs general accuracy per dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_ds = {}
    for section in ("symmetric_datasets", "non_symmetric_datasets"):
        for ds, data in result_7_4.get(section, {}).items():
            if data.get("status") == "insufficient_data":
                continue
            all_ds[ds] = data

    if not all_ds:
        logger.warning("Skipping equivariant_comparison: no data")
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    ds_names = sorted(all_ds.keys())
    x = np.arange(len(ds_names))
    w = 0.35
    equi_vals = [all_ds[ds].get("equivariant_mean_acc", 0) for ds in ds_names]
    gen_vals = [all_ds[ds].get("general_mean_acc", 0) for ds in ds_names]
    ax.bar(x - w / 2, equi_vals, w, label="Equivariant", color="#4C72B0")
    ax.bar(x + w / 2, gen_vals, w, label="General", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, rotation=45, ha="right")
    ax.set_ylabel("Mean Accuracy", fontsize=FONT_SIZE_LABEL)
    ax.set_title("Equivariant vs General Encodings", fontsize=FONT_SIZE_TITLE)
    ax.legend()
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    _save_figure(fig, figure_dir, "equivariant_comparison")


def plot_barren_plateau_regression(result_7_5: dict, profiles: dict, figure_dir: str) -> None:
    """7.5 multi-series: x=depth, y=log(gradient_var), one line per family."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_family = result_7_5.get("per_family", {})
    if not per_family:
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    colors = plt.cm.tab10(np.linspace(0, 1, len(per_family)))

    for (fam, data), color in zip(sorted(per_family.items()), colors):
        if data.get("status"):
            continue
        slope = data.get("slope", 0)
        intercept = data.get("intercept", 0)
        n = data.get("n_points", 0)
        if n < 2:
            continue
        # Plot regression line
        x_range = np.linspace(1, 20, 50)
        ax.plot(x_range, slope * x_range + intercept, "-", color=color,
                label=f"{fam} (slope={slope:.3f})", alpha=0.7)

    ax.set_xlabel("Circuit Depth", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("log(Gradient Variance)", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.set_title("Barren Plateau Onset by Encoding Family", fontsize=FONT_SIZE_TITLE)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    _save_figure(fig, figure_dir, "barren_plateau_regression")


def plot_vqc_vs_kernel_ranking(result_7_6: dict, figure_dir: str) -> None:
    """7.6 paired bar: VQC rank vs kernel rank."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Use first dataset with data
    per_ds = result_7_6.get("per_dataset", {})
    ds_data = None
    for ds, data in per_ds.items():
        if data.get("vqc_ranking"):
            ds_data = data
            break
    if ds_data is None:
        return

    vqc_ranking = ds_data["vqc_ranking"]
    ker_ranking = ds_data["kernel_ranking"]
    n = len(vqc_ranking)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    x = np.arange(n)
    vqc_ranks = [vqc_ranking.index(enc) + 1 if enc in vqc_ranking else n for enc in vqc_ranking]
    ker_ranks = [ker_ranking.index(enc) + 1 if enc in ker_ranking else n for enc in vqc_ranking]
    w = 0.35
    ax.bar(x - w / 2, vqc_ranks, w, label="VQC Rank")
    ax.bar(x + w / 2, ker_ranks, w, label="Kernel Rank")
    ax.set_xticks(x)
    ax.set_xticklabels([_abbreviate(e) for e in vqc_ranking], rotation=45, ha="right")
    ax.set_ylabel("Rank (1=best)", fontsize=FONT_SIZE_LABEL)
    ax.set_title("VQC vs Kernel Rankings", fontsize=FONT_SIZE_TITLE)
    ax.legend()
    ax.invert_yaxis()
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    _save_figure(fig, figure_dir, "vqc_vs_kernel_ranking")


def plot_pareto_front_2d(result_7_7: dict, profiles: dict, figure_dir: str) -> None:
    """7.7a scatter: accuracy vs 1/depth, Pareto frontier line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    encodings = result_7_7.get("encodings", {})
    if not encodings:
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    pareto_x, pareto_y = [], []
    for name, data in encodings.items():
        objs = data.get("objectives", [0, 0, 0, 0])
        acc, inv_d = objs[0], objs[1]
        is_p = data.get("is_pareto", False)
        color = "#E24A33" if is_p else "#999999"
        marker = "*" if is_p else "o"
        size = 120 if is_p else 40
        ax.scatter(inv_d, acc, c=color, marker=marker, s=size, zorder=3 if is_p else 2)
        ax.annotate(_abbreviate(name), (inv_d, acc), fontsize=8,
                     xytext=(4, 4), textcoords="offset points")
        if is_p:
            pareto_x.append(inv_d)
            pareto_y.append(acc)

    # Connect Pareto frontier
    if pareto_x:
        order = np.argsort(pareto_x)
        ax.plot([pareto_x[i] for i in order], [pareto_y[i] for i in order],
                "--", color="#E24A33", alpha=0.5)

    ax.set_xlabel("1 / Depth (Resource Efficiency)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Mean VQC Accuracy", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.set_title("Pareto Front: Accuracy vs Resource Efficiency", fontsize=FONT_SIZE_TITLE)
    plt.tight_layout()
    _save_figure(fig, figure_dir, "pareto_front_2d")


def plot_pareto_front_parallel(result_7_7: dict, profiles: dict, figure_dir: str) -> None:
    """7.7b parallel coordinates: 4 axes, Pareto-optimal highlighted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    encodings = result_7_7.get("encodings", {})
    obj_names = result_7_7.get("objective_names", ["acc", "inv_d", "train", "noise"])
    if not encodings:
        return

    fig, axes = plt.subplots(1, len(obj_names) - 1, figsize=FIGURE_SIZE_WIDE, sharey=False)
    if len(obj_names) - 1 == 1:
        axes = [axes]

    for name, data in encodings.items():
        norm_objs = data.get("objectives_normalized", [])
        if len(norm_objs) != len(obj_names):
            continue
        is_p = data.get("is_pareto", False)
        color = "#E24A33" if is_p else "#CCCCCC"
        alpha = 0.9 if is_p else 0.3
        lw = 2 if is_p else 0.8
        for i, ax in enumerate(axes):
            ax.plot([0, 1], [norm_objs[i], norm_objs[i + 1]],
                    color=color, alpha=alpha, linewidth=lw)

    for i, ax in enumerate(axes):
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([obj_names[i], obj_names[i + 1]], fontsize=9)
        ax.tick_params(labelsize=FONT_SIZE_TICK)

    fig.suptitle("Parallel Coordinates: Pareto-optimal Highlighted", fontsize=FONT_SIZE_TITLE)
    plt.tight_layout()
    _save_figure(fig, figure_dir, "pareto_front_parallel")


def plot_ranking_sensitivity(result_7_8: dict, figure_dir: str) -> None:
    """7.8 box plot: rank distribution per encoding."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_enc = result_7_8.get("per_encoding", {})
    if not per_enc:
        return

    # Sort by mean rank
    sorted_encs = sorted(per_enc.items(), key=lambda x: x[1]["mean_rank"])
    names = [_abbreviate(n) for n, _ in sorted_encs]
    means = [d["mean_rank"] for _, d in sorted_encs]
    stds = [d["std_rank"] for _, d in sorted_encs]
    mins = [d["min_rank"] for _, d in sorted_encs]
    maxs = [d["max_rank"] for _, d in sorted_encs]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=3, color="#4C72B0", alpha=0.8)
    ax.scatter(x, mins, marker="v", color="green", zorder=3, s=20, label="Min rank")
    ax.scatter(x, maxs, marker="^", color="red", zorder=3, s=20, label="Max rank")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Rank (1=best)", fontsize=FONT_SIZE_LABEL)
    ax.set_title("Ranking Sensitivity (1000 weight samples)", fontsize=FONT_SIZE_TITLE)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    _save_figure(fig, figure_dir, "ranking_sensitivity")


def plot_quantum_vs_classical(result_7_10: dict, figure_dir: str) -> None:
    """7.10 grouped bar: x=dataset, bars=best quantum + classical baselines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_ds = result_7_10.get("per_dataset", {})
    if not per_ds:
        return

    ds_names = sorted(k for k, v in per_ds.items() if v.get("best_quantum_acc") is not None)
    if not ds_names:
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    x = np.arange(len(ds_names))

    # Quantum bars
    q_accs = [per_ds[ds].get("best_quantum_acc", 0) for ds in ds_names]
    ax.bar(x - 0.2, q_accs, 0.4, label="Best Quantum", color="#4C72B0")

    # Classical baseline bars (use best baseline)
    bl_accs = []
    for ds in ds_names:
        bls = per_ds[ds].get("baselines", {})
        best_bl = max((bl.get("acc", 0) or 0 for bl in bls.values()), default=0)
        bl_accs.append(best_bl)
    ax.bar(x + 0.2, bl_accs, 0.4, label="Best Classical", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, rotation=45, ha="right")
    ax.set_ylabel("Mean Test Accuracy", fontsize=FONT_SIZE_LABEL)
    ax.set_title("Quantum vs Classical Baselines", fontsize=FONT_SIZE_TITLE)
    ax.legend()
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    _save_figure(fig, figure_dir, "quantum_vs_classical")


def plot_family_radar(result_4_1: dict, figure_dir: str) -> None:
    """4.1 radar chart: one polygon per encoding family."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_metric = result_4_1.get("per_metric", {})
    metrics = result_4_1.get("metrics_analyzed", [])
    if not per_metric or not metrics:
        return

    # Collect families with data
    families: set[str] = set()
    for m in metrics:
        for fam in per_metric.get(m, {}).get("family_stats", {}):
            families.add(fam)

    if len(families) < 2:
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SQUARE, subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
    for (fam, color) in zip(sorted(families), colors):
        values = []
        for m in metrics:
            stats = per_metric.get(m, {}).get("family_stats", {}).get(fam, {})
            values.append(stats.get("mean", 0))
        # Normalize to [0, 1] for radar
        max_val = max(values) if values and max(values) > 0 else 1
        values = [v / max_val for v in values]
        values += values[:1]
        ax.plot(angles, values, "o-", label=fam, color=color, linewidth=1.5)
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_title("Encoding Family Comparison", fontsize=FONT_SIZE_TITLE, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=8)
    plt.tight_layout()
    _save_figure(fig, figure_dir, "family_radar")


def plot_noise_sensitivity(result_4_3: dict, figure_dir: str) -> None:
    """4.3 grouped bar: entangling vs non-entangling fidelity decay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ent_mean = result_4_3.get("entangling_mean")
    non_ent_mean = result_4_3.get("non_entangling_mean")
    if ent_mean is None or non_ent_mean is None:
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    ax.bar(["Entangling", "Non-Entangling"], [ent_mean, non_ent_mean],
           color=["#E24A33", "#4C72B0"])
    ax.set_ylabel("Mean Fidelity Decay", fontsize=FONT_SIZE_LABEL)
    ax.set_title("Noise Sensitivity: Entangling vs Non-Entangling", fontsize=FONT_SIZE_TITLE)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    _save_figure(fig, figure_dir, "noise_sensitivity")


def plot_kta_vs_accuracy(result_4_4: dict, figure_dir: str) -> None:
    """4.4 scatter: x=centered_KTA, y=kernel accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # This would need profile data — simplified version
    overall = result_4_4.get("overall", {})
    if overall.get("status") == "insufficient_data":
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    rho = overall.get("rho", 0)
    p_val = overall.get("p_value", 1)
    ax.annotate(f"rho={rho:.3f}, p={p_val:.3f}",
                xy=(0.02, 0.98), xycoords="axes fraction", fontsize=10, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
    ax.set_xlabel("Centered KTA", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Kernel-SVM Accuracy", fontsize=FONT_SIZE_LABEL)
    ax.set_title("KTA vs Kernel Accuracy", fontsize=FONT_SIZE_TITLE)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    _save_figure(fig, figure_dir, "kta_vs_accuracy")


def plot_overfitting_gap(result_4_5: dict, profiles: dict, figure_dir: str) -> None:
    """4.5 scatter: x=depth, y=train-test gap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_enc = result_4_5.get("per_encoding", {})
    if not per_enc:
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    for name, data in per_enc.items():
        depth = profiles.get(name, {}).get("depth")
        gap = data.get("mean_gap")
        if depth is None or gap is None:
            continue
        color = "#E24A33" if data.get("flagged") else "#4C72B0"
        ax.scatter(depth, gap, c=color, s=60, zorder=3)
        ax.annotate(_abbreviate(name), (depth, gap), fontsize=8,
                     xytext=(4, 4), textcoords="offset points")

    ax.axhline(0.15, linestyle="--", color="red", alpha=0.5, label="Overfitting threshold")
    ax.set_xlabel("Circuit Depth", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Train-Test Accuracy Gap", fontsize=FONT_SIZE_LABEL)
    ax.set_title("Overfitting Analysis", fontsize=FONT_SIZE_TITLE)
    ax.legend()
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    _save_figure(fig, figure_dir, "overfitting_gap")


def plot_expressibility_entanglement(result_4_6: dict, figure_dir: str) -> None:
    """4.6 2D scatter with quadrant lines and accuracy as color."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_enc = result_4_6.get("per_encoding", {})
    if not per_enc:
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SQUARE)
    exprs = [d["expr"] for d in per_enc.values()]
    ents = [d["ent"] for d in per_enc.values()]
    accs = [d["mean_accuracy"] for d in per_enc.values()]
    names = list(per_enc.keys())

    sc = ax.scatter(exprs, ents, c=accs, cmap="viridis", s=80, zorder=3)
    plt.colorbar(sc, ax=ax, label="Mean Accuracy")

    for i, name in enumerate(names):
        ax.annotate(_abbreviate(name), (exprs[i], ents[i]), fontsize=8,
                     xytext=(4, 4), textcoords="offset points")

    # Quadrant lines at median
    if exprs:
        ax.axvline(np.median(exprs), linestyle="--", color="gray", alpha=0.5)
    if ents:
        ax.axhline(np.median(ents), linestyle="--", color="gray", alpha=0.5)

    ax.set_xlabel("Expressibility", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Entanglement Capability", fontsize=FONT_SIZE_LABEL)
    ax.set_title("Expressibility-Entanglement Joint Analysis", fontsize=FONT_SIZE_TITLE)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    _save_figure(fig, figure_dir, "expressibility_entanglement")


def plot_efficiency_frontier(result_4_7: dict, profiles: dict, figure_dir: str) -> None:
    """4.7 scatter: x=gate_count, y=accuracy with frontier line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_enc = result_4_7.get("per_encoding", {})
    frontier = result_4_7.get("frontier_encodings", [])
    if not per_enc:
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    for name in per_enc:
        p = profiles.get(name, {})
        gc = p.get("gate_count")
        vqc_acc = p.get("vqc_accuracy")
        if gc is None or not vqc_acc:
            continue
        mean_acc = float(np.mean(list(vqc_acc.values())))
        on_f = per_enc[name].get("on_frontier", False)
        color = "#E24A33" if on_f else "#999999"
        marker = "*" if on_f else "o"
        size = 120 if on_f else 40
        ax.scatter(gc, mean_acc, c=color, marker=marker, s=size, zorder=3 if on_f else 2)
        ax.annotate(_abbreviate(name), (gc, mean_acc), fontsize=8,
                     xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Gate Count", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Mean VQC Accuracy", fontsize=FONT_SIZE_LABEL)
    ax.set_title("Resource Efficiency Frontier", fontsize=FONT_SIZE_TITLE)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    _save_figure(fig, figure_dir, "efficiency_frontier")


def plot_scaling_behavior(result_4_8: dict, figure_dir: str) -> None:
    """4.8 multi-series line: x=n_features, y=expressibility/entanglement."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_enc = result_4_8.get("per_encoding", {})
    if not per_enc:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE_WIDE)

    for name, data in per_enc.items():
        expr_vals = data.get("expr_values", {})
        if expr_vals:
            nfs = sorted(expr_vals.keys(), key=int)
            ax1.plot([int(n) for n in nfs], [expr_vals[n] for n in nfs],
                     "o-", label=_abbreviate(name), markersize=4)

        ent_vals = data.get("ent_values", {})
        if ent_vals:
            nfs = sorted(ent_vals.keys(), key=int)
            ax2.plot([int(n) for n in nfs], [ent_vals[n] for n in nfs],
                     "o-", label=_abbreviate(name), markersize=4)

    ax1.set_xlabel("n_features", fontsize=FONT_SIZE_LABEL)
    ax1.set_ylabel("Expressibility", fontsize=FONT_SIZE_LABEL)
    ax1.set_title("Expressibility Scaling", fontsize=12)
    ax1.legend(fontsize=7, ncol=2)
    ax1.tick_params(labelsize=FONT_SIZE_TICK)

    ax2.set_xlabel("n_features", fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel("Entanglement", fontsize=FONT_SIZE_LABEL)
    ax2.set_title("Entanglement Scaling", fontsize=12)
    ax2.legend(fontsize=7, ncol=2)
    ax2.tick_params(labelsize=FONT_SIZE_TICK)

    fig.suptitle("Scaling Behavior", fontsize=FONT_SIZE_TITLE)
    plt.tight_layout()
    _save_figure(fig, figure_dir, "scaling_behavior")


def plot_pairwise_heatmap(result_4_9: dict, figure_dir: str) -> None:
    """4.9 heatmap: encoding x encoding, colored by Cliff's delta."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vqc_data = result_4_9.get("vqc", {}).get("per_dataset", {})
    if not vqc_data:
        return

    # Use first available dataset
    ds_name = next(iter(vqc_data), None)
    if ds_name is None:
        return
    pw = vqc_data[ds_name]
    pairs = pw.get("pairs", [])
    if not pairs:
        return

    # Collect unique names
    names_set: set[str] = set()
    for p in pairs:
        names_set.add(p["method_a"])
        names_set.add(p["method_b"])
    names = sorted(names_set)
    n = len(names)
    idx = {name: i for i, name in enumerate(names)}

    matrix = np.zeros((n, n))
    for p in pairs:
        i, j = idx[p["method_a"]], idx[p["method_b"]]
        delta = p.get("cliffs_delta", 0)
        matrix[i, j] = delta
        matrix[j, i] = -delta

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SQUARE)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Cliff's delta")

    abbr_names = [_abbreviate(n) for n in names]
    ax.set_xticks(range(n))
    ax.set_xticklabels(abbr_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(abbr_names, fontsize=8)

    # Mark significant pairs
    for p in pairs:
        if p.get("significant"):
            i, j = idx[p["method_a"]], idx[p["method_b"]]
            ax.text(j, i, "*", ha="center", va="center", fontsize=12, fontweight="bold")
            ax.text(i, j, "*", ha="center", va="center", fontsize=12, fontweight="bold")

    ax.set_title(f"Pairwise Comparisons ({ds_name})", fontsize=FONT_SIZE_TITLE)
    plt.tight_layout()
    _save_figure(fig, figure_dir, "pairwise_heatmap")


def plot_convergence_analysis(result_4_10: dict, figure_dir: str) -> None:
    """4.10 bar chart: divergence rate per encoding."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_enc = result_4_10.get("per_encoding", {})
    if not per_enc:
        return

    sorted_encs = sorted(per_enc.items(), key=lambda x: x[1].get("divergence_rate", 0), reverse=True)
    names = [_abbreviate(n) for n, _ in sorted_encs]
    rates = [d.get("divergence_rate", 0) for _, d in sorted_encs]
    colors = ["#E24A33" if d.get("flagged") else "#4C72B0" for _, d in sorted_encs]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    ax.bar(range(len(names)), rates, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.axhline(0.20, linestyle="--", color="red", alpha=0.5, label="20% threshold")
    ax.set_ylabel("Divergence Rate", fontsize=FONT_SIZE_LABEL)
    ax.set_title("VQC Convergence Quality", fontsize=FONT_SIZE_TITLE)
    ax.legend()
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    _save_figure(fig, figure_dir, "convergence_analysis")


# ── Master Generator ───────────────────────────────────────────────

def generate_all_figures(
    analysis_results: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
    figure_dir: str,
) -> list[str]:
    """Generate all Stage 7 figures.

    Parameters
    ----------
    analysis_results : dict
        Complete analysis results from ``run_tradeoff_analysis``.
    profiles : dict
        Per-encoding profiles.
    figure_dir : str
        Output directory for figures.

    Returns
    -------
    list[str]
        List of generated file paths.
    """
    generated: list[str] = []

    plot_calls = [
        ("accuracy_vs_depth", lambda: plot_accuracy_vs_depth(
            analysis_results.get("7.1", {}), profiles, figure_dir)),
        ("accuracy_vs_trainability", lambda: plot_accuracy_vs_trainability(
            analysis_results.get("7.2", {}), profiles, figure_dir)),
        ("expressibility_vs_accuracy", lambda: plot_expressibility_vs_accuracy(
            analysis_results.get("7.3", {}), profiles, figure_dir)),
        ("equivariant_comparison", lambda: plot_equivariant_comparison(
            analysis_results.get("7.4", {}), figure_dir)),
        ("barren_plateau_regression", lambda: plot_barren_plateau_regression(
            analysis_results.get("7.5", {}), profiles, figure_dir)),
        ("vqc_vs_kernel_ranking", lambda: plot_vqc_vs_kernel_ranking(
            analysis_results.get("7.6", {}), figure_dir)),
        ("pareto_front_2d", lambda: plot_pareto_front_2d(
            analysis_results.get("7.7", {}), profiles, figure_dir)),
        ("pareto_front_parallel", lambda: plot_pareto_front_parallel(
            analysis_results.get("7.7", {}), profiles, figure_dir)),
        ("ranking_sensitivity", lambda: plot_ranking_sensitivity(
            analysis_results.get("7.8", {}), figure_dir)),
        ("quantum_vs_classical", lambda: plot_quantum_vs_classical(
            analysis_results.get("7.10", {}), figure_dir)),
        ("family_radar", lambda: plot_family_radar(
            analysis_results.get("4.1", {}), figure_dir)),
        ("noise_sensitivity", lambda: plot_noise_sensitivity(
            analysis_results.get("4.3", {}), figure_dir)),
        ("kta_vs_accuracy", lambda: plot_kta_vs_accuracy(
            analysis_results.get("4.4", {}), figure_dir)),
        ("overfitting_gap", lambda: plot_overfitting_gap(
            analysis_results.get("4.5", {}), profiles, figure_dir)),
        ("expressibility_entanglement", lambda: plot_expressibility_entanglement(
            analysis_results.get("4.6", {}), figure_dir)),
        ("efficiency_frontier", lambda: plot_efficiency_frontier(
            analysis_results.get("4.7", {}), profiles, figure_dir)),
        ("scaling_behavior", lambda: plot_scaling_behavior(
            analysis_results.get("4.8", {}), figure_dir)),
        ("pairwise_heatmap", lambda: plot_pairwise_heatmap(
            analysis_results.get("4.9", {}), figure_dir)),
        ("convergence_analysis", lambda: plot_convergence_analysis(
            analysis_results.get("4.10", {}), figure_dir)),
    ]

    for name, func in plot_calls:
        try:
            func()
            generated.append(name)
            logger.info("Generated figure: %s", name)
        except Exception as e:
            logger.warning("Failed to generate figure %s: %s", name, e)

    return generated


# ── Standalone Mode ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Regenerate Stage 7 figures from saved results.")
    parser.add_argument("--results-dir", default="experiments/results/raw/stage7_tradeoff",
                        help="Directory containing analysis JSON files.")
    parser.add_argument("--figure-dir", default="experiments/results/figures",
                        help="Output directory for figures.")
    args = parser.parse_args()

    # Load analysis results from summary.json
    summary_path = os.path.join(args.results_dir, "summary.json")
    if not os.path.isfile(summary_path):
        print(f"Error: {summary_path} not found.")
        sys.exit(1)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Extract sub-analysis results from the single tradeoff result
    results_list = summary.get("results", [])
    if not results_list:
        print("Error: No results in summary.json")
        sys.exit(1)

    tradeoff_result = results_list[0].get("result", {})
    analysis_results = tradeoff_result.get("sub_analyses", {})

    # We don't have profiles saved — build them from stage dirs
    # For standalone mode, use empty profiles (figures will be limited)
    profiles: dict[str, dict[str, Any]] = {}

    generated = generate_all_figures(analysis_results, profiles, args.figure_dir)
    print(f"Generated {len(generated)} figures in {args.figure_dir}")
