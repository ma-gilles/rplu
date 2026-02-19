#!/usr/bin/env python3
"""
Paper-style experiments: baryrat AAA vs CUR-Half (RPLU/C2PLU)
for tan(sqrt(d) * z^sqrt(d)) on the unit disk.

Outputs:
- Summary timing/accuracy plots across d values
- Poles + error plots for a selected d value
"""

import argparse
import inspect
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from cauchy.loewner_utils import sample_unit_disk

try:
    from cauchy.barycentric import barycentric_poles, evaluate_barycentric
    BARYCENTRIC_AVAILABLE = True
except Exception:
    barycentric_poles = None  # type: ignore
    evaluate_barycentric = None  # type: ignore
    BARYCENTRIC_AVAILABLE = False

from cauchy.cur_approximant import build_cur_half_approximant
CUR_AVAILABLE = True

try:
    from baryrat import aaa as baryrat_aaa
    BARYRAT_AVAILABLE = True
except Exception:
    baryrat_aaa = None  # type: ignore
    BARYRAT_AVAILABLE = False


CASE_KEY = "tan_power_disk"
CASE_LABEL = "tan(sqrt(d)*z^sqrt(d)) on unit disk"
METHOD_STYLE = {
    "rplu": {"label": "RPLU", "color": "#1f77b4", "marker": "s", "linestyle": "--"},
    "c2plu": {"label": "C2PLU", "color": "#2ca02c", "marker": "^", "linestyle": "-"},
    "aaa": {"label": "AAA", "color": "#D62728", "marker": "o", "linestyle": "-"},
}


def _next_available_path(path: Path, max_tries: int = 999) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for k in range(1, max_tries + 1):
        candidate = path.with_name(f"{stem}_{k}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Unable to find an unused filename for {path} after {max_tries} tries.")


def set_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-paper")
    except Exception:
        try:
            plt.style.use("seaborn-paper")
        except Exception:
            plt.style.use("default")

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Liberation Serif"],
            "font.size": 14,
            "axes.labelsize": 15,
            "axes.titlesize": 15,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 12,
            "figure.titlesize": 16,
            "axes.linewidth": 2.0,
            "grid.linewidth": 1.2,
            "lines.linewidth": 3.0,
            "lines.markersize": 8,
            "patch.linewidth": 1.2,
            "xtick.major.width": 1.6,
            "ytick.major.width": 1.6,
            "xtick.minor.width": 1.0,
            "ytick.minor.width": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.25,
            "text.usetex": False,
        }
    )


def ensure_dir(path_str: str) -> Path:
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (complex, np.complexfloating)):
        return {"real": float(np.real(obj)), "imag": float(np.imag(obj))}
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _complex_pairs(arr) -> list[list[float]]:
    if arr is None:
        return []
    z = np.asarray(arr).ravel()
    if z.size == 0:
        return []
    return [[float(v.real), float(v.imag)] for v in z]


def _extract_pole_data_for_json(d: float, method_data: dict, train_points=None) -> dict:
    out = {"case_key": CASE_KEY, "d": float(d), "methods": {}}
    true_poles = None
    for m in ("rplu", "c2plu", "aaa"):
        if m in method_data and method_data[m].get("true_poles") is not None:
            true_poles = method_data[m].get("true_poles")
            break
    if true_poles is not None:
        out["true_poles"] = _complex_pairs(true_poles)
    if train_points is not None:
        out["train_points"] = _complex_pairs(train_points)

    for method_key, info in method_data.items():
        if method_key not in METHOD_STYLE:
            continue
        out["methods"][method_key] = {
            "support_points": _complex_pairs(info.get("support_points")),
            "poles": _complex_pairs(info.get("poles")),
            "rank": int(info.get("rank")) if info.get("rank") is not None else None,
            "side": info.get("side"),
            "max_error": float(info.get("max_error")) if info.get("max_error") is not None else None,
        }
    return out


def f_tan_power_factory(d: float):
    sqrt_d = int(np.round(np.sqrt(d)))

    def f(z: np.ndarray) -> np.ndarray:
        r = np.abs(z)
        theta = np.angle(z)
        z_power = (r ** sqrt_d) * np.exp(1j * sqrt_d * theta)
        return np.tan(sqrt_d * z_power)

    return f


def _tan_true_poles_in_disk(d: float, radius: float = 1.0) -> np.ndarray:
    sqrt_d = int(np.round(np.sqrt(d)))
    if sqrt_d <= 0:
        return np.array([], dtype=np.complex128)
    radius_power = radius ** sqrt_d
    k_min = int(np.ceil((-sqrt_d - 0.5 * np.pi) / np.pi))
    k_max = int(np.floor((sqrt_d - 0.5 * np.pi) / np.pi))
    poles = []
    for k in range(k_min, k_max + 1):
        z_power = (0.5 * np.pi + k * np.pi) / sqrt_d
        if abs(z_power) > radius_power:
            continue
        base = z_power + 0j
        r = abs(base) ** (1.0 / sqrt_d)
        base_arg = np.angle(base)
        for m in range(sqrt_d):
            angle = (base_arg + 2.0 * np.pi * m) / sqrt_d
            poles.append(r * np.exp(1j * angle))
    return np.array(poles, dtype=np.complex128)


def _pole_distance_stats(poles: np.ndarray, true_poles: np.ndarray, radius: float = 1.0) -> dict:
    poles = np.asarray(poles) if poles is not None else np.array([], dtype=np.complex128)
    true_poles = np.asarray(true_poles) if true_poles is not None else np.array([], dtype=np.complex128)
    poles_in = poles[np.abs(poles) <= radius]
    true_in = true_poles[np.abs(true_poles) <= radius]
    stats = {
        "approx_poles_in_disk": int(len(poles_in)),
        "true_poles_in_disk": int(len(true_in)),
        "max_pole_dist_to_true": np.nan,
        "max_true_dist_to_approx": np.nan,
    }
    if len(poles_in) > 0 and len(true_in) > 0:
        distances = np.abs(poles_in[:, None] - true_in[None, :])
        stats["max_pole_dist_to_true"] = float(np.max(np.min(distances, axis=1)))
        stats["max_true_dist_to_approx"] = float(np.max(np.min(distances, axis=0)))
    return stats


def _first_attr(obj, names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _extract_barycentric_fields(obj):
    eval_obj = obj
    if isinstance(obj, tuple):
        if len(obj) >= 7:
            eval_obj = obj[0] if callable(obj[0]) else obj
            support, values, weights = obj[4], obj[5], obj[6]
        elif len(obj) >= 4 and callable(obj[0]):
            eval_obj = obj[0]
            support, values, weights = obj[1], obj[2], obj[3]
        elif len(obj) >= 3:
            support, values, weights = obj[0], obj[1], obj[2]
        else:
            raise RuntimeError("Unsupported AAA tuple return.")
    else:
        support = _first_attr(obj, ["support_points", "support", "z", "nodes", "zj"])
        values = _first_attr(obj, ["support_values", "values", "f", "fj"])
        weights = _first_attr(obj, ["weights", "w", "wj"])
    support = np.asarray(support) if support is not None else None
    values = np.asarray(values) if values is not None else None
    weights = np.asarray(weights) if weights is not None else None
    return eval_obj, support, values, weights


def _call_baryrat_aaa(z, f, tol, max_terms, cleanup, cleanup_tol):
    if baryrat_aaa is None:
        raise RuntimeError("baryrat AAA unavailable")
    kwargs = {}
    try:
        sig = inspect.signature(baryrat_aaa)
        params = sig.parameters
        if "tol" in params:
            kwargs["tol"] = tol
        elif "rtol" in params:
            kwargs["rtol"] = tol
        elif "eps" in params:
            kwargs["eps"] = tol
        if "mmax" in params:
            kwargs["mmax"] = max_terms
        elif "max_terms" in params:
            kwargs["max_terms"] = max_terms
        elif "max_degree" in params:
            kwargs["max_degree"] = max_terms
        if cleanup:
            if "cleanup" in params:
                kwargs["cleanup"] = True
            elif "clean_up" in params:
                kwargs["clean_up"] = True
            if "cleanup_tol" in params:
                kwargs["cleanup_tol"] = cleanup_tol
            elif "clean_up_tol" in params:
                kwargs["clean_up_tol"] = cleanup_tol
    except Exception:
        kwargs = {}
    return baryrat_aaa(z, f, **kwargs)


def baryrat_cleanup_support():
    if baryrat_aaa is None:
        return False, False
    try:
        params = inspect.signature(baryrat_aaa).parameters
    except Exception:
        return False, False
    supports_cleanup = ("cleanup" in params) or ("clean_up" in params)
    supports_cleanup_tol = ("cleanup_tol" in params) or ("clean_up_tol" in params)
    return supports_cleanup, supports_cleanup_tol


def mask_points(z, poles, pole_buffer):
    if poles is None or len(poles) == 0:
        return np.ones(len(z), dtype=bool)
    return np.min(np.abs(z[:, None] - poles[None, :]), axis=1) > pole_buffer


def max_error(f, approx_eval, z, poles, pole_buffer):
    mask = mask_points(z, poles, pole_buffer)
    if not np.any(mask):
        return np.nan
    f_true = f(z[mask])
    f_pred = approx_eval(z[mask])
    errs = np.abs(f_true - f_pred)
    errs = errs[np.isfinite(errs)]
    return float(np.max(errs)) if errs.size else np.nan


def _initial_eval_grid_size(grid_size, min_points):
    if min_points is None or min_points <= 0:
        return int(grid_size)
    target = int(min_points)
    grid_size = int(grid_size)
    ratio = np.pi / 4.0
    approx_size = int(np.ceil(np.sqrt(target / ratio)))
    return max(grid_size, approx_size)


def build_eval_grid(grid_size, min_points=None):
    grid_size = _initial_eval_grid_size(grid_size, min_points)
    target = int(min_points) if min_points is not None else None
    while True:
        grid = np.linspace(-1.0, 1.0, grid_size)
        X, Y = np.meshgrid(grid, grid)
        Z = X + 1j * Y
        mask = np.abs(Z) <= 1.0
        if target is None or int(np.sum(mask)) >= target:
            return Z, mask, Z[mask]
        grid_size += 1


def _reference_scaling(d_vals, series, power, scale_factor=1.0):
    for row in reversed(series):
        y = row.get("time", np.nan)
        if np.isfinite(y) and y > 0:
            d0 = float(row["d"])
            scale = y / (d0 ** power)
            return scale_factor * scale * (d_vals ** power)
    return None


def plot_summary(case_label, case_key, n, results, output_dir):
    fig, axes = plt.subplots(4, 1, figsize=(7.8, 10.6), sharex=True)
    fig.patch.set_facecolor("white")
    time_vals = []

    def _plot_series(ax, metric, yscale, ylabel):
        plotted = False
        for method_key, series in results["methods"].items():
            if not series:
                continue
            data = np.array([row[metric] for row in series], dtype=float)
            d_vals_local = np.array([row["d"] for row in series], dtype=float)
            style = METHOD_STYLE[method_key]
            ax.plot(
                d_vals_local,
                data,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=3.0,
                markersize=8,
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=1.2,
                label=style["label"],
            )
            plotted = True
            if metric == "time":
                for val in data:
                    if np.isfinite(val) and val > 0:
                        time_vals.append(float(val))
        if yscale:
            ax.set_yscale(yscale)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.9)
        ax.set_ylabel(ylabel, fontsize=12)
        return plotted

    def _plot_train_and_fine_errors(ax):
        for method_key, series in results["methods"].items():
            if not series:
                continue
            d_vals_local = np.array([row["d"] for row in series], dtype=float)
            train = np.array([row.get("train_max_error", np.nan) for row in series], dtype=float)
            fine = np.array([row.get("fine_max_error", np.nan) for row in series], dtype=float)
            style = METHOD_STYLE[method_key]

            ax.plot(
                d_vals_local,
                fine,
                color=style["color"],
                linestyle="-",
                linewidth=3.0,
                marker=style["marker"],
                markersize=8,
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=1.2,
                label="_nolegend_",
            )
            ax.plot(
                d_vals_local,
                train,
                color=style["color"],
                linestyle=":",
                linewidth=2.4,
                alpha=0.9,
                label="_nolegend_",
            )

        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.9)
        ax.set_ylabel("Max error (log scale)", fontsize=12)

    _plot_series(axes[0], "time", "log", "Build time (s, log-log)")
    axes[0].set_xscale("log")
    ref_anchor = results["methods"].get("rplu") or results["methods"].get("c2plu") or []
    ref_d_vals = np.array([row["d"] for row in ref_anchor], dtype=float)
    ref_d = _reference_scaling(ref_d_vals, ref_anchor, power=1, scale_factor=0.9)
    if ref_d is not None:
        axes[0].plot(ref_d_vals, ref_d, color="#555555", linestyle=":", linewidth=2.0, label="O(d)")
    if results["methods"].get("aaa"):
        ref_d2_vals = np.array([row["d"] for row in results["methods"]["aaa"]], dtype=float)
        ref_d2 = _reference_scaling(ref_d2_vals, results["methods"]["aaa"], power=2, scale_factor=0.95)
        if ref_d2 is not None:
            axes[0].plot(ref_d2_vals, ref_d2, color="#111111", linestyle="--", linewidth=2.0, label="O(d^2)")
    if time_vals:
        t_min = min(time_vals)
        t_max = max(time_vals)
        axes[0].set_ylim(t_min * 0.8, t_max * 1.25)

    _plot_series(axes[1], "time", None, "Build time (s, linear)")
    _plot_train_and_fine_errors(axes[2])
    _plot_series(axes[3], "poles", None, "# poles")
    _plot_series(axes[3], "support_points", None, "# support points")

    axes[3].set_xlabel("d", fontsize=12)
    axes[0].legend(loc="best", fontsize=11, frameon=True, edgecolor="black")
    axes[0].set_title(f"{case_label} (n={n})", fontsize=14)
    plt.tight_layout()

    fname = os.path.join(output_dir, f"summary_{case_key}_n{n}.png")
    print(f"Saving plot: {fname}", flush=True)
    fig.savefig(fname, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig_leg, ax_leg = plt.subplots(1, 1, figsize=(4.2, 1.4))
    ax_leg.axis("off")
    h_test = ax_leg.plot([], [], color="#111111", linestyle="-", linewidth=3.0, label="test")[0]
    h_train = ax_leg.plot([], [], color="#111111", linestyle=":", linewidth=2.4, label="train")[0]
    fig_leg.legend(
        [h_test, h_train],
        ["test", "train"],
        loc="center",
        ncol=2,
        frameon=True,
        edgecolor="black",
        fontsize=12,
    )
    fig_leg.tight_layout()
    legend_path = os.path.join(output_dir, f"summary_{case_key}_n{n}_train_test_legend.png")
    print(f"Saving plot: {legend_path}", flush=True)
    fig_leg.savefig(legend_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_leg)
    return fname


def plot_summary_panels(case_label, case_key, n, results, output_dir):
    panels = [
        ("time_log", "time", "log", "Build time (s, log-log)", True),
        ("time_linear", "time", None, "Build time (s, linear)", False),
        ("errors", "__train_fine__", "log", "Max error (train + fine)", False),
        ("poles", "poles", None, "# poles", False),
        ("support_points", "support_points", None, "# support points", False),
    ]
    paths = []
    legend_handles = []
    legend_labels = []
    time_vals = []
    for key, metric, yscale, ylabel, add_refs in panels:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
        fig.patch.set_facecolor("white")
        for method_key, series in results["methods"].items():
            if not series:
                continue
            d_vals_local = np.array([row["d"] for row in series], dtype=float)
            style = METHOD_STYLE[method_key]
            if metric == "__train_fine__":
                train = np.array([row.get("train_max_error", np.nan) for row in series], dtype=float)
                fine = np.array([row.get("fine_max_error", np.nan) for row in series], dtype=float)
                ax.plot(
                    d_vals_local,
                    fine,
                    color=style["color"],
                    linestyle="-",
                    linewidth=3.0,
                    marker=style["marker"],
                    markersize=8,
                    markerfacecolor="white",
                    markeredgecolor=style["color"],
                    markeredgewidth=1.2,
                    label=style["label"],
                )
                ax.plot(
                    d_vals_local,
                    train,
                    color=style["color"],
                    linestyle=":",
                    linewidth=2.4,
                    alpha=0.9,
                    label="_nolegend_",
                )
            else:
                data = np.array([row[metric] for row in series], dtype=float)
                ax.plot(
                    d_vals_local,
                    data,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=3.0,
                    markersize=8,
                    markerfacecolor="white",
                    markeredgecolor=style["color"],
                    markeredgewidth=1.2,
                    label=style["label"],
                )
                if metric == "time":
                    for val in data:
                        if np.isfinite(val) and val > 0:
                            time_vals.append(float(val))
        if add_refs:
            ref_anchor = results["methods"].get("rplu") or results["methods"].get("c2plu") or []
            ref_d_vals = np.array([row["d"] for row in ref_anchor], dtype=float)
            ref_d = _reference_scaling(ref_d_vals, ref_anchor, power=1, scale_factor=0.9)
            if ref_d is not None:
                ax.plot(ref_d_vals, ref_d, color="#555555", linestyle=":", linewidth=2.0)
            if results["methods"].get("aaa"):
                ref_d2_vals = np.array([row["d"] for row in results["methods"]["aaa"]], dtype=float)
                ref_d2 = _reference_scaling(ref_d2_vals, results["methods"]["aaa"], power=2, scale_factor=0.95)
                if ref_d2 is not None:
                    ax.plot(ref_d2_vals, ref_d2, color="#111111", linestyle="--", linewidth=2.0)
        if not legend_handles:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        if metric == "__train_fine__":
            h_fine = ax.plot([], [], color="#111111", linestyle="-", linewidth=3.0, label="fine grid")[0]
            h_train = ax.plot([], [], color="#111111", linestyle=":", linewidth=2.4, label="train")[0]
            ax.legend(handles=[h_fine, h_train], loc="best", fontsize=11, frameon=True, edgecolor="black")
        if yscale:
            ax.set_yscale(yscale)
        if key == "time_log":
            ax.set_xscale("log")
            if time_vals:
                t_min = min(time_vals)
                t_max = max(time_vals)
                ax.set_ylim(t_min * 0.8, t_max * 1.25)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.tick_params(axis="both", labelsize=18, width=1.6, length=5)
        plt.tight_layout()
        fname = os.path.join(output_dir, f"summary_{case_key}_n{n}_{key}.png")
        print(f"Saving plot: {fname}", flush=True)
        fig.savefig(fname, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(fname)
    if legend_handles:
        fig_leg, ax_leg = plt.subplots(1, 1, figsize=(7.2, 1.6))
        ax_leg.axis("off")
        fig_leg.legend(
            legend_handles,
            legend_labels,
            loc="center",
            ncol=min(3, max(1, len(legend_labels))),
            frameon=True,
            edgecolor="black",
            fontsize=12,
        )
        fig_leg.tight_layout()
        legend_path = os.path.join(output_dir, f"summary_{case_key}_n{n}_legend.png")
        print(f"Saving plot: {legend_path}", flush=True)
        fig_leg.savefig(legend_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig_leg)
        paths.append(legend_path)

    fig_leg, ax_leg = plt.subplots(1, 1, figsize=(4.2, 1.4))
    ax_leg.axis("off")
    h_test = ax_leg.plot([], [], color="#111111", linestyle="-", linewidth=3.0, label="test")[0]
    h_train = ax_leg.plot([], [], color="#111111", linestyle=":", linewidth=2.4, label="train")[0]
    fig_leg.legend(
        [h_test, h_train],
        ["test", "train"],
        loc="center",
        ncol=2,
        frameon=True,
        edgecolor="black",
        fontsize=12,
    )
    fig_leg.tight_layout()
    legend_path = os.path.join(output_dir, f"summary_{case_key}_n{n}_train_test_legend.png")
    print(f"Saving plot: {legend_path}", flush=True)
    fig_leg.savefig(legend_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_leg)
    paths.append(legend_path)
    return paths


def plot_poles_and_error(case_label, case_key, n, d, method_data, output_dir):
    methods = list(method_data.keys())
    ncols = len(methods)
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 9))
    fig.patch.set_facecolor("white")

    vmin = None
    vmax = None
    all_vals = []
    for info in method_data.values():
        if info["error_grid"] is None:
            continue
        grid_vals = info["error_grid"]
        finite = grid_vals[np.isfinite(grid_vals)]
        if finite.size:
            all_vals.append(finite)
    if all_vals:
        merged = np.concatenate(all_vals)
        vmin = np.nanmin(merged)
        vmax = np.nanmax(merged)

    def _plot_poles(ax, support, poles, title, circle_color="#1f77b4"):
        support = np.asarray(support)
        poles = np.asarray(poles)
        ax.scatter(
            np.real(support),
            np.imag(support),
            s=26,
            c="#5c5c5c",
            alpha=0.6,
            marker="o",
            linewidths=0.0,
            rasterized=True,
            label="support",
            zorder=1,
        )
        if len(poles) > 0:
            ax.scatter(
                np.real(poles),
                np.imag(poles),
                s=50,
                facecolors="none",
                edgecolors="#d62728",
                linewidths=1.2,
                marker="o",
                alpha=0.9,
                label="poles",
                zorder=3,
            )
            ax.scatter(
                np.real(poles),
                np.imag(poles),
                s=10,
                c="#d62728",
                marker="o",
                linewidths=0.0,
                alpha=0.6,
                label="_nolegend_",
                zorder=4,
            )
        circle = plt.Circle((0.0, 0.0), 1.0, color=circle_color, fill=False, linestyle="--", linewidth=1.2)
        ax.add_artist(circle)
        ax.set_aspect("equal", "box")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel("Re(z)")
        ax.set_ylabel("Im(z)")
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    for col, method_key in enumerate(methods):
        info = method_data[method_key]
        style = METHOD_STYLE[method_key]
        ax_p = axes[0, col]
        _plot_poles(
            ax_p,
            info["support_points"],
            info["poles"],
            f"{style['label']} poles ({len(info['support_points'])} pts)",
        )

        ax_e = axes[1, col]
        img = ax_e.imshow(
            info["error_grid"],
            origin="lower",
            extent=[-1, 1, -1, 1],
            cmap="magma",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(img, ax=ax_e, fraction=0.046, pad=0.04, label="log10 |error|")
        ax_e.set_xlabel("Re(z)")
        ax_e.set_ylabel("Im(z)")
        ax_e.set_title(f"{style['label']} error (max={info['max_error']:.2e})", fontsize=11)

    plt.tight_layout()
    fname = os.path.join(output_dir, f"poles_grid_{case_key}_d{d}_n{n}.png")
    print(f"Saving plot: {fname}", flush=True)
    fig.savefig(fname, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fname


def plot_poles_grid_panels(
    case_label,
    case_key,
    n,
    d,
    method_data,
    output_dir,
    subdir="poles_grid_panels",
    spurious_tol: float = 1e-6,
):
    panel_dir = ensure_dir(os.path.join(output_dir, subdir))
    methods = list(method_data.keys())

    vmin = None
    vmax = None
    all_vals = []
    for info in method_data.values():
        if info["error_grid"] is None:
            continue
        grid_vals = info["error_grid"]
        finite = grid_vals[np.isfinite(grid_vals)]
        if finite.size:
            all_vals.append(finite)
    if all_vals:
        merged = np.concatenate(all_vals)
        vmin = np.nanmin(merged)
        vmax = np.nanmax(merged)

    def _plot_poles_panel(ax, support, poles, true_poles=None, circle_color="#1f77b4"):
        support = np.asarray(support)
        poles = np.asarray(poles)
        true_poles = np.asarray(true_poles) if true_poles is not None else None
        ax.scatter(
            np.real(support),
            np.imag(support),
            s=26,
            c="#5c5c5c",
            alpha=0.6,
            marker="o",
            linewidths=0.0,
            rasterized=True,
            zorder=1,
        )
        if len(poles) > 0:
            ax.scatter(
                np.real(poles),
                np.imag(poles),
                s=50,
                facecolors="none",
                edgecolors="#d62728",
                linewidths=1.2,
                marker="o",
                alpha=0.9,
                zorder=3,
            )
            ax.scatter(
                np.real(poles),
                np.imag(poles),
                s=10,
                c="#d62728",
                marker="o",
                linewidths=0.0,
                alpha=0.6,
                zorder=4,
            )
            if true_poles is not None and len(true_poles) > 0:
                true_in = true_poles[np.abs(true_poles) <= 1.0]
                poles_in = poles[np.abs(poles) <= 1.0]
                if len(true_in) > 0 and len(poles_in) > 0:
                    dists = np.abs(poles_in[:, None] - true_in[None, :])
                    min_true = np.min(dists, axis=1)
                    spurious = poles_in[min_true > float(spurious_tol)]
                    if len(spurious) > 0:
                        ax.scatter(
                            np.real(spurious),
                            np.imag(spurious),
                            s=90,
                            c="#08306b",
                            marker="x",
                            linewidths=2.2,
                            alpha=0.95,
                            zorder=6,
                        )

        circle = plt.Circle((0.0, 0.0), 1.0, color=circle_color, fill=False, linestyle="--", linewidth=1.2)
        ax.add_artist(circle)
        ax.set_aspect("equal", "box")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

    legend_handles = None
    legend_labels = None
    for method_key in methods:
        info = method_data[method_key]

        fig, ax = plt.subplots(1, 1, figsize=(4.4, 4.4))
        fig.patch.set_facecolor("white")
        _plot_poles_panel(ax, info["support_points"], info["poles"], true_poles=info.get("true_poles"))
        if legend_handles is None:
            h_support = ax.scatter([], [], s=26, c="#5c5c5c", alpha=0.6, marker="o", label="support")
            h_poles = ax.scatter([], [], s=50, facecolors="none", edgecolors="#d62728",
                                 linewidths=1.2, marker="o", alpha=0.9, label="poles")
            h_spurious = ax.scatter([], [], s=90, c="#08306b", marker="x", linewidths=2.2, alpha=0.95, label="spurious poles")
            legend_handles, legend_labels = [h_support, h_poles, h_spurious], ["support", "poles", "spurious poles"]
        plt.tight_layout()
        fname = os.path.join(panel_dir, f"poles_grid_{case_key}_d{d}_n{n}_poles_{method_key}.png")
        print(f"Saving plot: {fname}", flush=True)
        fig.savefig(fname, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(4.4, 4.4))
        fig.patch.set_facecolor("white")
        img = ax.imshow(
            info["error_grid"],
            origin="lower",
            extent=[-1, 1, -1, 1],
            cmap="magma",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04, label=None)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        plt.tight_layout()
        fname = os.path.join(panel_dir, f"poles_grid_{case_key}_d{d}_n{n}_error_{method_key}.png")
        print(f"Saving plot: {fname}", flush=True)
        fig.savefig(fname, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    if legend_handles:
        fig_leg, ax_leg = plt.subplots(1, 1, figsize=(4.0, 1.4))
        ax_leg.axis("off")
        fig_leg.legend(
            legend_handles,
            legend_labels,
            loc="center",
            ncol=2,
            frameon=True,
            edgecolor="black",
            fontsize=12,
        )
        fig_leg.tight_layout()
        legend_path = os.path.join(panel_dir, f"poles_grid_{case_key}_d{d}_n{n}_legend.png")
        print(f"Saving plot: {legend_path}", flush=True)
        fig_leg.savefig(legend_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig_leg)
    return str(panel_dir)


def plot_poles_zoom_panels(
    case_label,
    case_key,
    n,
    d,
    method_data,
    output_dir,
    xlim=(0.75, 1.25),
    ylim=(-0.25, 0.25),
    all_points=None,
    subdir="poles_zoom_panels",
):
    panel_dir = ensure_dir(os.path.join(output_dir, subdir))
    methods = list(method_data.keys())
    for method_key in methods:
        info = method_data[method_key]
        support = np.asarray(info.get("support_points", []))
        poles = np.asarray(info.get("poles", []))
        true_poles = np.asarray(info.get("true_poles", [])) if info.get("true_poles") is not None else np.array([], dtype=np.complex128)

        fig, ax = plt.subplots(1, 1, figsize=(4.4, 4.4))
        fig.patch.set_facecolor("white")

        if all_points is not None:
            all_points_arr = np.asarray(all_points)
            unpicked_mask = ~np.isin(all_points_arr, support)
            unpicked = all_points_arr[unpicked_mask]
            if len(unpicked) > 0:
                ax.scatter(
                    np.real(unpicked),
                    np.imag(unpicked),
                    s=10,
                    c="#1f77b4",
                    alpha=0.10,
                    marker="o",
                    linewidths=0.0,
                    rasterized=True,
                    zorder=0,
                )

        ax.scatter(
            np.real(support),
            np.imag(support),
            s=52,
            c="#303030",
            alpha=0.92,
            marker="o",
            linewidths=0.4,
            edgecolors="white",
            rasterized=True,
            zorder=1,
        )
        if len(true_poles) > 0:
            ax.scatter(
                np.real(true_poles),
                np.imag(true_poles),
                s=34,
                c="#666666",
                alpha=0.55,
                marker="x",
                linewidths=1.6,
                zorder=2,
            )
        if len(poles) > 0:
            ax.scatter(
                np.real(poles),
                np.imag(poles),
                s=92,
                facecolors="none",
                edgecolors="#d62728",
                linewidths=1.8,
                marker="o",
                alpha=0.9,
                zorder=3,
            )
            ax.scatter(
                np.real(poles),
                np.imag(poles),
                s=22,
                c="#d62728",
                marker="o",
                linewidths=0.0,
                alpha=0.65,
                zorder=4,
            )
        circle = plt.Circle((0.0, 0.0), 1.0, color="#1f77b4", fill=False, linestyle="--", linewidth=1.1)
        ax.add_artist(circle)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

        plt.tight_layout()
        fname = os.path.join(panel_dir, f"poles_zoom_{case_key}_d{d}_n{n}_{method_key}.png")
        print(f"Saving plot: {fname}", flush=True)
        fig.savefig(fname, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    return str(panel_dir)


def plot_tan_pole_distance(case_label, case_key, n, d, method_data, output_dir):
    labels = []
    values_approx = []
    values_true = []
    approx_counts = []
    colors = []
    true_count = None
    for method_key, info in method_data.items():
        max_dist = info.get("max_pole_dist_to_true", np.nan)
        max_true = info.get("max_true_dist_to_approx", np.nan)
        if not (np.isfinite(max_dist) or np.isfinite(max_true)):
            continue
        style = METHOD_STYLE.get(method_key, {})
        labels.append(style.get("label", method_key))
        values_approx.append(float(max_dist) if np.isfinite(max_dist) else np.nan)
        values_true.append(float(max_true) if np.isfinite(max_true) else np.nan)
        approx_counts.append(float(info.get("approx_pole_count_in_disk", np.nan)))
        colors.append(style.get("color", "#333333"))
        if true_count is None:
            true_count = info.get("true_pole_count", None)
    if not values_approx and not values_true:
        return None
    x = np.arange(len(labels))
    offset = 0.18
    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)
    fig.patch.set_facecolor("white")
    ax = axes[0]
    ax.bar(x - offset, values_approx, width=width, color=colors, alpha=0.75, label="approx->true")
    ax.bar(x + offset, values_true, width=width, color=colors, alpha=0.45, label="true->approx")
    if np.any(np.array(values_approx) > 0) or np.any(np.array(values_true) > 0):
        ax.set_yscale("log")
    ax.set_ylabel("max distance to nearest pole (|z|<=1)")
    ax.set_title(f"{case_label} pole distances (d={d})", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    for idx, val in enumerate(values_approx):
        if np.isfinite(val):
            ax.text(idx - offset, val, f"{val:.2e}", ha="center", va="bottom", fontsize=9)
    for idx, val in enumerate(values_true):
        if np.isfinite(val):
            ax.text(idx + offset, val, f"{val:.2e}", ha="center", va="bottom", fontsize=9)
    ax.legend(fontsize=9)

    ax_counts = axes[1]
    ax_counts.bar(x, approx_counts, color=colors, alpha=0.6, label="approx poles")
    if true_count is not None and np.isfinite(true_count):
        ax_counts.axhline(true_count, color="#111111", linestyle="--", linewidth=1.2, label="true poles")
    ax_counts.set_ylabel("poles in disk")
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels(labels)
    ax_counts.grid(True, axis="y", alpha=0.3)
    ax_counts.legend(fontsize=9)
    plt.tight_layout()
    fname = os.path.join(output_dir, f"pole_distance_{case_key}_d{d}_n{n}.png")
    print(f"Saving plot: {fname}", flush=True)
    fig.savefig(fname, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fname


def run_case(
    d_values,
    n,
    grid_size,
    max_rank,
    block_size,
    rng_seed,
    cur_tol,
    cur_cleanup,
    cur_cleanup_tol,
    aaa_tol,
    aaa_cleanup,
    aaa_cleanup_tol,
    aaa_max_terms,
    pole_buffer,
    include_aaa=True,
    aaa_time_threshold=100.0,
    allow_extra_after_tol=False,
    cur_sampling_method="upper_bound",
    cur_residual_norm="frobenius",
    cur_split_percent=50.0,
    cur_force_side="auto",
):
    results = {
        "case": CASE_KEY,
        "label": CASE_LABEL,
        "domain": "disk",
        "d_values": list(d_values),
        "methods": {"rplu": [], "c2plu": [], "aaa": []},
    }
    if not include_aaa:
        results["methods"].pop("aaa")

    skip_aaa_after = False
    for d in d_values:
        f = f_tan_power_factory(d)
        rng = np.random.default_rng(rng_seed + int(d) + 10000)
        z_train = sample_unit_disk(n, rng)
        f_train = f(z_train)
        _, _, z_fine = build_eval_grid(grid_size, min_points=10 * n)

        for mode_key, mode in [("rplu", "random"), ("c2plu", "greedy")]:
            print(f"  CUR-Half ({mode}) d={d} ...", end=" ", flush=True)
            cur_start = time.time()
            cur_approx, cur_info = build_cur_half_approximant(
                z_train,
                f_train,
                f_func=None,
                relative_gb_tolerance=cur_tol,
                frob_norm_relative_tolerance=cur_tol,
                block_size=block_size,
                use_exact_norm=False,
                barnes_hut_theta=1.0,
                max_rank=max_rank,
                rng_seed=rng.integers(0, 2**31 - 1),
                sampling_method=cur_sampling_method,
                allow_extra_after_tol=allow_extra_after_tol,
                row_sampling=mode,
                column_sampling=mode,
                use_jax_svd=True,
                weight_method="svd_jax",
                chebfun_cleanup=cur_cleanup,
                chebfun_cleanup_tol=cur_cleanup_tol,
                residual_norm_kind=cur_residual_norm,
                split_percent=cur_split_percent,
                force_side=cur_force_side,
            )
            cur_time = time.time() - cur_start
            print(f"{cur_time:.2f}s", flush=True)
            cur_poles = cur_approx.poles()

            def cur_eval(z):
                return cur_approx(z)

            train_err = max_error(f, cur_eval, z_train, cur_poles, pole_buffer)
            fine_err = max_error(f, cur_eval, z_fine, cur_poles, pole_buffer)
            print(f"    train_max_error={train_err:.3e} fine_max_error={fine_err:.3e}", flush=True)
            results["methods"][mode_key].append(
                {
                    "d": float(d),
                    "time": float(cur_time),
                    "poles": int(len(cur_poles)),
                    "support_points": int(len(cur_approx.z_support)),
                    "train_max_error": float(train_err),
                    "fine_max_error": float(fine_err),
                    "side": cur_info.get("side", "?"),
                }
            )

        if include_aaa and not skip_aaa_after:
            print(f"  AAA (baryrat) d={d} ...", end=" ", flush=True)
            aaa_start = time.time()
            aaa_obj = _call_baryrat_aaa(
                z_train,
                f_train,
                tol=aaa_tol,
                max_terms=aaa_max_terms,
                cleanup=aaa_cleanup,
                cleanup_tol=aaa_cleanup_tol,
            )
            aaa_time = time.time() - aaa_start
            print(f"{aaa_time:.2f}s", flush=True)
            if aaa_time_threshold is not None and aaa_time > float(aaa_time_threshold):
                print(f"    AAA time {aaa_time:.2f}s exceeded threshold {aaa_time_threshold:.2f}s; skipping AAA for higher d.")
                skip_aaa_after = True

            eval_obj, support, values, weights = _extract_barycentric_fields(aaa_obj)
            if support is None or values is None or weights is None:
                raise RuntimeError("Unable to extract barycentric data from baryrat AAA.")

            def aaa_eval(z):
                if callable(eval_obj):
                    try:
                        return eval_obj(z)
                    except Exception:
                        pass
                return evaluate_barycentric(z, support, values, weights)

            aaa_poles = barycentric_poles(support, weights)
            print(f"    AAA poles: {len(aaa_poles)} (support points: {len(support)})")
            train_err = max_error(f, aaa_eval, z_train, aaa_poles, pole_buffer)
            fine_err = max_error(f, aaa_eval, z_fine, aaa_poles, pole_buffer)
            print(f"    train_max_error={train_err:.3e} fine_max_error={fine_err:.3e}", flush=True)

            results["methods"]["aaa"].append(
                {
                    "d": float(d),
                    "time": float(aaa_time),
                    "poles": int(len(aaa_poles)),
                    "support_points": int(len(support)),
                    "train_max_error": float(train_err),
                    "fine_max_error": float(fine_err),
                }
            )

    return results


def compute_poles_grid_case(
    d,
    n,
    grid_size,
    max_rank,
    block_size,
    rng_seed,
    cur_tol,
    cur_cleanup,
    cur_cleanup_tol,
    aaa_tol,
    aaa_cleanup,
    aaa_cleanup_tol,
    aaa_max_terms,
    pole_buffer,
    include_aaa=True,
    allow_extra_after_tol=False,
    cur_sampling_method="upper_bound",
    cur_residual_norm="frobenius",
    cur_split_percent=50.0,
    cur_force_side="auto",
):
    if not CUR_AVAILABLE or build_cur_half_approximant is None:
        raise RuntimeError("CUR-Half requires JAX; install JAX or run with `--from-json` to regenerate plots.")
    f = f_tan_power_factory(d)
    rng = np.random.default_rng(rng_seed + int(d) + 10000)
    z_train = sample_unit_disk(n, rng)
    f_train = f(z_train)
    Z_full, disk_mask, z_eval = build_eval_grid(grid_size, min_points=10 * n)

    def error_grid_for(eval_fn, poles):
        mask = mask_points(z_eval, poles, pole_buffer)
        if not np.any(mask):
            return None, np.nan
        f_true = f(z_eval[mask])
        f_pred = eval_fn(z_eval[mask])
        err = np.abs(f_true - f_pred)
        finite = np.isfinite(err)
        max_err = float(np.max(err[finite])) if np.any(finite) else np.nan
        err_full = np.full(len(z_eval), np.nan)
        err_full[mask] = err
        err_full[~np.isfinite(err_full)] = np.nan
        log_err = np.log10(np.maximum(err_full, 1e-16))
        log_err[~np.isfinite(err_full)] = np.nan
        grid = np.full(Z_full.shape, np.nan)
        grid[disk_mask] = log_err
        return grid, max_err

    true_poles = _tan_true_poles_in_disk(d, radius=1.0)
    pole_metrics = {
        "true_pole_count": int(len(true_poles)),
        "approx_pole_count_in_disk": {},
        "max_pole_dist_to_true": {},
        "max_true_dist_to_approx": {},
    }

    method_data = {}
    for mode_key, mode in [("rplu", "random"), ("c2plu", "greedy")]:
        cur_approx, cur_info = build_cur_half_approximant(
            z_train,
            f_train,
            f_func=None,
            relative_gb_tolerance=cur_tol,
            frob_norm_relative_tolerance=cur_tol,
            block_size=block_size,
            use_exact_norm=False,
            barnes_hut_theta=1.0,
            max_rank=max_rank,
            rng_seed=rng.integers(0, 2**31 - 1),
            sampling_method=cur_sampling_method,
            allow_extra_after_tol=allow_extra_after_tol,
            row_sampling=mode,
            column_sampling=mode,
            use_jax_svd=True,
            weight_method="svd_jax",
            chebfun_cleanup=cur_cleanup,
            chebfun_cleanup_tol=cur_cleanup_tol,
            residual_norm_kind=cur_residual_norm,
            split_percent=cur_split_percent,
            force_side=cur_force_side,
        )
        poles = cur_approx.poles()

        def cur_eval(z):
            return cur_approx(z)

        grid, max_err = error_grid_for(cur_eval, poles)
        method_data[mode_key] = {
            "support_points": cur_approx.z_support,
            "poles": poles,
            "error_grid": grid,
            "max_error": max_err,
            "rank": cur_info.get("rank") if isinstance(cur_info, dict) else None,
            "side": cur_info.get("side") if isinstance(cur_info, dict) else None,
            "true_poles": true_poles,
        }
        stats = _pole_distance_stats(poles, true_poles, radius=1.0)
        method_data[mode_key]["max_pole_dist_to_true"] = stats["max_pole_dist_to_true"]
        method_data[mode_key]["max_true_dist_to_approx"] = stats["max_true_dist_to_approx"]
        method_data[mode_key]["true_pole_count"] = stats["true_poles_in_disk"]
        method_data[mode_key]["approx_pole_count_in_disk"] = stats["approx_poles_in_disk"]
        pole_metrics["approx_pole_count_in_disk"][mode_key] = stats["approx_poles_in_disk"]
        pole_metrics["max_pole_dist_to_true"][mode_key] = stats["max_pole_dist_to_true"]
        pole_metrics["max_true_dist_to_approx"][mode_key] = stats["max_true_dist_to_approx"]

    if include_aaa:
        if not BARYCENTRIC_AVAILABLE or barycentric_poles is None or evaluate_barycentric is None:
            raise RuntimeError("AAA pole/error plots require `cauchy.barycentric` (and its dependencies).")
        aaa_obj = _call_baryrat_aaa(
            z_train,
            f_train,
            tol=aaa_tol,
            max_terms=aaa_max_terms,
            cleanup=aaa_cleanup,
            cleanup_tol=aaa_cleanup_tol,
        )
        eval_obj, support, values, weights = _extract_barycentric_fields(aaa_obj)
        if support is None or values is None or weights is None:
            raise RuntimeError("Unable to extract barycentric data from baryrat AAA.")

        def aaa_eval(z):
            if callable(eval_obj):
                try:
                    return eval_obj(z)
                except Exception:
                    pass
            return evaluate_barycentric(z, support, values, weights)

        poles = barycentric_poles(support, weights)
        grid, max_err = error_grid_for(aaa_eval, poles)
        method_data["aaa"] = {
            "support_points": support,
            "poles": poles,
            "error_grid": grid,
            "max_error": max_err,
            "true_poles": true_poles,
        }
        stats = _pole_distance_stats(poles, true_poles, radius=1.0)
        method_data["aaa"]["max_pole_dist_to_true"] = stats["max_pole_dist_to_true"]
        method_data["aaa"]["max_true_dist_to_approx"] = stats["max_true_dist_to_approx"]
        method_data["aaa"]["true_pole_count"] = stats["true_poles_in_disk"]
        method_data["aaa"]["approx_pole_count_in_disk"] = stats["approx_poles_in_disk"]
        pole_metrics["approx_pole_count_in_disk"]["aaa"] = stats["approx_poles_in_disk"]
        pole_metrics["max_pole_dist_to_true"]["aaa"] = stats["max_pole_dist_to_true"]
        pole_metrics["max_true_dist_to_approx"]["aaa"] = stats["max_true_dist_to_approx"]

    print(f"  True tan poles inside unit disk: {len(true_poles)}")
    for method_key, info in method_data.items():
        label = METHOD_STYLE.get(method_key, {}).get("label", method_key)
        approx_count = info.get("approx_pole_count_in_disk", np.nan)
        max_dist = info.get("max_pole_dist_to_true", np.nan)
        max_true = info.get("max_true_dist_to_approx", np.nan)
        print(f"    {label} poles in disk: {approx_count}")
        if np.isfinite(max_dist):
            print(f"      max approx->true distance: {max_dist:.3e}")
        else:
            print("      max approx->true distance: n/a")
        if np.isfinite(max_true):
            print(f"      max true->approx distance: {max_true:.3e}")
        else:
            print("      max true->approx distance: n/a")
    return method_data, pole_metrics, z_train


def comma_int_list(val):
    return [int(x) for x in val.split(",")] if val else []


def comma_str_list(val):
    return [x.strip() for x in val.split(",") if x.strip()] if val else []


def main():
    parser = argparse.ArgumentParser(description="Baryrat AAA vs CUR-Half for tan(sqrt(d)*z^sqrt(d))")
    parser.add_argument("--from-json", type=str, default=None,
                        help="Load results JSON and regenerate plots without rerunning experiments.")
    parser.add_argument("--n", type=int, default=200000)
    parser.add_argument("--tan-d-values", type=comma_int_list, default=(np.arange(1, 21, 1) ** 2).tolist(),
                        help="Comma-separated d values for tan case.")
    parser.add_argument("--grid-size", type=int, default=400)
    parser.add_argument("--max-rank", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--rng-seed", type=int, default=123)
    parser.add_argument("--output-dir", type=str, default="plots/baryrat_cur_half")
    parser.add_argument("--overwrite-json", action="store_true", default=False,
                        help="Allow overwriting the output results JSON (default: write a new *_k.json instead).")
    parser.add_argument("--pole-buffer", type=float, default=0.0)
    parser.add_argument("--tan-pole-d", type=float, default=400,
                        help="d value for tan poles/error plots (default: 400)")
    parser.add_argument("--cur-tol", type=float, default=1e-11)
    parser.add_argument("--cur-cleanup", action="store_true",
                        help="Enable Chebfun-style cleanup for CUR-Half (default off).")
    parser.add_argument("--cur-cleanup-tol", type=float, default=None,
                        help="CUR cleanup tolerance (default: cur-tol).")

    parser.add_argument("--aaa-tol", type=float, default=1e-11)
    parser.add_argument("--aaa-cleanup", action="store_true",
                        help="Enable baryrat AAA cleanup (default off).")
    parser.add_argument("--aaa-cleanup-tol", type=float, default=None,
                        help="AAA cleanup tolerance (default: aaa-tol).")
    parser.add_argument("--aaa-max-terms", type=int, default=600,
                        help="AAA max support points/iterations (default: max-rank).")
    parser.add_argument("--aaa-time-threshold", type=float, default=30000.0,
                        help="Stop AAA for higher d if a run exceeds this many seconds.")
    parser.add_argument("--skip-aaa", action="store_true",
                        help="Skip baryrat AAA (run CUR-Half only).")
    parser.add_argument("--tasks", type=comma_str_list, default=None,
                        help="Comma-separated tasks: summary,poles (default: summary,poles).")
    parser.add_argument("--cur-sampling-method", type=str, default="upper_bound",
                        choices=["upper_bound", "rejection", "generator_norm"],
                        help="CUR-Half sampling backend (default: upper_bound).")
    parser.add_argument("--cur-extra-after-tol", dest="cur_allow_extra_after_tol", action="store_true",
                        default=False,
                        help="Allow one extra CUR-Half pivot after crossing tolerance (default: disabled).")
    parser.add_argument("--cur-residual-norm", type=str, default="max_row",
                        choices=["frobenius", "max_row"],
                        help="CUR-Half stopping norm (default: max_row).")
    parser.add_argument("--split", type=float, default=50.0,
                        help="Percent of points in x (rows) for CUR-Half (default: 50).")
    parser.add_argument("--cur-side", type=str, default="auto",
                        choices=["auto", "x", "y"],
                        help="Force CUR-Half support side (default: auto).")
    parser.add_argument("--poles-subdir", type=str, default="poles_grid_panels",
                        help="Subdirectory for poles_grid subplot panels.")
    args = parser.parse_args()

    if args.from_json:
        with open(args.from_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        set_plot_style()
        output_dir = ensure_dir(args.output_dir)

        case_results = None
        if "cases" in data and isinstance(data["cases"], dict):
            case_results = data["cases"].get(CASE_KEY)
            if case_results is None and data["cases"]:
                # Backward-compatible fallback: use the first case in JSON.
                case_results = next(iter(data["cases"].values()))
        elif "summary" in data:
            case_results = data["summary"]

        if not isinstance(case_results, dict):
            raise ValueError("No tan summary case found in JSON.")
        if case_results.get("skipped_summary"):
            print("No summary plots generated: summary was skipped in JSON.", flush=True)
            return

        n_val = data.get("n", args.n)
        summary_path = plot_summary(
            case_label=case_results.get("label", CASE_LABEL),
            case_key=CASE_KEY,
            n=n_val,
            results=case_results,
            output_dir=str(output_dir),
        )
        print(f"Saved summary plot: {summary_path}")
        panel_paths = plot_summary_panels(
            case_label=case_results.get("label", CASE_LABEL),
            case_key=CASE_KEY,
            n=n_val,
            results=case_results,
            output_dir=str(output_dir),
        )
        for panel_path in panel_paths:
            print(f"Saved summary panel: {panel_path}")
        return

    if not CUR_AVAILABLE:
        raise RuntimeError("JAX/CUR is required to run experiments. Use `--from-json` to only regenerate plots.")
    if not BARYRAT_AVAILABLE:
        raise RuntimeError("baryrat is required. Install with `pip install baryrat`.")

    if not args.tan_d_values:
        raise ValueError("--tan-d-values must include at least one value.")

    tasks = args.tasks or ["summary", "poles"]
    tasks = [t.strip().lower() for t in tasks if t and t.strip()]
    if not tasks:
        raise ValueError("--tasks must include at least one of: summary, poles")
    for task in tasks:
        if task not in {"summary", "poles"}:
            raise ValueError("--tasks must be a subset of: summary, poles")

    if args.aaa_max_terms is None:
        aaa_max_terms = args.max_rank
    else:
        aaa_max_terms = args.aaa_max_terms

    aaa_time_threshold = args.aaa_time_threshold
    cur_cleanup_tol = args.cur_cleanup_tol if args.cur_cleanup_tol is not None else args.cur_tol
    aaa_cleanup_tol = args.aaa_cleanup_tol if args.aaa_cleanup_tol is not None else args.aaa_tol
    tan_pole_d = args.tan_pole_d

    supports_cleanup, supports_cleanup_tol = baryrat_cleanup_support()
    if args.aaa_cleanup and not supports_cleanup:
        print("Warning: baryrat AAA does not expose a cleanup flag; cleanup will be ignored.")
    if args.aaa_cleanup_tol is not None and not supports_cleanup_tol:
        print("Warning: baryrat AAA does not expose cleanup_tol; cleanup tol will be ignored.")

    set_plot_style()
    output_dir = ensure_dir(args.output_dir)

    all_results = {
        "n": args.n,
        "tan_d_values": args.tan_d_values,
        "grid_size": args.grid_size,
        "max_rank": args.max_rank,
        "block_size": args.block_size,
        "cur_tol": args.cur_tol,
        "cur_cleanup": args.cur_cleanup,
        "cur_cleanup_tol": cur_cleanup_tol,
        "cur_allow_extra_after_tol": bool(args.cur_allow_extra_after_tol),
        "cur_residual_norm": args.cur_residual_norm,
        "cur_split_percent": float(args.split),
        "cur_force_side": args.cur_side,
        "aaa_tol": args.aaa_tol,
        "aaa_cleanup": args.aaa_cleanup,
        "aaa_cleanup_tol": aaa_cleanup_tol,
        "aaa_max_terms": aaa_max_terms,
        "aaa_time_threshold": aaa_time_threshold,
        "skip_aaa": bool(args.skip_aaa),
        "aaa_cleanup_supported": bool(supports_cleanup),
        "aaa_cleanup_tol_supported": bool(supports_cleanup_tol),
        "tasks": tasks,
        "cases": {},
    }

    print(f"\n=== Case: {CASE_LABEL} ===")

    if "summary" in tasks:
        case_results = run_case(
            d_values=args.tan_d_values,
            n=args.n,
            grid_size=args.grid_size,
            max_rank=args.max_rank,
            block_size=args.block_size,
            rng_seed=args.rng_seed,
            cur_tol=args.cur_tol,
            cur_cleanup=args.cur_cleanup,
            cur_cleanup_tol=cur_cleanup_tol,
            aaa_tol=args.aaa_tol,
            aaa_cleanup=args.aaa_cleanup,
            aaa_cleanup_tol=aaa_cleanup_tol,
            aaa_max_terms=aaa_max_terms,
            pole_buffer=args.pole_buffer,
            include_aaa=not args.skip_aaa,
            aaa_time_threshold=aaa_time_threshold,
            allow_extra_after_tol=args.cur_allow_extra_after_tol,
            cur_sampling_method=args.cur_sampling_method,
            cur_residual_norm=args.cur_residual_norm,
            cur_split_percent=args.split,
            cur_force_side=args.cur_side,
        )
        all_results["cases"][CASE_KEY] = case_results

        summary_path = plot_summary(
            case_label=CASE_LABEL,
            case_key=CASE_KEY,
            n=args.n,
            results=case_results,
            output_dir=str(output_dir),
        )
        print(f"Saved summary plot: {summary_path}")
        panel_paths = plot_summary_panels(
            case_label=CASE_LABEL,
            case_key=CASE_KEY,
            n=args.n,
            results=case_results,
            output_dir=str(output_dir),
        )
        for panel_path in panel_paths:
            print(f"Saved summary panel: {panel_path}")
    else:
        all_results["cases"][CASE_KEY] = {"skipped_summary": True, "label": CASE_LABEL}

    if "poles" in tasks:
        pole_data, pole_metrics, pole_train = compute_poles_grid_case(
            d=tan_pole_d,
            n=args.n,
            grid_size=args.grid_size,
            max_rank=args.max_rank,
            block_size=args.block_size,
            rng_seed=args.rng_seed + 123,
            cur_tol=args.cur_tol,
            cur_cleanup=args.cur_cleanup,
            cur_cleanup_tol=cur_cleanup_tol,
            aaa_tol=args.aaa_tol,
            aaa_cleanup=args.aaa_cleanup,
            aaa_cleanup_tol=aaa_cleanup_tol,
            aaa_max_terms=aaa_max_terms,
            pole_buffer=args.pole_buffer,
            include_aaa=not args.skip_aaa,
            allow_extra_after_tol=args.cur_allow_extra_after_tol,
            cur_sampling_method=args.cur_sampling_method,
            cur_residual_norm=args.cur_residual_norm,
            cur_split_percent=args.split,
            cur_force_side=args.cur_side,
        )
        all_results["cases"][CASE_KEY]["pole_metrics"] = pole_metrics
        all_results["cases"][CASE_KEY]["pole_data"] = _extract_pole_data_for_json(
            d=tan_pole_d,
            method_data=pole_data,
            train_points=pole_train,
        )
        pole_path = plot_poles_and_error(
            case_label=CASE_LABEL,
            case_key=CASE_KEY,
            n=args.n,
            d=tan_pole_d,
            method_data=pole_data,
            output_dir=str(output_dir),
        )
        print(f"Saved poles/error plot: {pole_path}")
        plot_poles_grid_panels(
            case_label=CASE_LABEL,
            case_key=CASE_KEY,
            n=args.n,
            d=tan_pole_d,
            method_data=pole_data,
            output_dir=str(output_dir),
            subdir=args.poles_subdir,
        )
        zoom_dir = plot_poles_zoom_panels(
            case_label=CASE_LABEL,
            case_key=CASE_KEY,
            n=args.n,
            d=tan_pole_d,
            method_data=pole_data,
            output_dir=str(output_dir),
            xlim=(0.75, 1.25),
            ylim=(-0.25, 0.25),
            all_points=pole_train,
        )
        if zoom_dir:
            print(f"Saved tan poles-zoom panels: {zoom_dir}")
        dist_path = plot_tan_pole_distance(
            case_label=CASE_LABEL,
            case_key=CASE_KEY,
            n=args.n,
            d=tan_pole_d,
            method_data=pole_data,
            output_dir=str(output_dir),
        )
        if dist_path:
            print(f"Saved tan pole-distance plot: {dist_path}")

    out_json = output_dir / f"baryrat_cur_half_results_n{args.n}.json"
    if out_json.exists() and not args.overwrite_json:
        new_path = _next_available_path(out_json)
        print(f"Note: {out_json} exists; writing to {new_path} (use --overwrite-json to overwrite).")
        out_json = new_path
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    print(f"\nSaved results: {out_json}")


if __name__ == "__main__":
    main()
