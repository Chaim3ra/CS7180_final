"""src/plot_results.py — Generate presentation-ready plots from results/all_results.csv.

Saves four PNG plots and four companion CSVs to results/plots/.

Usage
-----
    python src/plot_results.py
    python src/plot_results.py --results results/all_results.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PLOTS_DIR = ROOT / "results" / "plots"

# Consistent color/style scheme
_COLORS = {"v1": "#1f77b4", "v2": "#ff7f0e"}   # blue / orange
_ALPHA_ZERO   = 1.0
_ALPHA_FINE   = 0.85
_LS_ZERO      = "-"
_LS_FINE      = "--"
_MARKER_ZERO  = "o"
_MARKER_FINE  = "s"


def _fmt_float(v) -> str:
    try:
        f = float(v)
        return "" if math.isnan(f) else str(f)
    except (TypeError, ValueError):
        return str(v) if v is not None else ""


def _is_nan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def _load_results(csv_path: Path):
    """Load all_results.csv as a list of dicts with numeric coercion."""
    import csv
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ("mae", "rmse", "mape", "r2", "skill_score", "peak_mae",
                      "generalization_gap", "ny_days"):
                try:
                    r[k] = float(r[k]) if r.get(k, "") != "" else float("nan")
                except (ValueError, TypeError):
                    r[k] = float("nan")
            rows.append(r)
    return rows


def _save_df(df, path: Path) -> None:
    import csv
    if not df:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=df[0].keys())
        writer.writeheader()
        writer.writerows(df)
    print(f"  Saved CSV: {path.relative_to(ROOT)}", flush=True)


def _savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved PNG: {path.relative_to(ROOT)}", flush=True)


def _apply_style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)


# ---------------------------------------------------------------------------
# Plot 1 — Data efficiency curve (MAE vs ny_days)
# ---------------------------------------------------------------------------

def plot_data_efficiency_curve(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    versions = sorted({r["model_version"] for r in rows if r.get("model_version")})
    fig, ax  = plt.subplots(figsize=(7, 5))

    csv_rows: list[dict] = []
    plotted = 0

    for version in versions:
        color = _COLORS.get(version, "gray")
        # Zero-shot point (ny_days == 0)
        zs = [r for r in rows if r["model_version"] == version and r["experiment"] == "zero_shot"]
        # Fine-tuned points
        ft = sorted(
            [r for r in rows if r["model_version"] == version and "finetune" in r.get("experiment", "")],
            key=lambda r: r["ny_days"],
        )
        if not zs and not ft:
            continue

        x_pts, y_pts = [], []
        if zs:
            x_pts.append(0)
            y_pts.append(zs[0]["mae"])
        for r in ft:
            if not _is_nan(r["ny_days"]) and not _is_nan(r["mae"]):
                x_pts.append(int(r["ny_days"]))
                y_pts.append(r["mae"])

        if len(x_pts) < 1:
            continue

        ax.plot(x_pts, y_pts, color=color, marker="o", linewidth=2,
                label=f"{version.upper()}", linestyle="-")

        for xv, yv in zip(x_pts, y_pts):
            csv_rows.append({
                "model_version": version,
                "ny_days": xv,
                "mae": _fmt_float(yv),
                "rmse": _fmt_float(next((r["rmse"] for r in rows
                                         if r["model_version"] == version
                                         and (xv == 0 and r["experiment"] == "zero_shot"
                                              or int(r.get("ny_days", -1)) == xv
                                              and "finetune" in r.get("experiment", ""))), float("nan"))),
                "r2":          "",
                "skill_score": "",
                "peak_mae":    "",
            })
        plotted += 1

    if plotted == 0:
        print("  [SKIP] data_efficiency_curve: no suitable data yet", flush=True)
        plt.close(fig)
        return

    ax.set_xticks([0, 7, 30, 90])
    ax.set_xticklabels(["Zero-shot\n(0 days)", "7 days", "30 days", "90 days"])
    _apply_style(ax, "Data Efficiency: MAE vs NY Fine-Tuning Days",
                 "NY Days Used for Fine-Tuning", "MAE (kWh)")
    ax.legend(title="Model", fontsize=10)
    _savefig(fig, PLOTS_DIR / "data_efficiency_curve.png")
    plt.close(fig)
    _save_df(csv_rows, PLOTS_DIR / "data_efficiency_curve.csv")


# ---------------------------------------------------------------------------
# Plot 2 — Metric comparison bar chart
# ---------------------------------------------------------------------------

def plot_metric_comparison(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ["mae", "rmse", "mape", "r2", "skill_score", "peak_mae"]
    filtered = [r for r in rows if not _is_nan(r.get("mae", float("nan")))]
    if not filtered:
        print("  [SKIP] metric_comparison: no data yet", flush=True)
        return

    experiments = sorted({r["experiment"] for r in filtered})
    x = np.arange(len(metrics))
    width = 0.8 / max(len(experiments), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    csv_rows: list[dict] = []

    for i, exp in enumerate(experiments):
        exp_rows = [r for r in filtered if r["experiment"] == exp]
        if not exp_rows:
            continue
        r = exp_rows[0]
        version = r.get("model_version", "v1")
        color   = _COLORS.get(version, "gray")
        alpha   = _ALPHA_ZERO if "finetune" not in exp else _ALPHA_FINE
        vals    = [r.get(m, float("nan")) for m in metrics]
        offsets = x + (i - len(experiments) / 2 + 0.5) * width
        bars    = [v for v in vals if not _is_nan(v)]
        ax.bar(offsets[[j for j, v in enumerate(vals) if not _is_nan(v)]],
               bars, width * 0.9, label=f"{version} {exp}", color=color, alpha=alpha)

        for m, v in zip(metrics, vals):
            csv_rows.append({
                "model_version": version,
                "experiment":    exp,
                "metric_name":   m,
                "metric_value":  _fmt_float(v),
            })

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    _apply_style(ax, "Metric Comparison Across Experiments", "Metric", "Value")
    ax.legend(fontsize=8, ncol=2)
    _savefig(fig, PLOTS_DIR / "metric_comparison.png")
    plt.close(fig)
    _save_df(csv_rows, PLOTS_DIR / "metric_comparison.csv")


# ---------------------------------------------------------------------------
# Plot 3 — Generalization gap
# ---------------------------------------------------------------------------

def plot_generalization_gap(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    # Collect pairs: in-region MAE + out-region MAE
    versions = sorted({r["model_version"] for r in rows if r.get("model_version")})
    csv_rows: list[dict] = []
    data: list[tuple] = []  # (version, experiment, in_mae, out_mae)

    for version in versions:
        zs = [r for r in rows if r["model_version"] == version and r["experiment"] == "zero_shot"]
        ir_keys = [k for k in ("in_region_tx", "in_region_ca", "in_region") if
                   any(r["experiment"] == k and r["model_version"] == version for r in rows)]
        if zs:
            out_mae = zs[0]["mae"]
            in_mae_candidates = [r["mae"] for r in rows
                                 if r["model_version"] == version and r["experiment"] in ir_keys
                                 and not _is_nan(r["mae"])]
            in_mae = float(np.mean(in_mae_candidates)) if in_mae_candidates else float("nan")
            if not _is_nan(out_mae):
                data.append((version, "zero_shot", in_mae, out_mae))

        ft_rows = sorted(
            [r for r in rows if r["model_version"] == version and "finetune" in r.get("experiment", "")],
            key=lambda r: r["ny_days"],
        )
        for r in ft_rows:
            if not _is_nan(r["mae"]):
                data.append((version, r["experiment"], in_mae if not _is_nan(in_mae) else float("nan"), r["mae"]))

    if not data:
        print("  [SKIP] generalization_gap: no data yet", flush=True)
        return

    labels  = [f"{d[0]} {d[1]}" for d in data]
    out_maes = [d[3] for d in data]
    in_maes  = [d[2] for d in data]
    x        = np.arange(len(labels))
    w        = 0.35
    colors   = [_COLORS.get(d[0], "gray") for d in data]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    ax.bar(x - w / 2, in_maes,  w, label="In-region MAE",  color="steelblue", alpha=0.85)
    ax.bar(x + w / 2, out_maes, w, label="Out-region MAE", color="tomato",    alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    _apply_style(ax, "Generalisation Gap: In-Region vs Out-Region MAE",
                 "Experiment", "MAE (kWh)")
    ax.legend(fontsize=10)
    _savefig(fig, PLOTS_DIR / "generalization_gap.png")
    plt.close(fig)

    for d in data:
        gap = d[3] - d[2] if not _is_nan(d[2]) else float("nan")
        csv_rows.append({
            "model_version": d[0], "experiment": d[1],
            "in_region_mae": _fmt_float(d[2]), "out_region_mae": _fmt_float(d[3]),
            "gap": _fmt_float(gap),
        })
    _save_df(csv_rows, PLOTS_DIR / "generalization_gap.csv")


# ---------------------------------------------------------------------------
# Plot 4 — Skill score curve
# ---------------------------------------------------------------------------

def plot_skill_score_curve(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    versions = sorted({r["model_version"] for r in rows if r.get("model_version")})
    fig, ax  = plt.subplots(figsize=(7, 5))
    csv_rows: list[dict] = []
    plotted = 0

    for version in versions:
        color = _COLORS.get(version, "gray")
        zs = [r for r in rows if r["model_version"] == version and r["experiment"] == "zero_shot"]
        ft = sorted(
            [r for r in rows if r["model_version"] == version and "finetune" in r.get("experiment", "")],
            key=lambda r: r["ny_days"],
        )
        x_pts, y_pts, pers_pts, model_pts = [], [], [], []
        if zs and not _is_nan(zs[0].get("skill_score", float("nan"))):
            x_pts.append(0)
            y_pts.append(zs[0]["skill_score"])
        for r in ft:
            if not _is_nan(r.get("skill_score", float("nan"))):
                x_pts.append(int(r["ny_days"]))
                y_pts.append(r["skill_score"])
        if not x_pts:
            continue
        ax.plot(x_pts, y_pts, color=color, marker="o", linewidth=2, label=f"{version.upper()}")
        plotted += 1
        for xv, yv in zip(x_pts, y_pts):
            csv_rows.append({
                "model_version": version, "ny_days": xv,
                "skill_score": _fmt_float(yv),
                "persistence_mae": "",
                "model_mae": "",
            })

    if plotted == 0:
        print("  [SKIP] skill_score_curve: no skill_score data yet", flush=True)
        plt.close(fig)
        return

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", label="Persistence baseline")
    ax.set_xticks([0, 7, 30, 90])
    ax.set_xticklabels(["Zero-shot\n(0 days)", "7 days", "30 days", "90 days"])
    _apply_style(ax, "Skill Score vs NY Fine-Tuning Days",
                 "NY Days Used for Fine-Tuning", "Skill Score vs Persistence (%)")
    ax.legend(title="Model", fontsize=10)
    _savefig(fig, PLOTS_DIR / "skill_score_curve.png")
    plt.close(fig)
    _save_df(csv_rows, PLOTS_DIR / "skill_score_curve.csv")


# ---------------------------------------------------------------------------
# README for plots dir
# ---------------------------------------------------------------------------

def write_plots_readme() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    readme = PLOTS_DIR / "README.md"
    readme.write_text(
        "# results/plots\n\n"
        "Each `.png` has a companion `.csv` with the exact data used to generate it.\n"
        "Re-create any plot without re-running the model:\n\n"
        "| PNG | CSV | Description |\n"
        "|-----|-----|-------------|\n"
        "| data_efficiency_curve.png | data_efficiency_curve.csv | MAE vs NY fine-tuning days (0/7/30/90) |\n"
        "| metric_comparison.png | metric_comparison.csv | All 7 metrics, all experiments, side-by-side bars |\n"
        "| generalization_gap.png | generalization_gap.csv | In-region vs out-region MAE per experiment |\n"
        "| skill_score_curve.png | skill_score_curve.csv | Skill score vs persistence, by fine-tuning days |\n\n"
        "**Color scheme:** M1 (real data only) = blue (#1f77b4) · M2 (real + synthetic) = orange (#ff7f0e)\n"
        "**Line style:** Zero-shot = solid · Fine-tuned = dashed\n",
        encoding="utf-8",
    )
    print(f"  Wrote {readme.relative_to(ROOT)}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate result plots from all_results.csv.")
    parser.add_argument("--results", default="results/all_results.csv",
                        help="Path to all_results.csv (relative to repo root).")
    args = parser.parse_args()

    csv_path = ROOT / args.results
    if not csv_path.exists():
        print(f"  [ERROR] {csv_path} not found. Run evaluate.py or finetune.py first.", flush=True)
        sys.exit(1)

    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend

    rows = _load_results(csv_path)
    print(f"  Loaded {len(rows)} rows from {csv_path.relative_to(ROOT)}", flush=True)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    write_plots_readme()

    plot_data_efficiency_curve(rows)
    plot_metric_comparison(rows)
    plot_generalization_gap(rows)
    plot_skill_score_curve(rows)

    print("\n  Done — plots and CSVs saved to results/plots/", flush=True)


if __name__ == "__main__":
    main()
