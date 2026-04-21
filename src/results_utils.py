"""Shared utilities: save evaluation rows to CSV/MD and auto-commit to git."""
from __future__ import annotations

import csv
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RESULTS_CSV    = ROOT / "results" / "all_results.csv"
RESULTS_MD     = ROOT / "results" / "all_results.md"
PER_HOME_CSV   = ROOT / "results" / "per_home_metrics.csv"
PLOTS_DIR      = ROOT / "results" / "plots"

CSV_COLUMNS = [
    "model_version", "experiment", "ny_days",
    "mae", "rmse", "mape", "r2", "skill_score", "peak_mae",
    "generalization_gap", "epoch_stopped", "timestamp", "checkpoint_s3_path",
]

PER_HOME_COLUMNS = [
    "model_version", "experiment", "ny_days",
    "dataid", "mae", "r2", "n_windows", "timestamp",
]

_METRIC_DESCS = {
    "mae":                "Mean absolute error (kWh) — average magnitude of prediction error",
    "rmse":               "Root mean squared error (kWh) — penalises large errors more than MAE",
    "mape":               "Mean absolute percentage error (%) — relative error; zero-actual steps excluded",
    "r2":                 "Coefficient of determination — fraction of variance explained (1.0 = perfect)",
    "skill_score":        "Skill score vs persistence (%) — positive means better than same-time-yesterday",
    "peak_mae":           "Peak-hour MAE (kWh) — MAE restricted to 8 am–4 pm solar production hours",
    "generalization_gap": "Generalisation gap (kWh) — out-of-region MAE minus in-region MAE",
}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(v, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    try:
        return f"{float(v):.{decimals}f}"
    except (ValueError, TypeError):
        return str(v)


def save_row(row: dict) -> None:
    """Append or overwrite one row in all_results.csv (keyed on model_version+experiment+ny_days)."""
    key = (str(row.get("model_version", "")), str(row.get("experiment", "")), str(row.get("ny_days", "")))
    existing = _read_csv(RESULTS_CSV)

    replaced = False
    new_rows: list[dict] = []
    for r in existing:
        rkey = (str(r.get("model_version", "")), str(r.get("experiment", "")), str(r.get("ny_days", "")))
        if rkey == key:
            new_rows.append({c: row.get(c, "") for c in CSV_COLUMNS})
            replaced = True
        else:
            new_rows.append(r)

    if not replaced:
        new_rows.append({c: row.get(c, "") for c in CSV_COLUMNS})

    _write_csv(RESULTS_CSV, CSV_COLUMNS, new_rows)
    action = "Updated" if replaced else "Appended"
    print(f"  {action} row in {RESULTS_CSV.relative_to(ROOT)}", flush=True)
    generate_markdown()


def generate_markdown() -> None:
    """Regenerate all_results.md from all_results.csv."""
    rows = _read_csv(RESULTS_CSV)
    if not rows:
        return

    try:
        import pandas as pd  # type: ignore[import]
        df = pd.DataFrame(rows)
        for c in ("mae", "rmse", "mape", "r2", "skill_score", "peak_mae", "generalization_gap"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df_sorted = df.sort_values("mae", ascending=True, na_position="last")
    except Exception:
        df_sorted = None

    lines = [
        "# Solar Forecasting — All Experiment Results\n\n",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n",
        "## Metric Definitions\n\n",
    ]
    for metric, desc in _METRIC_DESCS.items():
        lines.append(f"- **{metric}**: {desc}\n")
    lines.append("\n")

    if df_sorted is not None and not df_sorted.empty:
        display_cols = ["model_version", "experiment", "ny_days", "mae", "rmse", "r2", "skill_score", "peak_mae"]
        display_cols = [c for c in display_cols if c in df_sorted.columns]
        lines.append("## Summary Table (sorted by MAE)\n\n")
        lines.append("| " + " | ".join(display_cols) + " |\n")
        lines.append("|" + "|".join("---" for _ in display_cols) + "|\n")
        for _, r in df_sorted.iterrows():
            vals = []
            for c in display_cols:
                v = r.get(c, "")
                if c in ("mae", "rmse", "peak_mae"):
                    vals.append(_fmt(v, 4))
                elif c in ("r2", "skill_score"):
                    vals.append(_fmt(v, 3))
                else:
                    vals.append(str(v) if str(v) != "nan" else "")
            lines.append("| " + " | ".join(vals) + " |\n")
        lines.append("\n")

        for version in sorted(df_sorted["model_version"].dropna().unique()):
            lines.append(f"## {version}\n\n")
            vdf = df_sorted[df_sorted["model_version"] == version]
            for _, r in vdf.iterrows():
                lines.append(f"### {r.get('experiment','')} (ny_days={r.get('ny_days','')})\n\n")
                for m in ["mae", "rmse", "mape", "r2", "skill_score", "peak_mae", "generalization_gap"]:
                    v = r.get(m, "")
                    lines.append(f"- {m}: {_fmt(v, 4)}\n")
                for field in ("epoch_stopped", "timestamp", "checkpoint_s3_path"):
                    v = r.get(field, "")
                    if v and str(v) != "nan":
                        lines.append(f"- {field}: {v}\n")
                lines.append("\n")

    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text("".join(lines), encoding="utf-8")
    print(f"  Regenerated {RESULTS_MD.relative_to(ROOT)}", flush=True)


def save_per_home(records: list[dict]) -> None:
    """Append or overwrite per-home metrics in per_home_metrics.csv."""
    if not records:
        return
    key_set = {
        (str(r.get("model_version", "")), str(r.get("experiment", "")),
         str(r.get("ny_days", "")), str(r.get("dataid", "")))
        for r in records
    }
    existing = [
        r for r in _read_csv(PER_HOME_CSV)
        if (str(r.get("model_version", "")), str(r.get("experiment", "")),
            str(r.get("ny_days", "")), str(r.get("dataid", ""))) not in key_set
    ]
    new_rows = [{c: r.get(c, "") for c in PER_HOME_COLUMNS} for r in records]
    _write_csv(PER_HOME_CSV, PER_HOME_COLUMNS, existing + new_rows)
    print(f"  Saved per-home metrics ({len(new_rows)} homes) -> {PER_HOME_CSV.relative_to(ROOT)}", flush=True)


def auto_commit(mae: float, r2: float, skill: float, model_version: str, experiment: str) -> None:
    """Stage results/, commit, pull --rebase, push.  Uses GITHUB_TOKEN if available."""
    try:
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        repo_root = str(ROOT)

    # Configure git identity (idempotent, local scope only)
    subprocess.run(["git", "-C", repo_root, "config", "user.email", "necunningham122@gmail.com"], check=False)
    subprocess.run(["git", "-C", repo_root, "config", "user.name",  "ncunning122"], check=False)

    github_token = os.environ.get("GITHUB_TOKEN", "")
    if github_token:
        remote_url = f"https://{github_token}@github.com/Chaim3ra/CS7180_final"
        subprocess.run(["git", "-C", repo_root, "remote", "set-url", "origin", remote_url], check=False)

    subprocess.run(["git", "-C", repo_root, "add", "results/"], check=False)

    skill_str = f"{skill:.1f}" if not math.isnan(skill) else "nan"
    r2_str    = f"{r2:.3f}"    if not math.isnan(r2)    else "nan"
    msg = f"Results: {model_version} {experiment} MAE={mae:.4f} R2={r2_str} Skill={skill_str}%"
    result = subprocess.run(
        ["git", "-C", repo_root, "commit", "-m", msg],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        stdout_err = (result.stdout + result.stderr).lower()
        if "nothing to commit" in stdout_err:
            print("  No new results to commit.", flush=True)
        else:
            print(f"  Git commit warning: {result.stderr.strip()}", flush=True)
        return

    print(f"  Committed: {msg}", flush=True)

    if not github_token:
        print("  Results saved locally — push manually or add GITHUB_TOKEN to Colab Secrets", flush=True)
        return

    subprocess.run(["git", "-C", repo_root, "pull", "--rebase", "origin", "main"], check=False)
    push = subprocess.run(["git", "-C", repo_root, "push", "origin", "main"],
                          capture_output=True, text=True)
    if push.returncode == 0:
        print("  Pushed to origin/main", flush=True)
    else:
        print(f"  Push failed: {push.stderr.strip()}", flush=True)
