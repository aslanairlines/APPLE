"""Summarize paired APPLE sweep metrics and plot proxy ID/attribute trade-off."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", type=Path, default=Path(__file__).resolve().parent / "scores.csv")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "figures")
    return p.parse_args()


def nondominated(rows: pd.DataFrame) -> list[bool]:
    """Maximize ID, minimize landmark error. Flag equal points as non-dominated."""
    x = rows["id_similarity"].to_numpy(dtype=float)
    y = rows["landmark_error"].to_numpy(dtype=float)
    return [not bool(np.any((x >= x[i]) & (y <= y[i]) & ((x > x[i]) | (y < y[i]))))
            for i in range(len(rows))]


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    required = {"stage", "id_guidance_scale", "id_weight", "pair_id", "success", "id_similarity", "landmark_error"}
    if missing := required.difference(data.columns):
        raise SystemExit(f"Missing CSV columns: {sorted(missing)}")
    data["success"] = data["success"].astype(str).str.lower().eq("true")
    group_cols = ["stage", "id_guidance_scale", "id_weight"]
    if data.duplicated(group_cols + ["pair_id"]).any():
        raise SystemExit("Duplicate pair observations in a setting; do not mix runs/seeds")
    if data[group_cols + ["pair_id"]].isna().any().any():
        raise SystemExit("Missing sweep setting or pair ID")
    for column in ["id_similarity", "landmark_error"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    valid = data[data["success"] & np.isfinite(data["id_similarity"]) & np.isfinite(data["landmark_error"])]
    if valid.empty:
        raise SystemExit("No valid results: check evaluate.py error column")
    pair_sets = list(data.groupby(group_cols)["pair_id"].agg(set))
    if any(pair_ids != pair_sets[0] for pair_ids in pair_sets[1:]):
        raise SystemExit("Settings contain different generated pair sets; evaluate the same pairs")
    valid_sets = {key: set(part["pair_id"]) for key, part in valid.groupby(group_cols)}
    common = set.intersection(*(valid_sets.get(key, set()) for key in data.groupby(group_cols).groups))
    if not common:
        raise SystemExit("No common successful pairs across all settings; inspect detection failures")
    paired = valid[valid["pair_id"].isin(common)]
    print(f"Comparing {len(common)}/{len(pair_sets[0])} common successful pairs across all settings")
    count_all = data.groupby(group_cols).size().rename("n_generated")
    count_detected = valid.groupby(group_cols).size().rename("n_detected")
    summary = (paired.groupby(group_cols)
               .agg(id_similarity=("id_similarity", "mean"),
                    landmark_error=("landmark_error", "mean"),
                    n_valid=("pair_id", "nunique"),
                    id_std=("id_similarity", "std"),
                    landmark_std=("landmark_error", "std"))
               .join(count_all).join(count_detected).reset_index())
    summary["detection_success_rate"] = summary["n_detected"] / summary["n_generated"]
    summary["pareto"] = nondominated(summary)
    return summary


def main():
    args = parse_args()
    data = pd.read_csv(args.scores, dtype={"pair_id": str})
    summary = summarize(data)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    for stage, part in summary.groupby("stage"):
        part = part.sort_values("id_guidance_scale" if stage == "cfg" else "id_weight")
        ax.plot(part["id_similarity"], part["landmark_error"], marker="o", label=stage)
        for _, p in part.iterrows():
            label = f"{p['id_guidance_scale']:g}" if stage == "cfg" else f"{p['id_weight']:g}"
            ax.annotate(label, (p["id_similarity"], p["landmark_error"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    pareto = summary[summary["pareto"]]
    ax.scatter(pareto["id_similarity"], pareto["landmark_error"],
               s=120, facecolors="none", edgecolors="black", linewidths=1.3, label="non-dominated")
    ax.set_xlabel("Identity similarity (buffalo_l cosine; higher is better)")
    ax.set_ylabel("2D landmark error to target / eye distance (lower is better)")
    ax.set_title("APPLE: identity vs target geometry (proxy, not full attribute preservation)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "tradeoff.png", dpi=180)
    fig.savefig(args.out_dir / "tradeoff.svg")
    plt.close(fig)
    print(summary.to_string(index=False))
    print(f"Saved {summary_path} and figures to {args.out_dir}")


if __name__ == "__main__":
    main()
