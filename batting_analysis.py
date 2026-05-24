"""
batting_analysis.py
===================
Phase-wise batting metrics, consistency index, boundary analysis, and
impact scoring for IPL 2026.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Core batting metrics
# ---------------------------------------------------------------------------

def phase_strike_rates(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Compute batter strike rates broken down by match phase.

    Returns a wide DataFrame with columns:
        batter, powerplay_sr, middle_sr, death_sr, overall_sr
    """
    grp = (
        deliveries.groupby(["batter", "phase"])
        .agg(runs=("runs_off_bat", "sum"), balls=("ball", "count"))
        .reset_index()
    )
    grp["sr"] = (grp["runs"] / grp["balls"] * 100).round(2)

    wide = grp.pivot_table(
        index="batter", columns="phase", values="sr", aggfunc="first"
    ).reset_index()
    wide.columns.name = None

    # Ensure all phases present
    for col in ["powerplay", "middle", "death"]:
        if col not in wide.columns:
            wide[col] = np.nan

    # Overall SR
    overall = (
        deliveries.groupby("batter")
        .agg(runs=("runs_off_bat", "sum"), balls=("ball", "count"))
        .assign(overall_sr=lambda d: (d["runs"] / d["balls"] * 100).round(2))
        .reset_index()[["batter", "runs", "balls", "overall_sr"]]
    )
    wide = wide.merge(overall, on="batter")
    wide = wide.rename(
        columns={
            "powerplay": "powerplay_sr",
            "middle": "middle_sr",
            "death": "death_sr",
        }
    )
    return wide.sort_values("overall_sr", ascending=False).reset_index(drop=True)


def boundary_percentage(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Returns boundary percentage (4s + 6s) per batter.
    boundary_pct = (fours + sixes) / balls_faced * 100
    """
    df = deliveries.copy()
    df["is_four"] = (df["runs_off_bat"] == 4).astype(int)
    df["is_six"] = (df["runs_off_bat"] == 6).astype(int)

    stats = (
        df.groupby("batter")
        .agg(
            balls=("ball", "count"),
            fours=("is_four", "sum"),
            sixes=("is_six", "sum"),
            runs=("runs_off_bat", "sum"),
        )
        .reset_index()
    )
    stats["boundary_pct"] = ((stats["fours"] + stats["sixes"]) / stats["balls"] * 100).round(2)
    return stats.sort_values("boundary_pct", ascending=False).reset_index(drop=True)


def consistency_index(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Consistency Index = (runs per dismissal) × log(1 + innings_played).

    Higher values indicate batters who score big AND do so repeatedly.
    """
    # Per-innings totals
    innings_df = (
        deliveries.groupby(["match_id", "innings", "batter"])
        .agg(runs=("runs_off_bat", "sum"), dismissed=("is_wicket", "max"))
        .reset_index()
    )
    batter_stats = (
        innings_df.groupby("batter")
        .agg(
            innings=("match_id", "count"),
            total_runs=("runs", "sum"),
            dismissals=("dismissed", "sum"),
        )
        .reset_index()
    )
    batter_stats["average"] = (
        batter_stats["total_runs"] / batter_stats["dismissals"].clip(lower=1)
    ).round(2)
    batter_stats["consistency_index"] = (
        batter_stats["average"] * np.log1p(batter_stats["innings"])
    ).round(2)
    return batter_stats.sort_values("consistency_index", ascending=False).reset_index(drop=True)


def impact_score(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Impact Score = 0.4 × normalised_SR + 0.35 × normalised_avg + 0.25 × normalised_boundary_pct.
    All components normalised 0–100.
    """
    sr_df = phase_strike_rates(deliveries)[["batter", "overall_sr", "runs", "balls"]]
    ci_df = consistency_index(deliveries)[["batter", "average", "innings"]]
    bp_df = boundary_percentage(deliveries)[["batter", "boundary_pct"]]

    df = sr_df.merge(ci_df, on="batter").merge(bp_df, on="batter")
    df = df[df["balls"] >= 30]  # min balls threshold

    def normalise(series: pd.Series) -> pd.Series:
        rng = series.max() - series.min()
        return ((series - series.min()) / rng * 100).round(2) if rng > 0 else series * 0

    df["norm_sr"] = normalise(df["overall_sr"])
    df["norm_avg"] = normalise(df["average"])
    df["norm_bp"] = normalise(df["boundary_pct"])

    df["impact_score"] = (
        0.40 * df["norm_sr"] + 0.35 * df["norm_avg"] + 0.25 * df["norm_bp"]
    ).round(2)

    return (
        df[["batter", "innings", "runs", "balls", "overall_sr", "average", "boundary_pct", "impact_score"]]
        .sort_values("impact_score", ascending=False)
        .reset_index(drop=True)
    )


def top_run_scorers(deliveries: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the top-N run scorers with key batting stats."""
    stats = (
        deliveries.groupby("batter")
        .agg(
            runs=("runs_off_bat", "sum"),
            balls=("ball", "count"),
            fours=("runs_off_bat", lambda x: (x == 4).sum()),
            sixes=("runs_off_bat", lambda x: (x == 6).sum()),
            dismissals=("is_wicket", "sum"),
            matches=("match_id", "nunique"),
        )
        .reset_index()
    )
    stats["sr"] = (stats["runs"] / stats["balls"] * 100).round(2)
    stats["average"] = (stats["runs"] / stats["dismissals"].clip(lower=1)).round(2)
    return stats.sort_values("runs", ascending=False).head(n).reset_index(drop=True)


def partnership_analysis(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Compute batting partnerships (runs while each pair was at the crease).
    Returns top partnerships by runs.
    """
    df = deliveries.copy()
    df["pair"] = df.apply(
        lambda r: "_".join(sorted([r["batter"], r["non_striker"]])), axis=1
    )
    partnerships = (
        df.groupby(["match_id", "innings", "pair"])
        .agg(runs=("runs_off_bat", "sum"), balls=("ball", "count"))
        .reset_index()
    )
    summary = (
        partnerships.groupby("pair")
        .agg(
            total_runs=("runs", "sum"),
            avg_partnership=("runs", "mean"),
            times_together=("runs", "count"),
        )
        .reset_index()
    )
    summary["avg_partnership"] = summary["avg_partnership"].round(1)
    return summary.sort_values("total_runs", ascending=False).reset_index(drop=True)
