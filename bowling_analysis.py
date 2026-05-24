"""
bowling_analysis.py
===================
Economy analysis, Dot Ball Pressure Index, death-over specialist ratings,
and phase-wise wicket heatmaps for IPL 2026 bowlers.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Core bowling metrics
# ---------------------------------------------------------------------------

def bowling_summary(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Master bowling summary table with: overs, runs, wickets, economy,
    average, strike rate, dot-ball %.
    """
    df = deliveries.copy()
    df["is_dot"] = ((df["runs_off_bat"] == 0) & (df["extras"] == 0)).astype(int)

    stats = (
        df.groupby("bowler")
        .agg(
            balls=("ball", "count"),
            runs=("total_runs", "sum"),
            wickets=("is_wicket", "sum"),
            dots=("is_dot", "sum"),
            matches=("match_id", "nunique"),
        )
        .reset_index()
    )
    stats["overs"] = (stats["balls"] / 6).round(2)
    stats["economy"] = (stats["runs"] / stats["overs"]).round(2)
    stats["average"] = (stats["runs"] / stats["wickets"].clip(lower=1)).round(2)
    stats["strike_rate"] = (stats["balls"] / stats["wickets"].clip(lower=1)).round(2)
    stats["dot_pct"] = (stats["dots"] / stats["balls"] * 100).round(2)
    return stats.sort_values("wickets", ascending=False).reset_index(drop=True)


def dot_ball_pressure_index(deliveries: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Dot Ball Pressure Index (DBPI) — measures a bowler's ability to string
    together consecutive dot balls within the same over/spell.

    DBPI = mean length of dot-ball runs (consecutive sequences ≥ window)
           normalised by total dots delivered.
    """
    df = deliveries.sort_values(["match_id", "innings", "bowler", "over", "ball"]).copy()
    df["is_dot"] = ((df["runs_off_bat"] == 0) & (df["extras"] == 0)).astype(int)

    results: list[dict] = []
    for bowler, grp in df.groupby("bowler"):
        dots = grp["is_dot"].values
        total_dots = dots.sum()
        if total_dots == 0:
            results.append({"bowler": bowler, "dbpi": 0.0, "total_dots": 0})
            continue

        # Find consecutive dot sequences
        run_lengths = []
        current_run = 0
        for d in dots:
            if d == 1:
                current_run += 1
            else:
                if current_run >= window:
                    run_lengths.append(current_run)
                current_run = 0
        if current_run >= window:
            run_lengths.append(current_run)

        avg_run = np.mean(run_lengths) if run_lengths else 0
        dbpi = round(avg_run / total_dots * 100, 3) if total_dots else 0
        results.append({"bowler": bowler, "dbpi": dbpi, "total_dots": int(total_dots)})

    out = pd.DataFrame(results)
    out = out.merge(bowling_summary(df)[["bowler", "overs", "economy", "wickets"]], on="bowler", how="left")
    return out.sort_values("dbpi", ascending=False).reset_index(drop=True)


def phase_economy(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Economy rates split by powerplay / middle / death for every bowler.
    Returns wide-format DataFrame.
    """
    grp = (
        deliveries.groupby(["bowler", "phase"])
        .agg(runs=("total_runs", "sum"), balls=("ball", "count"))
        .reset_index()
    )
    grp["overs"] = grp["balls"] / 6
    grp["economy"] = (grp["runs"] / grp["overs"]).round(2)

    wide = grp.pivot_table(
        index="bowler", columns="phase", values="economy", aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    for col in ["powerplay", "middle", "death"]:
        if col not in wide.columns:
            wide[col] = np.nan
    wide = wide.rename(
        columns={
            "powerplay": "powerplay_econ",
            "middle": "middle_econ",
            "death": "death_econ",
        }
    )
    return wide


def death_specialist_rating(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Death Specialist Rating (DSR) for overs 17–20:
        DSR = (1/economy) × wickets_per_over × 100 × (1 + dot_pct/100)
    Higher is better — rewards tight, wicket-taking death bowling.
    """
    death = deliveries[deliveries["over"] >= 16].copy()
    death["is_dot"] = ((death["runs_off_bat"] == 0) & (death["extras"] == 0)).astype(int)

    stats = (
        death.groupby("bowler")
        .agg(
            balls=("ball", "count"),
            runs=("total_runs", "sum"),
            wickets=("is_wicket", "sum"),
            dots=("is_dot", "sum"),
            matches=("match_id", "nunique"),
        )
        .reset_index()
    )
    stats = stats[stats["balls"] >= 18]  # min 3 overs in death
    stats["overs"] = stats["balls"] / 6
    stats["economy"] = stats["runs"] / stats["overs"]
    stats["wpo"] = stats["wickets"] / stats["overs"]
    stats["dot_pct"] = stats["dots"] / stats["balls"]
    stats["dsr"] = (
        (1 / stats["economy"].clip(lower=0.1))
        * stats["wpo"]
        * 100
        * (1 + stats["dot_pct"])
    ).round(3)
    stats["economy"] = stats["economy"].round(2)
    stats["wpo"] = stats["wpo"].round(2)
    stats["dot_pct"] = (stats["dot_pct"] * 100).round(2)
    return stats.sort_values("dsr", ascending=False).reset_index(drop=True)


def economy_bands(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each bowler into an economy band across the full season.
        Tight      : economy < 7.5
        Moderate   : 7.5 ≤ economy < 9.0
        Expensive  : economy ≥ 9.0
    """
    stats = bowling_summary(deliveries)[["bowler", "overs", "economy", "wickets", "matches"]]
    stats = stats[stats["overs"] >= 10]  # minimum overs threshold

    def band(econ: float) -> str:
        if econ < 7.5:
            return "Tight"
        if econ < 9.0:
            return "Moderate"
        return "Expensive"

    stats["band"] = stats["economy"].apply(band)
    return stats.sort_values("economy").reset_index(drop=True)


def top_wicket_takers(deliveries: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return top-N wicket takers with key bowling stats."""
    return bowling_summary(deliveries).head(n)
