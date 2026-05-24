"""
team_strategy.py
================
Team-level strategy metrics: Powerplay Aggressiveness Score, toss-venue
interaction, net run-rate analysis, and target-setting vs. chasing profiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Powerplay analysis
# ---------------------------------------------------------------------------

def powerplay_aggressiveness_score(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Powerplay Aggressiveness Score (PAS) — balances run-rate vs wicket risk.

    PAS = (powerplay_run_rate / league_avg_pp_rr) × (1 - wicket_rate_penalty)
    where wicket_rate_penalty = pp_wickets / 10   (normalised to 10 wickets max)

    Higher PAS → team bats aggressively without losing too many wickets.
    """
    pp = deliveries[deliveries["phase"] == "powerplay"].copy()

    team_pp = (
        pp.groupby(["match_id", "batting_team"])
        .agg(runs=("total_runs", "sum"), balls=("ball", "count"), wickets=("is_wicket", "sum"))
        .reset_index()
    )
    team_pp["pp_rr"] = team_pp["runs"] / (team_pp["balls"] / 6)

    season_avg_pp_rr = team_pp["pp_rr"].mean()

    summary = (
        team_pp.groupby("batting_team")
        .agg(avg_pp_rr=("pp_rr", "mean"), avg_pp_wkts=("wickets", "mean"))
        .reset_index()
    )
    summary["wicket_penalty"] = (summary["avg_pp_wkts"] / 10).clip(upper=0.5)
    summary["pas"] = (
        (summary["avg_pp_rr"] / season_avg_pp_rr) * (1 - summary["wicket_penalty"])
    ).round(3)
    summary["avg_pp_rr"] = summary["avg_pp_rr"].round(2)
    summary["avg_pp_wkts"] = summary["avg_pp_wkts"].round(2)
    return summary.sort_values("pas", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Middle-over consolidation
# ---------------------------------------------------------------------------

def middle_over_consolidation(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Net run-rate delta in overs 7–15 versus the league average.
    Positive values → team scores above average in the middle phase.
    """
    mid = deliveries[deliveries["phase"] == "middle"].copy()
    team_mid = (
        mid.groupby(["match_id", "batting_team"])
        .agg(runs=("total_runs", "sum"), balls=("ball", "count"))
        .reset_index()
    )
    team_mid["mid_rr"] = team_mid["runs"] / (team_mid["balls"] / 6)
    league_avg = team_mid["mid_rr"].mean()

    summary = (
        team_mid.groupby("batting_team")
        .agg(avg_mid_rr=("mid_rr", "mean"), innings=("match_id", "count"))
        .reset_index()
    )
    summary["mid_delta"] = (summary["avg_mid_rr"] - league_avg).round(2)
    summary["avg_mid_rr"] = summary["avg_mid_rr"].round(2)
    return summary.sort_values("mid_delta", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Toss & venue
# ---------------------------------------------------------------------------

def toss_venue_winrate(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Win-rate by toss decision × venue interaction.
    Expects columns: team1, team2, winner, venue, toss_winner, toss_decision.
    If toss columns are absent (synthetic data), returns an empty DataFrame.
    """
    required = {"toss_winner", "toss_decision", "winner", "venue"}
    if not required.issubset(matches.columns):
        return pd.DataFrame(
            columns=["venue", "toss_decision", "toss_winner_wins", "total", "win_pct"]
        )

    df = matches.copy()
    df["toss_winner_won"] = (df["toss_winner"] == df["winner"]).astype(int)
    grp = (
        df.groupby(["venue", "toss_decision"])
        .agg(toss_winner_wins=("toss_winner_won", "sum"), total=("match_id", "count"))
        .reset_index()
    )
    grp["win_pct"] = (grp["toss_winner_wins"] / grp["total"] * 100).round(1)
    return grp.sort_values("win_pct", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Net run rate
# ---------------------------------------------------------------------------

def season_nrr(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Season Net Run Rate (NRR) for each team, computed from match-level data.
    NRR = (runs_scored / overs_faced) − (runs_conceded / overs_bowled)
    """
    required = {"team1", "team2", "team1_runs", "team2_runs", "winner"}
    if not required.issubset(matches.columns):
        return pd.DataFrame()

    rows: list[dict] = []
    for _, row in matches.iterrows():
        overs = 20  # simplify to full 20 overs
        for team, runs_for, runs_against in [
            (row["team1"], row["team1_runs"], row["team2_runs"]),
            (row["team2"], row["team2_runs"], row["team1_runs"]),
        ]:
            won = int(row["winner"] == team)
            rows.append(
                {
                    "team": team,
                    "won": won,
                    "runs_for": runs_for,
                    "runs_against": runs_against,
                    "overs_faced": overs,
                    "overs_bowled": overs,
                }
            )

    df = pd.DataFrame(rows)
    agg = (
        df.groupby("team")
        .agg(
            matches=("won", "count"),
            wins=("won", "sum"),
            runs_for=("runs_for", "sum"),
            runs_against=("runs_against", "sum"),
            overs_faced=("overs_faced", "sum"),
            overs_bowled=("overs_bowled", "sum"),
        )
        .reset_index()
    )
    agg["losses"] = agg["matches"] - agg["wins"]
    agg["nrr"] = (
        agg["runs_for"] / agg["overs_faced"] - agg["runs_against"] / agg["overs_bowled"]
    ).round(3)
    agg["points"] = agg["wins"] * 2
    return agg.sort_values(["points", "nrr"], ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Chasing vs setting profile
# ---------------------------------------------------------------------------

def chase_vs_set_profile(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Win % when batting first (setting) vs batting second (chasing)
    for each team.
    """
    required = {"team1", "team2", "winner"}
    if not required.issubset(matches.columns):
        return pd.DataFrame()

    records: list[dict] = []
    for _, row in matches.iterrows():
        # team1 always bats first in this schema
        for role, team in [("setting", row["team1"]), ("chasing", row["team2"])]:
            records.append({"team": team, "role": role, "won": int(row["winner"] == team)})

    df = pd.DataFrame(records)
    profile = (
        df.groupby(["team", "role"])
        .agg(matches=("won", "count"), wins=("won", "sum"))
        .reset_index()
    )
    profile["win_pct"] = (profile["wins"] / profile["matches"] * 100).round(1)
    wide = profile.pivot_table(
        index="team", columns="role", values="win_pct", aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(
        columns={"chasing": "chase_win_pct", "setting": "set_win_pct"}
    )
    wide["preferred_role"] = np.where(
        wide.get("chase_win_pct", 50) > wide.get("set_win_pct", 50), "Chase", "Set"
    )
    return wide.sort_values("chase_win_pct", ascending=False).reset_index(drop=True)
