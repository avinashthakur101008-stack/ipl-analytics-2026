"""
data_loader.py
==============
Ingest, validate, and clean raw IPL ball-by-ball and match-level CSV data.
Supports Cricsheet-format CSVs as well as a built-in synthetic data generator
for testing and demos when real data is not yet available.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IPL_2026_TEAMS = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Punjab Kings",
    "Lucknow Super Giants",
    "Gujarat Titans",
]

VENUES = [
    "Wankhede Stadium, Mumbai",
    "M. A. Chidambaram Stadium, Chennai",
    "Eden Gardens, Kolkata",
    "Narendra Modi Stadium, Ahmedabad",
    "M. Chinnaswamy Stadium, Bengaluru",
    "Arun Jaitley Stadium, Delhi",
    "Rajiv Gandhi International Stadium, Hyderabad",
    "Sawai Mansingh Stadium, Jaipur",
    "Ekana Cricket Stadium, Lucknow",
    "Punjab Cricket Association Stadium, Mohali",
]

BOWLER_TYPES = ["pace", "medium", "spin", "medium-pace"]

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# Synthetic data generator (for demos / CI)
# ---------------------------------------------------------------------------

def generate_synthetic_deliveries(
    n_matches: int = 74,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic-looking ball-by-ball IPL data for demo purposes."""
    rng = np.random.default_rng(seed)
    random.seed(seed)

    rows: list[dict] = []
    match_id = 1

    for _ in range(n_matches):
        team1, team2 = random.sample(IPL_2026_TEAMS, 2)
        venue = random.choice(VENUES)
        date = pd.Timestamp("2026-03-22") + pd.Timedelta(days=random.randint(0, 55))

        for innings in [1, 2]:
            batting_team = team1 if innings == 1 else team2
            bowling_team = team2 if innings == 1 else team1

            # Build batting lineup (11 players)
            batters = [f"{batting_team[:3].upper()}_B{i}" for i in range(1, 12)]
            bowlers = [f"{bowling_team[:3].upper()}_BW{i}" for i in range(1, 6)]

            wickets = 0
            batter_idx = 0
            striker = batters[batter_idx]
            non_striker = batters[batter_idx + 1]
            batter_idx = 2

            for over in range(20):
                if wickets >= 10:
                    break
                bowler = bowlers[over % len(bowlers)]
                phase = (
                    "powerplay" if over < 6
                    else "middle" if over < 15
                    else "death"
                )

                for ball in range(1, 7):
                    # Outcome probabilities vary by phase
                    if phase == "powerplay":
                        weights = [35, 20, 15, 5, 10, 5, 10]  # 0,1,2,3,4,6,W
                    elif phase == "middle":
                        weights = [38, 28, 12, 3, 8, 4, 7]
                    else:  # death
                        weights = [25, 18, 12, 4, 16, 12, 13]

                    outcome = rng.choice(
                        ["0", "1", "2", "3", "4", "6", "W"],
                        p=[w / sum(weights) for w in weights],
                    )

                    is_wicket = outcome == "W"
                    runs_off_bat = 0 if is_wicket else int(outcome)
                    extras = 0
                    if not is_wicket and rng.random() < 0.05:
                        extras = 1  # wide or no-ball
                    total_runs = runs_off_bat + extras

                    rows.append(
                        {
                            "match_id": match_id,
                            "date": date,
                            "venue": venue,
                            "innings": innings,
                            "batting_team": batting_team,
                            "bowling_team": bowling_team,
                            "over": over,
                            "ball": ball,
                            "batter": striker,
                            "non_striker": non_striker,
                            "bowler": bowler,
                            "runs_off_bat": runs_off_bat,
                            "extras": extras,
                            "total_runs": total_runs,
                            "is_wicket": int(is_wicket),
                            "phase": phase,
                            "bowler_type": random.choice(BOWLER_TYPES),
                        }
                    )

                    if is_wicket:
                        wickets += 1
                        if batter_idx < len(batters):
                            striker = batters[batter_idx]
                            batter_idx += 1
                        if wickets >= 10:
                            break
                    elif runs_off_bat % 2 == 1:
                        striker, non_striker = non_striker, striker

                # Rotate strike at end of over
                striker, non_striker = non_striker, striker

        match_id += 1

    df = pd.DataFrame(rows)
    logger.info("Generated %d synthetic deliveries across %d matches.", len(df), n_matches)
    return df


def generate_synthetic_matches(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Derive match-level summary table from ball-by-ball data."""
    grp = deliveries.groupby(["match_id", "innings", "batting_team", "bowling_team"])
    innings_totals = grp.agg(
        total_runs=("total_runs", "sum"),
        wickets=("is_wicket", "sum"),
        balls_faced=("ball", "count"),
    ).reset_index()

    inn1 = innings_totals[innings_totals["innings"] == 1].rename(
        columns={
            "batting_team": "team1",
            "bowling_team": "team2",
            "total_runs": "team1_runs",
            "wickets": "team1_wickets",
        }
    )
    inn2 = innings_totals[innings_totals["innings"] == 2].rename(
        columns={
            "batting_team": "team2_chasing",
            "total_runs": "team2_runs",
            "wickets": "team2_wickets",
        }
    )

    matches = inn1.merge(inn2[["match_id", "team2_runs", "team2_wickets"]], on="match_id")
    matches["winner"] = np.where(
        matches["team2_runs"] > matches["team1_runs"],
        matches["team2"],
        matches["team1"],
    )
    date_map = deliveries.drop_duplicates("match_id")[["match_id", "date", "venue"]]
    matches = matches.merge(date_map, on="match_id")

    logger.info("Derived %d match summaries.", len(matches))
    return matches


# ---------------------------------------------------------------------------
# Real data loaders (Cricsheet CSV format)
# ---------------------------------------------------------------------------

def load_deliveries(path: Optional[Path] = None) -> pd.DataFrame:
    """Load deliveries CSV; falls back to synthetic data if file not found."""
    filepath = path or RAW_DATA_DIR / "deliveries.csv"
    if not filepath.exists():
        logger.warning(
            "deliveries.csv not found at %s — using synthetic data.", filepath
        )
        return generate_synthetic_deliveries()

    df = pd.read_csv(filepath)
    df = _clean_deliveries(df)
    logger.info("Loaded %d deliveries from %s.", len(df), filepath)
    return df


def load_matches(path: Optional[Path] = None) -> pd.DataFrame:
    """Load matches CSV; falls back to synthetic data if file not found."""
    filepath = path or RAW_DATA_DIR / "matches.csv"
    if not filepath.exists():
        logger.warning(
            "matches.csv not found at %s — using synthetic data.", filepath
        )
        deliveries = generate_synthetic_deliveries()
        return generate_synthetic_matches(deliveries)

    df = pd.read_csv(filepath)
    df = _clean_matches(df)
    logger.info("Loaded %d matches from %s.", len(df), filepath)
    return df


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def _clean_deliveries(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names and types for Cricsheet delivery CSVs."""
    col_map = {
        "match_id": "match_id",
        "inning": "innings",
        "batting_team": "batting_team",
        "bowling_team": "bowling_team",
        "over": "over",
        "ball": "ball",
        "batter": "batter",
        "non_striker": "non_striker",
        "bowler": "bowler",
        "runs_off_bat": "runs_off_bat",
        "extras": "extras",
        "total_runs": "total_runs",
        "wicket_type": "wicket_type",
        "player_dismissed": "player_dismissed",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["is_wicket"] = df.get("player_dismissed", pd.Series(dtype=str)).notna().astype(int)
    df["phase"] = pd.cut(
        df["over"],
        bins=[-1, 5, 14, 20],
        labels=["powerplay", "middle", "death"],
    )
    df["over"] = df["over"].astype(int)
    return df


def _clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names for Cricsheet match CSVs."""
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def save_processed(df: pd.DataFrame, name: str) -> None:
    """Persist a processed dataframe to data/processed/."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DATA_DIR / f"{name}.csv"
    df.to_csv(out, index=False)
    logger.info("Saved processed data → %s", out)
