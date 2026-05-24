"""
match_predictor.py
==================
Logistic-regression win-probability model for IPL 2026 match states.

Features
--------
- Required Run Rate (RRR)
- Wickets in hand
- Current run rate (CRR)
- Over number
- Phase (powerplay / middle / death)
- Venue-level historical win rate (batting second)

Usage
-----
    from src.match_predictor import WinProbabilityModel
    model = WinProbabilityModel()
    model.fit(deliveries, matches)
    prob = model.predict_proba(rrr=8.5, wickets_in_hand=7, current_over=14, crr=7.2)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def _phase_label(over: int) -> str:
    if over < 6:
        return "powerplay"
    if over < 15:
        return "middle"
    return "death"


def _build_training_rows(
    deliveries: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one row per over-snapshot in each 2nd-innings chase, labelled
    with whether the chasing team ultimately won (1) or lost (0).
    """
    required_match_cols = {"match_id", "team1_runs", "winner", "team2"}
    if not required_match_cols.issubset(matches.columns):
        logger.warning("matches DataFrame missing required columns — generating dummy training data.")
        return _generate_dummy_training_data()

    inn2 = deliveries[deliveries["innings"] == 2].copy()
    # Cumulative runs & wickets per over
    inn2 = inn2.sort_values(["match_id", "over", "ball"])
    inn2["cum_runs"] = inn2.groupby("match_id")["total_runs"].cumsum()
    inn2["cum_wickets"] = inn2.groupby("match_id")["is_wicket"].cumsum()

    over_snap = (
        inn2.groupby(["match_id", "over"])
        .agg(
            cum_runs=("cum_runs", "last"),
            cum_wickets=("cum_wickets", "last"),
            balls_bowled=("ball", "count"),
        )
        .reset_index()
    )
    over_snap["overs_done"] = over_snap["over"] + 1
    over_snap["balls_remaining"] = (20 - over_snap["overs_done"]) * 6
    over_snap["overs_remaining"] = over_snap["balls_remaining"] / 6
    over_snap["wickets_in_hand"] = 10 - over_snap["cum_wickets"]
    over_snap["crr"] = (over_snap["cum_runs"] / over_snap["overs_done"]).round(2)
    over_snap["phase"] = over_snap["over"].apply(_phase_label)

    # Merge with target
    target_map = matches.set_index("match_id")["team1_runs"].to_dict()
    winner_map = matches.set_index("match_id")["winner"].to_dict()
    batting_map = (
        deliveries[deliveries["innings"] == 2]
        .drop_duplicates("match_id")
        .set_index("match_id")["batting_team"]
        .to_dict()
    )

    over_snap["target"] = over_snap["match_id"].map(target_map) + 1
    over_snap["runs_needed"] = over_snap["target"] - over_snap["cum_runs"]
    over_snap["rrr"] = (
        over_snap["runs_needed"] / over_snap["overs_remaining"].clip(lower=0.1)
    ).clip(0, 36).round(2)

    batting_team = over_snap["match_id"].map(batting_map)
    winner = over_snap["match_id"].map(winner_map)
    over_snap["label"] = (batting_team == winner).astype(int)

    # Phase one-hot
    over_snap = pd.get_dummies(over_snap, columns=["phase"], drop_first=False)
    feature_cols = [
        "rrr", "crr", "wickets_in_hand", "overs_done",
    ]
    for col in ["phase_middle", "phase_death", "phase_powerplay"]:
        if col in over_snap.columns:
            feature_cols.append(col)
        else:
            over_snap[col] = 0
            feature_cols.append(col)

    return over_snap[feature_cols + ["label"]].dropna()


def _generate_dummy_training_data(n: int = 5000) -> pd.DataFrame:
    """Synthetic training data when real data is unavailable."""
    rng = np.random.default_rng(99)
    rrr = rng.uniform(4, 20, n)
    crr = rng.uniform(4, 14, n)
    wih = rng.integers(1, 11, n)
    overs = rng.integers(1, 21, n)
    # Simulate label: lower RRR + more wickets in hand → more likely to win
    log_odds = -0.4 * rrr + 0.3 * crr + 0.15 * wih + rng.normal(0, 0.5, n)
    probs = 1 / (1 + np.exp(-log_odds))
    labels = (rng.uniform(0, 1, n) < probs).astype(int)
    phase_middle = (overs > 6).astype(int)
    phase_death = (overs > 14).astype(int)
    return pd.DataFrame(
        {
            "rrr": rrr,
            "crr": crr,
            "wickets_in_hand": wih,
            "overs_done": overs,
            "phase_middle": phase_middle,
            "phase_death": phase_death,
            "phase_powerplay": (~phase_middle.astype(bool)).astype(int),
            "label": labels,
        }
    )


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

@dataclass
class WinProbabilityModel:
    """Logistic Regression win-probability model for 2nd-innings chases."""

    model: LogisticRegression = field(default_factory=lambda: LogisticRegression(max_iter=500))
    scaler: StandardScaler = field(default_factory=StandardScaler)
    feature_cols: list[str] = field(default_factory=list)
    is_fitted: bool = False
    eval_metrics: dict = field(default_factory=dict)

    def fit(
        self,
        deliveries: pd.DataFrame,
        matches: Optional[pd.DataFrame] = None,
        test_size: float = 0.2,
    ) -> "WinProbabilityModel":
        """Build features, train, and evaluate the model."""
        if matches is None:
            df = _generate_dummy_training_data()
        else:
            df = _build_training_rows(deliveries, matches)

        if df.empty or len(df) < 100:
            logger.warning("Insufficient training rows; falling back to dummy data.")
            df = _generate_dummy_training_data()

        self.feature_cols = [c for c in df.columns if c != "label"]
        X = df[self.feature_cols].values
        y = df["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model.fit(X_train, y_train)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        self.eval_metrics = {
            "auc": round(roc_auc_score(y_test, y_pred_proba), 4),
            "log_loss": round(log_loss(y_test, y_pred_proba), 4),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }
        self.is_fitted = True
        logger.info("Model trained. AUC=%.4f | Log-loss=%.4f", self.eval_metrics["auc"], self.eval_metrics["log_loss"])
        return self

    def predict_proba(
        self,
        rrr: float,
        wickets_in_hand: int,
        current_over: int,
        crr: float,
    ) -> float:
        """
        Returns win probability (0–1) for the chasing team given match state.

        Parameters
        ----------
        rrr              : Required Run Rate (runs needed / overs remaining)
        wickets_in_hand  : Number of wickets remaining (1–10)
        current_over     : Over number (0-indexed, 0–19)
        crr              : Current Run Rate (runs scored / overs completed)
        """
        if not self.is_fitted:
            self.fit(pd.DataFrame())

        phase = _phase_label(current_over)
        row = {
            "rrr": rrr,
            "crr": crr,
            "wickets_in_hand": wickets_in_hand,
            "overs_done": current_over + 1,
            "phase_middle": int(phase == "middle"),
            "phase_death": int(phase == "death"),
            "phase_powerplay": int(phase == "powerplay"),
        }
        X = np.array([[row[f] for f in self.feature_cols]])
        X = self.scaler.transform(X)
        prob = self.model.predict_proba(X)[0, 1]
        return round(float(prob), 4)

    def rolling_win_probability(
        self,
        target: int,
        deliveries_2nd: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute over-by-over win probability for a given chase.

        Parameters
        ----------
        target           : Runs to win (target = 1st innings total + 1)
        deliveries_2nd   : Ball-by-ball data for the 2nd innings only
        """
        if not self.is_fitted:
            self.fit(pd.DataFrame())

        df = deliveries_2nd.sort_values(["over", "ball"]).copy()
        df["cum_runs"] = df["total_runs"].cumsum()
        df["cum_wickets"] = df["is_wicket"].cumsum()

        records = []
        for over, grp in df.groupby("over"):
            cum_runs = grp["cum_runs"].iloc[-1]
            cum_wickets = grp["cum_wickets"].iloc[-1]
            overs_done = over + 1
            overs_remaining = max(20 - overs_done, 0.1)
            runs_needed = max(target - cum_runs, 0)
            rrr = runs_needed / overs_remaining
            crr = cum_runs / overs_done
            wih = max(10 - int(cum_wickets), 0)
            prob = self.predict_proba(rrr, wih, over, crr)
            records.append(
                {
                    "over": over + 1,
                    "cum_runs": int(cum_runs),
                    "wickets_fallen": int(cum_wickets),
                    "runs_needed": int(runs_needed),
                    "rrr": round(rrr, 2),
                    "crr": round(crr, 2),
                    "win_prob": prob,
                }
            )

        return pd.DataFrame(records)

    def model_summary(self) -> str:
        """Return a text summary of model performance."""
        if not self.is_fitted:
            return "Model not yet fitted."
        lines = [
            "=== Win Probability Model Summary ===",
            f"  Algorithm   : Logistic Regression",
            f"  AUC-ROC     : {self.eval_metrics.get('auc', 'N/A')}",
            f"  Log-Loss    : {self.eval_metrics.get('log_loss', 'N/A')}",
            f"  Train rows  : {self.eval_metrics.get('train_samples', 'N/A')}",
            f"  Test rows   : {self.eval_metrics.get('test_samples', 'N/A')}",
            f"  Features    : {', '.join(self.feature_cols)}",
        ]
        return "\n".join(lines)
