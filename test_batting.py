"""
tests/test_batting.py
=====================
Unit tests for batting_analysis module.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import generate_synthetic_deliveries
from src.batting_analysis import (
    boundary_percentage,
    consistency_index,
    impact_score,
    partnership_analysis,
    phase_strike_rates,
    top_run_scorers,
)


@pytest.fixture(scope="module")
def deliveries():
    return generate_synthetic_deliveries(n_matches=20, seed=1)


# ---------------------------------------------------------------------------
# phase_strike_rates
# ---------------------------------------------------------------------------

class TestPhaseStrikeRates:
    def test_returns_dataframe(self, deliveries):
        result = phase_strike_rates(deliveries)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, deliveries):
        result = phase_strike_rates(deliveries)
        for col in ["batter", "powerplay_sr", "middle_sr", "death_sr", "overall_sr"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_negative_sr(self, deliveries):
        result = phase_strike_rates(deliveries)
        for col in ["powerplay_sr", "middle_sr", "death_sr", "overall_sr"]:
            assert (result[col].dropna() >= 0).all(), f"Negative SR in {col}"

    def test_sorted_by_overall_sr(self, deliveries):
        result = phase_strike_rates(deliveries)
        srs = result["overall_sr"].dropna().tolist()
        assert srs == sorted(srs, reverse=True)


# ---------------------------------------------------------------------------
# boundary_percentage
# ---------------------------------------------------------------------------

class TestBoundaryPercentage:
    def test_pct_in_range(self, deliveries):
        result = boundary_percentage(deliveries)
        assert result["boundary_pct"].between(0, 100).all()

    def test_sorted_descending(self, deliveries):
        result = boundary_percentage(deliveries)
        pcts = result["boundary_pct"].tolist()
        assert pcts == sorted(pcts, reverse=True)


# ---------------------------------------------------------------------------
# consistency_index
# ---------------------------------------------------------------------------

class TestConsistencyIndex:
    def test_non_negative(self, deliveries):
        result = consistency_index(deliveries)
        assert (result["consistency_index"] >= 0).all()

    def test_average_positive(self, deliveries):
        result = consistency_index(deliveries)
        assert (result["average"] > 0).all()


# ---------------------------------------------------------------------------
# impact_score
# ---------------------------------------------------------------------------

class TestImpactScore:
    def test_score_range(self, deliveries):
        result = impact_score(deliveries)
        if result.empty:
            pytest.skip("Not enough balls for impact score threshold.")
        assert result["impact_score"].between(0, 100).all()

    def test_required_cols(self, deliveries):
        result = impact_score(deliveries)
        for col in ["batter", "impact_score", "runs", "balls"]:
            assert col in result.columns


# ---------------------------------------------------------------------------
# top_run_scorers
# ---------------------------------------------------------------------------

class TestTopRunScorers:
    def test_returns_n_rows(self, deliveries):
        result = top_run_scorers(deliveries, n=5)
        assert len(result) <= 5

    def test_sorted_by_runs(self, deliveries):
        result = top_run_scorers(deliveries, n=10)
        runs = result["runs"].tolist()
        assert runs == sorted(runs, reverse=True)


# ---------------------------------------------------------------------------
# partnership_analysis
# ---------------------------------------------------------------------------

class TestPartnershipAnalysis:
    def test_pair_format(self, deliveries):
        result = partnership_analysis(deliveries)
        # Each pair name should be two components joined by underscore
        for pair in result["pair"].head(5):
            assert "_" in pair

    def test_sorted_by_runs(self, deliveries):
        result = partnership_analysis(deliveries)
        runs = result["total_runs"].tolist()
        assert runs == sorted(runs, reverse=True)
