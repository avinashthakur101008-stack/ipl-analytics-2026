"""
tests/test_bowling.py
=====================
Unit tests for bowling_analysis module.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import generate_synthetic_deliveries
from src.bowling_analysis import (
    bowling_summary,
    death_specialist_rating,
    dot_ball_pressure_index,
    economy_bands,
    phase_economy,
    top_wicket_takers,
)


@pytest.fixture(scope="module")
def deliveries():
    return generate_synthetic_deliveries(n_matches=30, seed=7)


class TestBowlingSummary:
    def test_required_columns(self, deliveries):
        result = bowling_summary(deliveries)
        for col in ["bowler", "overs", "wickets", "economy", "dot_pct"]:
            assert col in result.columns

    def test_economy_positive(self, deliveries):
        result = bowling_summary(deliveries)
        assert (result["economy"] > 0).all()

    def test_dot_pct_range(self, deliveries):
        result = bowling_summary(deliveries)
        assert result["dot_pct"].between(0, 100).all()


class TestDotBallPressureIndex:
    def test_non_negative_dbpi(self, deliveries):
        result = dot_ball_pressure_index(deliveries)
        assert (result["dbpi"] >= 0).all()

    def test_sorted_desc(self, deliveries):
        result = dot_ball_pressure_index(deliveries)
        vals = result["dbpi"].tolist()
        assert vals == sorted(vals, reverse=True)


class TestPhaseEconomy:
    def test_columns_present(self, deliveries):
        result = phase_economy(deliveries)
        assert "bowler" in result.columns
        for col in ["powerplay_econ", "middle_econ", "death_econ"]:
            assert col in result.columns

    def test_no_negative_economy(self, deliveries):
        result = phase_economy(deliveries)
        for col in ["powerplay_econ", "middle_econ", "death_econ"]:
            assert (result[col].dropna() >= 0).all()


class TestDeathSpecialistRating:
    def test_dsr_positive(self, deliveries):
        result = death_specialist_rating(deliveries)
        if result.empty:
            pytest.skip("No bowlers with sufficient death overs.")
        assert (result["dsr"] >= 0).all()

    def test_sorted_desc(self, deliveries):
        result = death_specialist_rating(deliveries)
        if result.empty:
            pytest.skip("No bowlers with sufficient death overs.")
        dsrs = result["dsr"].tolist()
        assert dsrs == sorted(dsrs, reverse=True)


class TestEconomyBands:
    def test_valid_bands(self, deliveries):
        result = economy_bands(deliveries)
        assert result["band"].isin(["Tight", "Moderate", "Expensive"]).all()


class TestTopWicketTakers:
    def test_returns_n_rows(self, deliveries):
        result = top_wicket_takers(deliveries, n=5)
        assert len(result) <= 5
