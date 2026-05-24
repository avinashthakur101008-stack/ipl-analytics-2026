"""
tests/test_predictor.py
=======================
Unit tests for match_predictor module.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.match_predictor import WinProbabilityModel, _generate_dummy_training_data


class TestWinProbabilityModel:
    @pytest.fixture(scope="class")
    def fitted_model(self):
        model = WinProbabilityModel()
        model.fit(pd.DataFrame())  # uses dummy data
        return model

    def test_fit_sets_is_fitted(self, fitted_model):
        assert fitted_model.is_fitted

    def test_auc_above_threshold(self, fitted_model):
        auc = fitted_model.eval_metrics.get("auc", 0)
        assert auc > 0.55, f"AUC too low: {auc}"

    def test_predict_proba_range(self, fitted_model):
        prob = fitted_model.predict_proba(
            rrr=8.0, wickets_in_hand=7, current_over=12, crr=7.5
        )
        assert 0.0 <= prob <= 1.0

    def test_high_rrr_low_prob(self, fitted_model):
        """Very high RRR with few wickets should give low win probability."""
        prob = fitted_model.predict_proba(
            rrr=18.0, wickets_in_hand=2, current_over=18, crr=6.0
        )
        assert prob < 0.4, f"Expected low probability, got {prob}"

    def test_low_rrr_high_prob(self, fitted_model):
        """Very low RRR with many wickets should give high win probability."""
        prob = fitted_model.predict_proba(
            rrr=4.0, wickets_in_hand=9, current_over=10, crr=9.0
        )
        assert prob > 0.6, f"Expected high probability, got {prob}"

    def test_model_summary_string(self, fitted_model):
        summary = fitted_model.model_summary()
        assert "AUC" in summary
        assert "Log-Loss" in summary

    def test_rolling_win_probability(self, fitted_model):
        from src.data_loader import generate_synthetic_deliveries
        deliveries = generate_synthetic_deliveries(n_matches=5, seed=3)
        inn2 = deliveries[deliveries["innings"] == 2].copy()
        match_id = inn2["match_id"].iloc[0]
        chase = inn2[inn2["match_id"] == match_id]
        result = fitted_model.rolling_win_probability(target=175, deliveries_2nd=chase)
        assert isinstance(result, pd.DataFrame)
        assert "win_prob" in result.columns
        assert result["win_prob"].between(0, 1).all()


class TestDummyTrainingData:
    def test_shape(self):
        df = _generate_dummy_training_data(n=200)
        assert len(df) == 200
        assert "label" in df.columns

    def test_label_binary(self):
        df = _generate_dummy_training_data(n=500)
        assert set(df["label"].unique()).issubset({0, 1})
