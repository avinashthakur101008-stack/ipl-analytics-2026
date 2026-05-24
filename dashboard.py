"""
dashboard.py
============
Interactive Streamlit dashboard for IPL 2026 analytics.

Run with:
    streamlit run dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="IPL 2026 Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { color: #f5a623; }
    .stMetric label { color: #aaa !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading ball-by-ball data…")
def load_data():
    from src.data_loader import load_deliveries, load_matches
    return load_deliveries(), load_matches()


@st.cache_data(show_spinner="Computing batting metrics…")
def get_batting(_deliveries):
    from src.batting_analysis import (
        boundary_percentage,
        consistency_index,
        impact_score,
        phase_strike_rates,
        top_run_scorers,
    )
    return {
        "top_scorers": top_run_scorers(_deliveries),
        "phase_sr": phase_strike_rates(_deliveries),
        "boundary_pct": boundary_percentage(_deliveries),
        "consistency": consistency_index(_deliveries),
        "impact": impact_score(_deliveries),
    }


@st.cache_data(show_spinner="Computing bowling metrics…")
def get_bowling(_deliveries):
    from src.bowling_analysis import (
        bowling_summary,
        death_specialist_rating,
        dot_ball_pressure_index,
        economy_bands,
        phase_economy,
    )
    return {
        "summary": bowling_summary(_deliveries),
        "dbpi": dot_ball_pressure_index(_deliveries),
        "phase_econ": phase_economy(_deliveries),
        "dsr": death_specialist_rating(_deliveries),
        "bands": economy_bands(_deliveries),
    }


@st.cache_data(show_spinner="Computing team metrics…")
def get_team(_deliveries, _matches):
    from src.team_strategy import (
        chase_vs_set_profile,
        middle_over_consolidation,
        powerplay_aggressiveness_score,
        season_nrr,
    )
    return {
        "pas": powerplay_aggressiveness_score(_deliveries),
        "moc": middle_over_consolidation(_deliveries),
        "nrr": season_nrr(_matches),
        "chase_profile": chase_vs_set_profile(_matches),
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🏏 IPL 2026 Analytics")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Overview", "🏏 Batting", "🎯 Bowling", "🏆 Teams", "📈 Win Predictor"],
)

deliveries, matches = load_data()

# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

if page == "📊 Overview":
    st.title("IPL 2026 — Season Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Matches", len(matches))
    col2.metric("Total Deliveries", f"{len(deliveries):,}")
    col3.metric("Teams", deliveries["batting_team"].nunique())
    col4.metric("Unique Batters", deliveries["batter"].nunique())

    st.markdown("---")
    st.subheader("Season Run Distribution by Phase")
    phase_runs = (
        deliveries.groupby("phase")["total_runs"]
        .sum()
        .reindex(["powerplay", "middle", "death"])
    )
    st.bar_chart(phase_runs)

    st.subheader("Matches Played Over Season")
    if "date" in matches.columns:
        match_dates = matches.groupby(pd.to_datetime(matches["date"]).dt.date).size()
        st.line_chart(match_dates)


# ---------------------------------------------------------------------------
# Page: Batting
# ---------------------------------------------------------------------------

elif page == "🏏 Batting":
    st.title("Batting Analysis — IPL 2026")
    batting = get_batting(deliveries)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Top Scorers", "Phase Strike Rates", "Impact Score", "Consistency"]
    )

    with tab1:
        st.subheader("Top Run Scorers")
        st.dataframe(batting["top_scorers"], use_container_width=True)

    with tab2:
        st.subheader("Phase-wise Strike Rate")
        df = batting["phase_sr"].dropna(subset=["powerplay_sr", "death_sr"]).head(15)
        st.dataframe(df, use_container_width=True)
        phases = ["powerplay_sr", "middle_sr", "death_sr"]
        chart_data = df.set_index("batter")[phases].fillna(0)
        st.bar_chart(chart_data)

    with tab3:
        st.subheader("Batter Impact Score")
        imp = batting["impact"]
        if not imp.empty:
            st.dataframe(imp.head(20), use_container_width=True)
            top10 = imp.head(10).set_index("batter")["impact_score"]
            st.bar_chart(top10)
        else:
            st.info("Insufficient balls faced for impact score computation.")

    with tab4:
        st.subheader("Consistency Index")
        st.dataframe(batting["consistency"].head(20), use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Bowling
# ---------------------------------------------------------------------------

elif page == "🎯 Bowling":
    st.title("Bowling Analysis — IPL 2026")
    bowling = get_bowling(deliveries)

    tab1, tab2, tab3 = st.tabs(
        ["Overview", "Dot Ball Pressure", "Death Specialists"]
    )

    with tab1:
        st.subheader("Season Bowling Summary")
        st.dataframe(bowling["summary"].head(20), use_container_width=True)
        st.subheader("Economy Bands")
        bands = bowling["bands"]
        band_counts = bands["band"].value_counts()
        st.bar_chart(band_counts)

    with tab2:
        st.subheader("Dot Ball Pressure Index (DBPI)")
        st.caption("Higher DBPI = bowler strings together more consecutive dot balls")
        top_dbpi = bowling["dbpi"].head(15).set_index("bowler")["dbpi"]
        st.bar_chart(top_dbpi)
        st.dataframe(bowling["dbpi"].head(15), use_container_width=True)

    with tab3:
        st.subheader("Death Over Specialist Rating (Overs 17–20)")
        dsr = bowling["dsr"]
        if not dsr.empty:
            st.dataframe(dsr.head(15), use_container_width=True)
            st.bar_chart(dsr.head(12).set_index("bowler")["dsr"])
        else:
            st.info("No bowlers with ≥3 death overs yet.")


# ---------------------------------------------------------------------------
# Page: Teams
# ---------------------------------------------------------------------------

elif page == "🏆 Teams":
    st.title("Team Strategy — IPL 2026")
    team = get_team(deliveries, matches)

    tab1, tab2, tab3 = st.tabs(
        ["Points Table", "Powerplay Aggressiveness", "Chase vs Set"]
    )

    with tab1:
        st.subheader("Season Standings")
        nrr = team["nrr"]
        if not nrr.empty:
            st.dataframe(nrr, use_container_width=True)
            st.bar_chart(nrr.set_index("team")["points"])

    with tab2:
        st.subheader("Powerplay Aggressiveness Score")
        st.caption("Higher PAS = aggressive batting in powerplay without losing wickets early")
        pas = team["pas"]
        st.dataframe(pas, use_container_width=True)
        st.bar_chart(pas.set_index("batting_team")["pas"])

    with tab3:
        st.subheader("Chasing vs Setting Win %")
        chase = team["chase_profile"]
        if not chase.empty:
            st.dataframe(chase, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Win Predictor
# ---------------------------------------------------------------------------

elif page == "📈 Win Predictor":
    st.title("Match Win Probability Simulator")
    st.caption("Logistic regression model trained on IPL 2026 data")

    col1, col2 = st.columns(2)
    with col1:
        rrr = st.slider("Required Run Rate (RRR)", 4.0, 24.0, 10.0, 0.5)
        wickets = st.slider("Wickets in Hand", 1, 10, 7)
    with col2:
        current_over = st.slider("Current Over (0 = start of over 1)", 0, 19, 10)
        crr = st.slider("Current Run Rate (CRR)", 3.0, 16.0, 8.0, 0.5)

    if st.button("🔮 Predict Win Probability", type="primary"):
        with st.spinner("Running model…"):
            from src.match_predictor import WinProbabilityModel
            model = WinProbabilityModel()
            model.fit(deliveries, matches)
            prob = model.predict_proba(rrr=rrr, wickets_in_hand=wickets, current_over=current_over, crr=crr)

        st.metric("Win Probability (Chasing Team)", f"{prob:.1%}")
        if prob > 0.6:
            st.success("Chasing team is in a strong position! 💪")
        elif prob < 0.4:
            st.error("Chasing team is under pressure 😰")
        else:
            st.warning("It's a coin toss from here! 🪙")

        st.caption(model.model_summary())
