"""
main.py
=======
CLI entry point for the IPL 2026 Analytics pipeline.

Usage
-----
    python main.py                    # full pipeline
    python main.py --module batting   # batting analysis only
    python main.py --module bowling
    python main.py --module team
    python main.py --module predict
    python main.py --no-plots         # skip chart generation
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
console = Console()


def _import_modules():
    """Deferred imports so --help is fast."""
    from src.data_loader import (
        load_deliveries,
        load_matches,
        generate_synthetic_deliveries,
        generate_synthetic_matches,
        save_processed,
    )
    from src.batting_analysis import (
        phase_strike_rates,
        boundary_percentage,
        consistency_index,
        impact_score,
        top_run_scorers,
    )
    from src.bowling_analysis import (
        bowling_summary,
        dot_ball_pressure_index,
        phase_economy,
        death_specialist_rating,
        top_wicket_takers,
    )
    from src.team_strategy import (
        powerplay_aggressiveness_score,
        middle_over_consolidation,
        season_nrr,
        chase_vs_set_profile,
    )
    from src.match_predictor import WinProbabilityModel
    from src import visualisations as viz

    return {
        "load_deliveries": load_deliveries,
        "load_matches": load_matches,
        "generate_synthetic_deliveries": generate_synthetic_deliveries,
        "generate_synthetic_matches": generate_synthetic_matches,
        "save_processed": save_processed,
        "phase_strike_rates": phase_strike_rates,
        "boundary_percentage": boundary_percentage,
        "consistency_index": consistency_index,
        "impact_score": impact_score,
        "top_run_scorers": top_run_scorers,
        "bowling_summary": bowling_summary,
        "dot_ball_pressure_index": dot_ball_pressure_index,
        "phase_economy": phase_economy,
        "death_specialist_rating": death_specialist_rating,
        "top_wicket_takers": top_wicket_takers,
        "powerplay_aggressiveness_score": powerplay_aggressiveness_score,
        "middle_over_consolidation": middle_over_consolidation,
        "season_nrr": season_nrr,
        "chase_vs_set_profile": chase_vs_set_profile,
        "WinProbabilityModel": WinProbabilityModel,
        "viz": viz,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _df_to_rich_table(df, title: str, max_rows: int = 10) -> Table:
    table = Table(title=title, show_header=True, header_style="bold magenta")
    for col in df.columns:
        table.add_column(str(col), style="cyan", no_wrap=False)
    for _, row in df.head(max_rows).iterrows():
        table.add_row(*[str(v) for v in row.values])
    return table


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def run_batting(m: dict, deliveries, plots: bool) -> None:
    console.rule("[bold yellow]🏏 BATTING ANALYSIS")

    trs = m["top_run_scorers"](deliveries)
    console.print(_df_to_rich_table(trs, "Top Run Scorers"))

    psr = m["phase_strike_rates"](deliveries)
    console.print(_df_to_rich_table(psr[["batter", "powerplay_sr", "middle_sr", "death_sr", "overall_sr"]], "Phase-wise Strike Rates"))

    imp = m["impact_score"](deliveries)
    console.print(_df_to_rich_table(imp[["batter", "innings", "impact_score"]], "Impact Scores"))

    ci = m["consistency_index"](deliveries)
    console.print(_df_to_rich_table(ci[["batter", "innings", "average", "consistency_index"]], "Consistency Index"))

    if plots:
        bp = m["boundary_percentage"](deliveries)
        m["viz"].plot_phase_strike_rates(psr)
        m["viz"].plot_boundary_scatter(bp)
        console.print("[green]✔ Batting charts saved to outputs/plots/")


def run_bowling(m: dict, deliveries, plots: bool) -> None:
    console.rule("[bold cyan]🎯 BOWLING ANALYSIS")

    twt = m["top_wicket_takers"](deliveries)
    console.print(_df_to_rich_table(twt[["bowler", "overs", "wickets", "economy", "dot_pct"]], "Top Wicket Takers"))

    dbpi = m["dot_ball_pressure_index"](deliveries)
    console.print(_df_to_rich_table(dbpi[["bowler", "dbpi", "total_dots", "economy"]], "Dot Ball Pressure Index"))

    dsr = m["death_specialist_rating"](deliveries)
    console.print(_df_to_rich_table(dsr[["bowler", "overs", "economy", "wpo", "dsr"]], "Death Specialist Rating"))

    if plots:
        pe = m["phase_economy"](deliveries)
        m["viz"].plot_economy_heatmap(pe)
        m["viz"].plot_death_specialist_rating(dsr)
        console.print("[green]✔ Bowling charts saved to outputs/plots/")


def run_team(m: dict, deliveries, matches, plots: bool) -> None:
    console.rule("[bold green]🏆 TEAM STRATEGY")

    pas = m["powerplay_aggressiveness_score"](deliveries)
    console.print(_df_to_rich_table(pas, "Powerplay Aggressiveness Score"))

    moc = m["middle_over_consolidation"](deliveries)
    console.print(_df_to_rich_table(moc, "Middle-Over Consolidation"))

    nrr = m["season_nrr"](matches)
    console.print(_df_to_rich_table(nrr[["team", "matches", "wins", "losses", "points", "nrr"]], "Season Points Table"))

    chase = m["chase_vs_set_profile"](matches)
    console.print(_df_to_rich_table(chase, "Chase vs Set Win %"))

    if plots and not nrr.empty:
        m["viz"].plot_points_table(nrr)
        console.print("[green]✔ Team charts saved to outputs/plots/")


def run_prediction(m: dict, deliveries, matches, plots: bool) -> None:
    console.rule("[bold red]📈 WIN PROBABILITY MODEL")

    model = m["WinProbabilityModel"]()
    model.fit(deliveries, matches)
    console.print(Panel(model.model_summary(), title="Model Summary", border_style="red"))

    # Demo prediction
    scenarios = [
        {"rrr": 6.0, "wickets_in_hand": 8, "current_over": 14, "crr": 9.0},
        {"rrr": 12.5, "wickets_in_hand": 4, "current_over": 17, "crr": 7.5},
        {"rrr": 8.0, "wickets_in_hand": 7, "current_over": 10, "crr": 7.8},
    ]
    table = Table(title="Win Probability Scenarios", header_style="bold magenta")
    for col in ["RRR", "Wickets", "Over", "CRR", "Win Prob"]:
        table.add_column(col, style="cyan")
    for s in scenarios:
        prob = model.predict_proba(**s)
        table.add_row(
            str(s["rrr"]), str(s["wickets_in_hand"]),
            str(s["current_over"]), str(s["crr"]),
            f"{prob:.1%}",
        )
    console.print(table)

    if plots:
        # Simulate a chase for visualisation
        import numpy as np
        rng = np.random.default_rng(7)
        chase_deliveries = deliveries[
            (deliveries["innings"] == 2) & (deliveries["match_id"] == deliveries["match_id"].iloc[0])
        ]
        if not chase_deliveries.empty:
            target = 175
            wp = model.rolling_win_probability(target, chase_deliveries)
            m["viz"].plot_win_probability(wp, team_name="Sample Chase", target=target)
            console.print("[green]✔ Win probability chart saved to outputs/plots/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IPL 2026 Data Analytics Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--module",
        choices=["batting", "bowling", "team", "predict", "all"],
        default="all",
        help="Which analysis module to run (default: all)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip chart generation",
    )
    args = parser.parse_args()
    plots = not args.no_plots

    console.print(
        Panel.fit(
            "[bold white]🏏 IPL Data Analytics Challenge 2026[/]\n"
            "[dim]Batting · Bowling · Strategy · Prediction[/]",
            border_style="yellow",
        )
    )

    t0 = time.time()
    m = _import_modules()

    console.print("[dim]Loading data...[/]")
    deliveries = m["load_deliveries"]()
    matches = m["load_matches"]()
    m["save_processed"](deliveries, "deliveries_clean")
    m["save_processed"](matches, "matches_clean")
    console.print(f"[green]✔ Data loaded:[/] {len(deliveries):,} deliveries · {len(matches):,} matches")

    mod = args.module
    if mod in ("batting", "all"):
        run_batting(m, deliveries, plots)
    if mod in ("bowling", "all"):
        run_bowling(m, deliveries, plots)
    if mod in ("team", "all"):
        run_team(m, deliveries, matches, plots)
    if mod in ("predict", "all"):
        run_prediction(m, deliveries, matches, plots)

    elapsed = time.time() - t0
    console.print(Panel(f"[bold green]✅ Pipeline complete in {elapsed:.1f}s[/]", border_style="green"))


if __name__ == "__main__":
    main()
