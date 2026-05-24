"""
visualisations.py
=================
Publication-quality Matplotlib/Seaborn charts for IPL 2026 analysis.
All functions save to outputs/plots/ and optionally return the Figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

PLOT_DIR = Path("outputs/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

PALETTE = {
    "primary": "#1a1a2e",
    "accent1": "#e94560",
    "accent2": "#f5a623",
    "accent3": "#0f3460",
    "light": "#f0f0f0",
    "grid": "#2a2a4a",
}

def _apply_dark_theme(fig: plt.Figure, axes) -> None:
    fig.patch.set_facecolor(PALETTE["primary"])
    axs = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    for ax in np.array(axs).flatten():
        ax.set_facecolor(PALETTE["primary"])
        ax.tick_params(colors=PALETTE["light"], labelsize=9)
        ax.xaxis.label.set_color(PALETTE["light"])
        ax.yaxis.label.set_color(PALETTE["light"])
        if ax.get_title():
            ax.title.set_color(PALETTE["light"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["grid"])
        ax.yaxis.grid(True, color=PALETTE["grid"], linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# 1. Phase-wise strike rate — grouped bar chart
# ---------------------------------------------------------------------------

def plot_phase_strike_rates(
    phase_sr_df: pd.DataFrame,
    top_n: int = 10,
    save: bool = True,
) -> plt.Figure:
    """Grouped bar chart of powerplay / middle / death strike rates for top batters."""
    df = phase_sr_df.dropna(subset=["powerplay_sr", "death_sr"]).head(top_n)

    phases = ["powerplay_sr", "middle_sr", "death_sr"]
    labels = ["Powerplay", "Middle", "Death"]
    colors = [PALETTE["accent2"], "#4ecdc4", PALETTE["accent1"]]

    x = np.arange(len(df))
    width = 0.28

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (col, label, color) in enumerate(zip(phases, labels, colors)):
        bars = ax.bar(
            x + i * width,
            df[col].fillna(0),
            width,
            label=label,
            color=color,
            alpha=0.9,
            zorder=3,
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1,
                f"{h:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=PALETTE["light"],
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(df["batter"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Strike Rate", fontsize=10)
    ax.set_title(f"Phase-wise Strike Rate — Top {top_n} Batters (IPL 2026)", fontsize=13, pad=15)
    ax.legend(framealpha=0.2, labelcolor="white")
    ax.set_ylim(0, df[phases].max().max() * 1.18)
    _apply_dark_theme(fig, ax)

    if save:
        path = PLOT_DIR / "phase_strike_rates.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2. Economy heatmap (bowler × phase)
# ---------------------------------------------------------------------------

def plot_economy_heatmap(
    phase_econ_df: pd.DataFrame,
    top_n: int = 15,
    save: bool = True,
) -> plt.Figure:
    """Heatmap of bowler economy across powerplay / middle / death."""
    cols = ["powerplay_econ", "middle_econ", "death_econ"]
    df = phase_econ_df.dropna(subset=cols).head(top_n).set_index("bowler")[cols]
    df.columns = ["Powerplay", "Middle", "Death"]

    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.45)))
    sns.heatmap(
        df,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        linewidths=0.5,
        linecolor="#333",
        ax=ax,
        cbar_kws={"label": "Economy Rate"},
        vmin=5,
        vmax=14,
    )
    ax.set_title("Bowler Economy by Phase — IPL 2026", fontsize=13, pad=12)
    ax.set_ylabel("")
    _apply_dark_theme(fig, ax)

    if save:
        path = PLOT_DIR / "economy_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 3. Win probability — rolling over-by-over line chart
# ---------------------------------------------------------------------------

def plot_win_probability(
    win_prob_df: pd.DataFrame,
    team_name: str = "Chasing Team",
    target: int = 0,
    save: bool = True,
) -> plt.Figure:
    """Rolling win-probability chart for a single match chase."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(
        win_prob_df["over"],
        win_prob_df["win_prob"],
        alpha=0.25,
        color=PALETTE["accent1"],
    )
    ax.plot(
        win_prob_df["over"],
        win_prob_df["win_prob"],
        color=PALETTE["accent1"],
        linewidth=2.5,
        marker="o",
        markersize=5,
        zorder=5,
    )
    ax.axhline(0.5, color=PALETTE["accent2"], linestyle="--", alpha=0.7, linewidth=1.2)
    ax.text(1, 0.52, "50% line", color=PALETTE["accent2"], fontsize=8)

    ax.set_ylim(0, 1)
    ax.set_xlim(1, 20)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Over", fontsize=10)
    ax.set_ylabel("Win Probability", fontsize=10)
    title = f"Win Probability — {team_name}"
    if target:
        title += f" (Target: {target})"
    ax.set_title(title, fontsize=13, pad=12)

    # Annotate final prob
    final = win_prob_df["win_prob"].iloc[-1]
    ax.annotate(
        f"Final: {final:.0%}",
        xy=(win_prob_df["over"].iloc[-1], final),
        xytext=(-40, 12),
        textcoords="offset points",
        fontsize=9,
        color=PALETTE["light"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["light"], lw=1),
    )
    _apply_dark_theme(fig, ax)

    if save:
        path = PLOT_DIR / "win_probability.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 4. Points table (season NRR) — horizontal bar
# ---------------------------------------------------------------------------

def plot_points_table(nrr_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Horizontal bar chart of team points and NRR."""
    if nrr_df.empty:
        return plt.figure()

    df = nrr_df.sort_values("points")
    colors = [PALETTE["accent1"] if n >= 0 else "#888" for n in df["nrr"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})

    # Points
    ax1.barh(df["team"], df["points"], color=PALETTE["accent2"], alpha=0.9, zorder=3)
    for i, (pts, team) in enumerate(zip(df["points"], df["team"])):
        ax1.text(pts + 0.3, i, str(int(pts)), va="center", fontsize=9, color=PALETTE["light"])
    ax1.set_xlabel("Points", fontsize=10)
    ax1.set_title("IPL 2026 Points Table", fontsize=12)

    # NRR
    ax2.barh(df["team"], df["nrr"], color=colors, alpha=0.9, zorder=3)
    ax2.axvline(0, color=PALETTE["light"], linewidth=0.8)
    ax2.set_xlabel("Net Run Rate", fontsize=10)
    ax2.set_title("NRR", fontsize=12)
    ax2.yaxis.set_visible(False)

    _apply_dark_theme(fig, [ax1, ax2])
    fig.suptitle("Season Standings — IPL 2026", color=PALETTE["light"], fontsize=14, y=1.02)
    fig.tight_layout()

    if save:
        path = PLOT_DIR / "points_table.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 5. Boundary percentage — scatter plot
# ---------------------------------------------------------------------------

def plot_boundary_scatter(
    boundary_df: pd.DataFrame,
    min_balls: int = 50,
    save: bool = True,
) -> plt.Figure:
    """Strike rate vs boundary % scatter for qualifying batters."""
    df = boundary_df[boundary_df["balls"] >= min_balls].copy()
    df["sr"] = (df["runs"] / df["balls"] * 100).round(2)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        df["boundary_pct"],
        df["sr"],
        s=df["runs"] / 5,
        c=df["boundary_pct"],
        cmap="plasma",
        alpha=0.8,
        zorder=5,
        edgecolors="white",
        linewidths=0.4,
    )
    plt.colorbar(scatter, ax=ax, label="Boundary %")

    ax.set_xlabel("Boundary %", fontsize=10)
    ax.set_ylabel("Strike Rate", fontsize=10)
    ax.set_title("Boundary % vs Strike Rate — IPL 2026 Batters\n(bubble size = total runs)", fontsize=12)
    _apply_dark_theme(fig, ax)

    if save:
        path = PLOT_DIR / "boundary_scatter.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 6. Death-over specialist rating — lollipop chart
# ---------------------------------------------------------------------------

def plot_death_specialist_rating(dsr_df: pd.DataFrame, top_n: int = 12, save: bool = True) -> plt.Figure:
    """Lollipop chart for Death Specialist Rating."""
    df = dsr_df.head(top_n).sort_values("dsr")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hlines(df["bowler"], 0, df["dsr"], color=PALETTE["grid"], linewidth=2)
    ax.plot(df["dsr"], df["bowler"], "o", color=PALETTE["accent1"], markersize=10, zorder=5)

    for _, row in df.iterrows():
        ax.text(row["dsr"] + 0.002, row["bowler"], f'{row["dsr"]:.3f}', va="center", fontsize=8, color=PALETTE["light"])

    ax.set_xlabel("Death Specialist Rating (DSR)", fontsize=10)
    ax.set_title("Top Death Bowlers — IPL 2026\n(overs 17–20)", fontsize=12)
    _apply_dark_theme(fig, ax)
    fig.tight_layout()

    if save:
        path = PLOT_DIR / "death_specialist_rating.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
