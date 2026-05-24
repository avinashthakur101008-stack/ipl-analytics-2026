# 🏏 IPL Data Analytics Challenge 2026

> A production-grade Python analytics project for the Indian Premier League 2026 season — featuring player performance modelling, team strategy analysis, match outcome prediction, and interactive visualisations.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green?style=flat-square&logo=pandas)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-orange?style=flat-square&logo=github-actions)

---

## 📌 Project Overview

This project provides a complete analytics pipeline for IPL 2026 data, covering:

| Module | What It Does |
|---|---|
| `data_loader.py` | Ingests & cleans raw ball-by-ball / match-level data |
| `batting_analysis.py` | Strike rates, phase-wise scoring, boundary %, consistency index |
| `bowling_analysis.py` | Economy buckets, death-over specialists, dot-ball pressure index |
| `team_strategy.py` | Powerplay aggressiveness, middle-over consolidation, death finishing |
| `match_predictor.py` | Logistic regression win-probability model, DLS-aware features |
| `visualisations.py` | Publication-quality Matplotlib/Seaborn charts |
| `dashboard.py` | Streamlit interactive dashboard |

---

## 🗂️ Repository Structure

```
ipl-analytics-2026/
├── data/
│   ├── raw/                  # Place IPL CSV data files here
│   │   ├── matches.csv
│   │   └── deliveries.csv
│   └── processed/            # Auto-generated cleaned datasets
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── batting_analysis.py
│   ├── bowling_analysis.py
│   ├── team_strategy.py
│   ├── match_predictor.py
│   └── visualisations.py
├── notebooks/
│   └── IPL_2026_EDA.ipynb    # Full exploratory analysis walkthrough
├── outputs/
│   └── plots/                # Auto-saved charts
├── tests/
│   ├── test_batting.py
│   ├── test_bowling.py
│   └── test_predictor.py
├── dashboard.py              # Streamlit app entry point
├── main.py                   # CLI pipeline runner
├── requirements.txt
├── .github/
│   └── workflows/
│       └── ci.yml
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/ipl-analytics-2026.git
cd ipl-analytics-2026
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add Your Data

Download IPL ball-by-ball CSVs from [Cricsheet](https://cricsheet.org/matches/) and place them in `data/raw/`:

```
data/raw/matches.csv
data/raw/deliveries.csv
```

### 3. Run the Pipeline

```bash
# Full analysis pipeline
python main.py

# Run only batting analysis
python main.py --module batting

# Launch interactive dashboard
streamlit run dashboard.py
```

---

## 📊 Key Analyses

### Batting

- **Phase-wise Strike Rate** — Powerplay (1–6), Middle (7–15), Death (16–20)
- **Consistency Index** — Runs per dismissal weighted by match importance
- **Boundary % by Bowler Type** — How batters handle pace vs spin
- **Impact Score** — Custom metric combining SR, avg, and chase success rate

### Bowling

- **Dot Ball Pressure Index (DBPI)** — Consecutive dot-ball sequences per spell
- **Economy Band Analysis** — Classifying bowlers into tight/moderate/expensive
- **Death Specialist Rating** — Weighted economy + wickets in overs 17–20
- **Match Phase Heatmap** — Wicket probability per over per bowler

### Team Strategy

- **Powerplay Aggressiveness Score** — Team-level run-rate vs wicket-risk trade-off
- **Middle-Over Consolidation** — Net run-rate delta in overs 7–15
- **Toss & Venue Interaction** — Win-rate by toss decision × ground × innings

### Prediction Model

- Logistic regression win-probability model trained on historical IPL data
- Features: current RRR, wickets in hand, phase, venue, head-to-head
- Rolling win-probability plot per match

---

## 📈 Sample Outputs

```
=== Top 10 Batters — Death-Over Strike Rate (Overs 17-20) ===
  Batter                 Runs   SR     Boundary%   Matches
  Tilak Varma            312   198.7   48.1%       14
  Heinrich Klaasen       278   215.5   52.3%       11
  ...

=== Bowling Economy Summary (Middle Overs 7-15) ===
  Bowler                 Overs  Runs  Wkts  Econ   DBPI
  Rashid Khan            42.3   278   18    6.54   0.71
  Jasprit Bumrah         38.1   241   22    6.31   0.84
  ...
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v --tb=short
```

---

## 🖥️ Dashboard Preview

The Streamlit dashboard (`dashboard.py`) offers:

- **Player Comparison** — Head-to-head stat comparison with radar charts
- **Match Simulator** — Win-probability slider for any chase scenario
- **Team Heatmaps** — Phase-wise scoring maps for all 10 franchises
- **Season Leaderboards** — Live-sortable tables for all batting/bowling metrics

```bash
streamlit run dashboard.py
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-analysis`
3. Commit your changes: `git commit -m 'Add powerplay aggression module'`
4. Push & open a Pull Request

Please follow the existing code style (Black + type hints) and add tests for new modules.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Data Credits

- [Cricsheet](https://cricsheet.org) — Ball-by-ball IPL data (YAML/CSV)
- [ESPNcricinfo](https://www.espncricinfo.com) — Supplementary player metadata
