# NBA Playoff Predictor

Predict NBA playoff game outcomes using team stats (offensive/defensive rating, rest days, etc.) and logistic regression.

## Tech stack

- [nba_api](https://github.com/swar/nba_api) — NBA stats & game data
- Pandas — data wrangling
- scikit-learn — logistic regression
- Seaborn / Matplotlib — exploration & visualization

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Project roadmap (learning steps)

| Step | Topic | Goal |
|------|--------|------|
| 1 | Project setup | Repo, venv, dependencies |
| 2 | Explore the NBA API | Fetch one endpoint; understand response shape |
| 3 | Pull historical games | Download regular-season games (last 10 seasons) |
| 4 | Team ratings | Offensive/defensive rating per team per game |
| 5 | Rest days | Days since last game for each team |
| 6 | Build a dataset | One row per game with features + winner label |
| 7 | EDA | Seaborn plots; sanity-check features |
| 8 | Logistic regression | Train/test split; baseline model |
| 9 | Evaluate & iterate | Accuracy, calibration, feature importance |
| 10 | Playoff focus | Filter to playoff games; compare vs regular season |

## Structure (will grow as we go)

```
nba-playoff-predictor/
├── data/           # raw & processed (gitignored)
├── notebooks/      # exploratory work
├── src/            # reusable functions
└── scripts/        # one-off fetch / train scripts
```
