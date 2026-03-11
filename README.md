# F1 Race Prediction (FastF1 + Streamlit)

This app predicts Formula 1 race results using recent race performance data from the **FastF1** API and exposes a clean **Streamlit** dashboard.

It stays up to date automatically, because schedules and results are pulled live from FastF1 for the selected season and race.

## Features

- **Live schedule**: load the full race calendar for any season.
- **Future race predictions**: choose any upcoming race and get a predicted finishing order.
- **Historical context**: model uses the most recent completed races across this and recent seasons.
- **Completed race analysis**:
  - Side‑by‑side **Prediction vs Actual** table.
  - Interactive scatter plot of predicted vs actual finishing positions (with diagonal “perfect prediction” line).
  - Optional compact backtest metrics in an expander.

## How the prediction works

For a selected race:

1. Collect up to **N recent races** (configurable in the sidebar) going backwards from that race, crossing seasons when needed.
2. Restrict to the **current grid** (drivers from the most recent race in that window).
3. For each driver, compute the **average finishing position** over those races.
4. Sort by that average to get the **predicted ranking**.

This is a simple but interpretable heuristic model; you can evolve it into a full ML model by adding more features and a learning algorithm.

## Local setup

```bash
git clone https://github.com/AdithyaReddyGeeda/F1-prediction.git
cd F1-prediction

pip install -r requirements.txt
streamlit run app.py
```

The first run may take longer because FastF1 needs to download and cache data.

## Usage

1. In the **sidebar**:
   - Pick a **season year**.
   - Choose how many **recent races** to use for the prediction.
2. In the main view:
   - Browse the **race calendar**.
   - Select a race from the dropdown and click **Run Prediction**.
3. For completed races:
   - Inspect the **Prediction vs Actual** section.
   - Optionally open the backtest metrics expander to see summary stats.

## Deployment on Streamlit Cloud

1. Push this repo to GitHub (already set up as `AdithyaReddyGeeda/F1-prediction`).
2. Go to Streamlit Cloud and create a **New app**:
   - Repo: `AdithyaReddyGeeda/F1-prediction`
   - Branch: `main`
   - Main file: `app.py`
3. Deploy. Streamlit will install dependencies from `requirements.txt` and run the app.

FastF1 will build its cache on the server filesystem at runtime; the cache directory is ignored in git via `.gitignore`.

