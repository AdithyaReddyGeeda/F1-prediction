# F1 Race Prediction (FastF1 + Streamlit)

This app predicts Formula 1 race results using recent race performance data from the **FastF1** API and exposes a clean **Streamlit** dashboard.

It stays up to date automatically, because schedules and results are pulled live from FastF1 for the selected season and race.

## Early 2026 handling: auto-fallback to 2025 + adjustments

For the **first race of the season** or when **2026 data is sparse** (e.g. Australian GP just run, API not yet updated):

1. **Fallback chain**: The app tries **FastF1 2026** first, then **2025 data** as baseline, then a **hardcoded 2026 grid** (22 drivers, 11 teams) so you always get a prediction.
2. **2026 grid**: Confirmed lineup is in `config.py` (Mercedes, Ferrari, McLaren, Red Bull Racing, Haas, Racing Bulls, Audi, Alpine, Williams, Cadillac, Aston Martin) with real driver names — no placeholders.
3. **Transfers and rookies**: Drivers who changed team (e.g. Hamilton → Ferrari, Perez → Cadillac) are mapped to 2026 teams; rookies (e.g. Antonelli, Lindblad, Bortoleto, Bearman) get a default **midfield form** (avg position ~12) when no F1 history exists.
4. **Manual override**: Use the sidebar option **“Force use 2025 data as baseline”** when you want to base form purely on last year.

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

1. Collect up to **N recent races** (configurable in the sidebar) going backwards from that race across **this and recent seasons**.
2. Infer the **current season grid** (drivers + teams) from the latest completed race of the selected year.
3. Restrict historical results to **only those drivers** on the current grid (so no old drivers that left F1).
4. For each current driver, compute the **average finishing position** over the collected races.
5. Sort by that average to get the **predicted ranking**, using the **current season’s team** for each driver.

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

1. **Sidebar**: Season **year** (default 2026), **Force use 2025 data as baseline**, and **Weather** (Dry / Wet / Rain).
2. **Race selector**: Choose a round (e.g. Australian GP, Chinese GP). Schedule comes from FastF1 or a hardcoded 2026 start.
3. **Grid order**: Assign **starting positions 1–22** to drivers (selectbox per position). Default order is the 2026 list.
4. Click **Run Prediction** to get predicted finishing order (XGBoost if trained, else heuristic + form).
5. **Track preview**: Simple Folium map and Plotly chart of predicted order.
6. For **completed races**, the app shows **Prediction vs Actual** and mean absolute error.

## Deployment on Streamlit Cloud

1. Push this repo to GitHub (already set up as `AdithyaReddyGeeda/F1-prediction`).
2. Go to Streamlit Cloud and create a **New app**:
   - Repo: `AdithyaReddyGeeda/F1-prediction`
   - Branch: `main`
   - Main file: `app.py`
3. Deploy. Streamlit will install dependencies from `requirements.txt` and run the app.

FastF1 will build its cache on the server filesystem at runtime; the cache directory is ignored in git via `.gitignore`.

## Optional: train XGBoost model

To use the ML model instead of the heuristic for predictions, train on 2020–2025 data (and optionally partial 2026):

```bash
pip install -r requirements.txt
python scripts/train_model.py
```

This saves `model_artifacts/xgboost_model.joblib` and `model_artifacts/encoders.joblib`. The app will load them automatically when you click **Run Prediction**.

## How to improve MAE (from ~7.45 → target &lt;6.0)

The race model is trained to minimize **Mean Absolute Error (MAE)** on hold-out seasons. To push MAE below 6.0 (ideally 5.0–5.5) without data leakage:

1. **Feature engineering** (in `utils/race_features.py` and training):
   - **EWMA form**: Exponentially weighted moving average of last 5–10 race positions (alpha ≈ 0.3–0.5) for driver and constructor.
   - **Track-specific performance**: Driver/constructor average finish at this circuit (last 3–5 visits).
   - **Quali strength**: Historical or predicted qualifying position (or gap to pole).
   - **Driver–team synergy**: Historical average finish for this driver with this constructor.
   - **Weather interactions**: Driver rain delta (avg position wet vs dry), team wet performance.
   - **Momentum**: Position change last race; relative to teammate.
   - **Circuit type**: Street / high-speed / technical dummies if derivable.
   - **DNF handling**: Impute finish position 20–22 from Status/laps when driver did not finish.

2. **Model and validation**:
   - Objective `reg:absoluteerror` (optimizes MAE directly).
   - **Time-series validation**: Train on earlier seasons (e.g. 2020–2023), validate on 2024, test on 2025 (no random split).
   - **Hyperparameter tuning**: RandomizedSearchCV or Optuna (n_estimators 300–1000, max_depth 3–8, learning_rate 0.01–0.1, subsample, colsample_bytree, reg_alpha/lambda, min_child_weight).
   - **Early stopping** on validation set.
   - **Feature selection**: Drop features with importance &lt; 0.01 after training.

3. **Post-processing and ensemble**:
   - Clip predictions to 1–22; optionally sort to enforce logical order.
   - Light ensemble: e.g. 60% race model + 40% quali-predicted position.

4. **Retrain and compare**:
   - Run `python scripts/train_model.py` and check the printed **validation MAE** and **top feature importances**.
   - Use the debug expander in the app to inspect feature values and importances.

