# F1 Race Prediction (FastF1 + Streamlit)

This app predicts **Formula 1 qualifying and race** results using the **FastF1** API and an **XGBoost** race model (with heuristic fallbacks). A **Streamlit** dashboard provides qualifying predictions, race predictions, and comparison with actual results.

Schedules and results are pulled from FastF1, with automatic fallback to 2025 data when the selected season (e.g. 2026) has no or insufficient results.

## Early 2026 / new season handling

- **Fallback**: The app tries the selected year first; if data is missing or the grid has fewer than 22 drivers, it **automatically** uses 2025 as baseline (no user toggle).
- **Grid**: 22 drivers and 11 teams come from FastF1, web scraping (official F1 site), or a hardcoded list in `config.py` for 2026.
- **Rookies / new teams**: Drivers with no F1 history get a default midfield form (~12); the model uses EWMA form, track-specific stats, and quali position where available.

## Features

- **Qualifying prediction**: Predict qualifying order (1–22) from driver/constructor quali form, circuit, and weather. Stored in session state and can be used as the race starting grid.
- **Race prediction**: XGBoost model (or heuristic) predicts finishing order from grid, form, track-specific stats, weather, teammate delta, DNF rate, and more.
- **Tabs**: **Qualifying** (predict quali → table + optional debug) and **Race** (grid from quali or manual, run race prediction, charts, prediction vs actual).
- **Override**: Check **“Override with manual grid”** in the Race tab to set the starting order by hand instead of using the predicted quali grid.
- **Live schedule**: Full race calendar for the chosen season.
- **Completed races**: **Prediction vs Actual** table and mean absolute error.
- **Debug**: Expanders for quali/race feature samples and top feature importances.

## How predictions work

1. **Qualifying**: Per-driver quali form (and constructor strength) from recent Q sessions; optional XGBoost quali model. Output is a predicted quali order 1–22.
2. **Race**:  
   - If you ran **Predict Qualifying**, that order is used as the starting grid unless you check **Override with manual grid**.  
   - The race model uses: grid position, quali position, EWMA driver/constructor form, track-specific averages, driver–team synergy, teammate delta, constructor DNF rate, momentum, weather (one-hot), circuit type, and interactions (e.g. grid × rain).  
   - Predictions are blended with quali (e.g. 0.65× race + 0.35× quali), then ranked per race and clipped to 1–22.  
   - If no trained model is available, a heuristic (form + grid + weather noise) is used.

## Local setup

```bash
git clone https://github.com/AdithyaReddyGeeda/F1-prediction.git
cd F1-prediction

pip install -r requirements.txt
streamlit run app.py
```

The first run can be slow while FastF1 downloads and caches data.

## Usage

1. **Sidebar**: Season **year** (default 2026), **Weather** (Dry / Wet / Rain).
2. **Race**: Select a round (e.g. Australian GP). Schedule is from FastF1 or a built-in 2026 start.
3. **Qualifying tab**: Click **Predict Qualifying** to get an order 1–22 and optionally inspect the debug expander. This grid is stored for the Race tab.
4. **Race tab**: By default the predicted quali grid is used. Check **Override with manual grid** to set positions 1–22 manually. Click **Run Race Prediction** for the predicted finishing order, charts, and (for past races) Prediction vs Actual and MAE.
5. **Track**: Simple Folium map and Plotly chart of predicted order.

## Deployment on Streamlit Cloud

1. Push the repo to GitHub (`AdithyaReddyGeeda/F1-prediction`).
2. In Streamlit Cloud, create a **New app**: connect the repo, branch `main`, main file `app.py`.
3. Deploy. Dependencies install from `requirements.txt`. FastF1 cache is created on the server (cache directory is in `.gitignore`).

## Training the models (race + qualifying)

To use the **XGBoost race and quali models** instead of heuristics:

```bash
pip install -r requirements.txt
python scripts/train_model.py
```

- **Race model**: Trains on 2020–2023, validates on 2024 (time-based split). Uses MAE objective, 80-trial hyperparameter search, StandardScaler on numeric features, and post-processing (blend with quali, rank per race). Saves **only if** validation MAE improves by more than 0.5 vs grid baseline or is below 7.0.
- **Outputs**:  
  - `model_artifacts/xgboost_model.joblib`, `model_artifacts/encoders.joblib` (encoders, scaler, lookup maps).  
  - `model_artifacts/diagnostics.png` (top 15 feature importances, error distribution).  
  - Console: baseline MAE, correlations (grid/form vs finish), validation MAE/RMSE/R², top 15 importances, sample pred vs actual, worst errors, per-race MAE.
- **Quali model**: Trains on 2020–2025 quali data; saves to `models/quali_model.joblib` and `models/quali_encoders.joblib`.

The app loads these artifacts automatically when you run **Predict Qualifying** and **Run Race Prediction**.

## How to improve MAE (target &lt;6.0)

The race model is tuned for **Mean Absolute Error** on a hold-out season. To push MAE lower:

1. **Features** (in `utils/race_features.py` and training):  
   EWMA driver/constructor form, track-specific driver/team averages, quali position, driver–team synergy, **teammate delta**, **constructor DNF rate**, **form × teammate_delta**, momentum, driver rain delta, circuit type, DNF imputation, grid × rain.

2. **Training**:  
   `reg:absoluteerror`, time-based train/val split, **RandomizedSearchCV** (e.g. 80 trials), early stopping, **StandardScaler** on numerics, oversample wet/rain if &lt;10%.

3. **Post-processing**:  
   Blend (e.g. 0.65× race + 0.35× quali), assign ranks per race (no ties), clip to 1–22.

4. **Retrain**:  
   Run `python scripts/train_model.py` and compare validation MAE and `diagnostics.png`; use the app’s debug expanders to inspect features and importances.
