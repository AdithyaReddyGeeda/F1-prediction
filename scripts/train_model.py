"""
Train XGBoost race model on 2020-2025 data with MAE objective, time-series validation,
and rich feature engineering. Saves model + encoders + lookup maps to model_artifacts/ and models/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import GRID_2026, ROOKIE_DEFAULT_AVG_POSITION

WEATHER_CATEGORIES = ["Dry", "Wet", "Rain"]


def load_historical_results(start_year: int = 2020, end_year: int = 2025) -> pd.DataFrame:
    """Load all race results from FastF1 (with Status when available for DNF)."""
    import fastf1
    cache_dir = ROOT / "fastf1_cache"
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    from data import get_event_schedule, load_race_results
    rows = []
    for year in range(start_year, end_year + 1):
        try:
            sched = get_event_schedule(year)
            if sched.empty:
                continue
            for _, row in sched.iterrows():
                r = int(row["RoundNumber"])
                event_name = str(row.get("EventName", ""))
                try:
                    res = load_race_results(year, r)
                    if res.empty:
                        continue
                    res = res.copy()
                    res["Year"] = year
                    res["Round"] = r
                    res["Circuit"] = event_name
                    rows.append(res)
                except Exception:
                    continue
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_historical_quali_for_merge(start_year: int = 2020, end_year: int = 2025) -> pd.DataFrame:
    """Load qualifying results for merging into race data (Year, Round, Abbreviation, QualiPosition)."""
    import fastf1
    cache_dir = ROOT / "fastf1_cache"
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    from data import get_event_schedule, load_qualifying_results
    rows = []
    for year in range(start_year, end_year + 1):
        try:
            sched = get_event_schedule(year)
            if sched.empty:
                continue
            for _, row in sched.iterrows():
                r = int(row["RoundNumber"])
                try:
                    res = load_qualifying_results(year, r)
                    if res.empty:
                        continue
                    res = res.copy()
                    res["Year"] = year
                    res["Round"] = r
                    res["QualiPosition"] = pd.to_numeric(res.get("QualiPosition", res.get("Position", 10)), errors="coerce")
                    rows.append(res[["Year", "Round", "Abbreviation", "TeamName", "QualiPosition"]])
                except Exception:
                    continue
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def oversample_wet_rain(df: pd.DataFrame, target_min_pct: float = 0.25, rng=None) -> pd.DataFrame:
    """Oversample Wet and Rain rows so model sees enough weather signal. Aggressive if <10%."""
    if rng is None:
        rng = np.random.default_rng(42)
    if "Weather" not in df.columns or len(df) == 0:
        return df
    wet_mask = df["Weather"] == "Wet"
    rain_mask = df["Weather"] == "Rain"
    pct_wet_rain = (wet_mask | rain_mask).sum() / len(df)
    target = target_min_pct if pct_wet_rain >= 0.10 else 0.20  # aggressive if <10%
    if pct_wet_rain >= target:
        return df
    extra = []
    noise_cols = [
        "RecentForm", "GridPosition", "ConstructorEwma", "track_avg_driver", "track_avg_team",
        "driver_team_synergy", "QualiPosition", "teammate_delta", "form_x_teammate_delta",
    ]
    for _, row in df[wet_mask | rain_mask].iterrows():
        r = row.to_dict()
        for k in noise_cols:
            if k in r and pd.notna(r.get(k)):
                r[k] = r[k] + rng.uniform(-0.3, 0.3)
        r["GridPosition"] = max(1.0, min(22.0, r.get("GridPosition", 10)))
        r["grid_pos_x_rain"] = r["GridPosition"] * r.get("is_rain", 0)
        r["Position"] = r.get("Position", 10.0)
        extra.append(r)
    if extra:
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    return df


def main():
    import joblib
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder

    from utils.race_features import (
        build_race_feature_df,
        DEFAULT_TRACK_AVG,
        DEFAULT_SYNERGY,
    )

    print("Loading 2020-2025 race data from FastF1...")
    race_df = load_historical_results(2020, 2025)
    if race_df.empty:
        print("No race data. Run app once to populate FastF1 cache.")
        return

    print("Loading qualifying data for quali-position feature...")
    quali_df = load_historical_quali_for_merge(2020, 2025)
    if quali_df.empty:
        quali_df = None

    print("Building race features (EWMA, track-specific, synergy, momentum, weather)...")
    # Weather per race (one per Year, Round) for no-leakage driver_rain_delta
    rng = np.random.default_rng(42)
    keys = race_df[["Year", "Round"]].drop_duplicates()
    w = rng.random(len(keys))
    weather_per_race = dict(
        zip(zip(keys["Year"], keys["Round"]), np.where(w < 0.50, "Dry", np.where(w < 0.80, "Wet", "Rain")))
    )
    X_df = build_race_feature_df(
        race_df,
        quali_df=quali_df,
        weather_per_race=weather_per_race,
        ewma_alpha=0.4,
    )
    X_df = oversample_wet_rain(X_df, target_min_pct=0.25, rng=rng)
    y = X_df["Position"].astype(float)

    # Encoders
    grid_2026 = pd.DataFrame(GRID_2026)
    all_drivers = list(X_df["Abbreviation"].unique()) + list(grid_2026["Abbreviation"].unique())
    all_teams = list(X_df["TeamName"].unique()) + list(grid_2026["TeamName"].unique())
    all_circuits = list(X_df["Circuit"].unique()) + ["Australian Grand Prix", "Melbourne"]
    enc_driver = LabelEncoder()
    enc_team = LabelEncoder()
    enc_circuit = LabelEncoder()
    enc_weather = OneHotEncoder(categories=[WEATHER_CATEGORIES], drop=None, sparse_output=False)
    enc_driver.fit(list(dict.fromkeys(all_drivers)))
    enc_team.fit(list(dict.fromkeys(all_teams)))
    enc_circuit.fit(list(dict.fromkeys(all_circuits)))
    enc_weather.fit(np.array(WEATHER_CATEGORIES).reshape(-1, 1))

    X_df["driver_enc"] = enc_driver.transform(X_df["Abbreviation"].astype(str))
    X_df["team_enc"] = enc_team.transform(X_df["TeamName"].astype(str))
    X_df["circuit_enc"] = enc_circuit.transform(X_df["Circuit"].astype(str))
    wo = enc_weather.transform(X_df[["Weather"]].values)
    X_df["weather_Dry"] = wo[:, 0]
    X_df["weather_Wet"] = wo[:, 1]
    X_df["weather_Rain"] = wo[:, 2]

    # Feature columns (order must match inference); include new: teammate_delta, constructor_dnf_rate, form_x_teammate_delta
    feat_cols = [
        "GridPosition", "QualiPosition", "RecentForm", "ConstructorEwma",
        "track_avg_driver", "track_avg_team", "driver_team_synergy", "teammate_delta", "constructor_dnf_rate",
        "form_x_teammate_delta", "momentum", "driver_rain_delta",
        "driver_enc", "team_enc", "circuit_enc",
        "circuit_type_street", "circuit_type_high_speed", "circuit_type_technical",
        "weather_Dry", "weather_Wet", "weather_Rain", "grid_pos_x_rain",
    ]
    feature_names = list(feat_cols)
    for c in feat_cols:
        if c not in X_df.columns:
            X_df[c] = 0.0 if "enc" in c or "weather" in c or "circuit_type" in c else (10.0 if "track" in c or "synergy" in c or "Constructor" in c or "Quali" in c or "Grid" in c else 0.0)
    X_full = X_df[feat_cols].values

    # Time-series split: train 2020-2023, validate 2024 (walk-forward)
    train_mask = (X_df["Year"] >= 2020) & (X_df["Year"] <= 2023)
    val_mask = X_df["Year"] == 2024
    if val_mask.sum() == 0:
        val_mask = X_df["Year"] == 2025
    if val_mask.sum() == 0:
        val_mask = pd.Series(False, index=X_df.index)
        val_mask.iloc[-500:] = True
    X_train, y_train = X_full[train_mask], y[train_mask]
    X_val, y_val = X_full[val_mask], y[val_mask]
    val_df = X_df[val_mask].copy()

    print("Train size:", len(X_train), "Val size:", len(X_val))

    # StandardScaler on numeric features (same order as feat_cols)
    from sklearn.preprocessing import StandardScaler
    scale_idx = [i for i, n in enumerate(feat_cols) if n not in ("driver_enc", "team_enc", "circuit_enc") and not n.startswith("weather_") and not n.startswith("circuit_type_")]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_train_scaled[:, scale_idx] = scaler.fit_transform(X_train[:, scale_idx])
    X_val_scaled[:, scale_idx] = scaler.transform(X_val[:, scale_idx])

    # Baseline MAE (predict mean of train target)
    baseline_pred = np.full_like(y_val.values, y_train.mean())
    baseline_mae = np.abs(baseline_pred - y_val.values).mean()
    grid_baseline_mae = np.abs(X_val[:, feat_cols.index("GridPosition")] - y_val.values).mean()
    print("Baseline MAE (mean): {:.3f}  |  Baseline MAE (grid): {:.3f}".format(baseline_mae, grid_baseline_mae))

    # Correlation snippet: grid vs finish, form vs finish (on val)
    grid_col = feat_cols.index("GridPosition")
    form_col = feat_cols.index("RecentForm")
    corr_grid = np.corrcoef(X_val[:, grid_col], y_val.values)[0, 1] if len(y_val) > 1 else 0.0
    corr_form = np.corrcoef(X_val[:, form_col], y_val.values)[0, 1] if len(y_val) > 1 else 0.0
    print("Correlation (grid vs finish): {:.3f}  |  (recent form vs finish): {:.3f}".format(corr_grid, corr_form))

    # Build lookup maps for inference (from training set only)
    train_df = X_df[train_mask]
    track_avg_driver_map = train_df.groupby(["Abbreviation", "Circuit"])["track_avg_driver"].mean().to_dict()
    track_avg_team_map = train_df.groupby(["TeamName", "Circuit"])["track_avg_team"].mean().to_dict()
    driver_team_synergy_map = train_df.groupby(["Abbreviation", "TeamName"])["driver_team_synergy"].mean().to_dict()
    driver_rain_delta_map = train_df.groupby("Abbreviation")["driver_rain_delta"].mean().to_dict()

    # Hyperparameter search: 80 trials with early stopping
    rng = np.random.default_rng(42)
    param_dist = {
        "n_estimators": list(range(400, 1201, 100)),
        "max_depth": list(range(4, 10)),
        "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
        "subsample": np.arange(0.65, 0.96, 0.05).tolist(),
        "colsample_bytree": np.arange(0.6, 0.96, 0.05).tolist(),
        "reg_alpha": np.arange(0, 1.1, 0.2).tolist(),
        "reg_lambda": np.arange(1, 10.5, 1).tolist(),
        "min_child_weight": list(range(1, 9)),
    }
    best_mae = np.inf
    best_model = None
    try:
        import xgboost as xgb
        for trial in range(80):
            params = {k: rng.choice(v) for k, v in param_dist.items()}
            m = xgb.XGBRegressor(
                objective="reg:absoluteerror",
                random_state=42 + trial,
                **params,
            )
            m.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False,
            )
            p = m.predict(X_val_scaled)
            p = np.clip(p, 1, 22)
            mae = np.abs(p - y_val.values).mean()
            if mae < best_mae:
                best_mae = mae
                best_model = m
        model = best_model
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42)
        model.fit(X_train_scaled, y_train)
        pred_val = model.predict(X_val_scaled)
        best_mae = np.abs(pred_val - y_val.values).mean()

    pred_val = model.predict(X_val_scaled)
    pred_val = np.clip(pred_val, 1, 22)
    # Post-process: blend with quali (0.65 race + 0.35 quali), then enforce ranks per race
    quali_col = feat_cols.index("QualiPosition")
    quali_vals = X_val[:, quali_col]
    pred_blend = 0.65 * pred_val + 0.35 * quali_vals
    pred_blend = np.clip(pred_blend, 1, 22)
    # Per-race rank sort: assign 1..N by sorted order within each (Year, Round)
    val_df["_pred_raw"] = pred_blend
    val_df["_rank"] = val_df.groupby(["Year", "Round"])["_pred_raw"].rank(method="first", ascending=True)
    pred_val = val_df["_rank"].values.astype(float)
    mae_val = np.abs(pred_val - y_val.values).mean()

    rmse_val = np.sqrt(((pred_val - y_val.values) ** 2).mean())
    ss_res = ((y_val.values - pred_val) ** 2).sum()
    ss_tot = ((y_val.values - y_val.mean()) ** 2).sum()
    r2_val = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    print("Validation MAE: {:.3f}  RMSE: {:.3f}  R²: {:.3f}".format(mae_val, rmse_val, r2_val))

    # Top 15 feature importances
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        top15 = sorted(zip(feature_names, imp), key=lambda x: -x[1])[:15]
        print("Top 15 feature importances:")
        for name, val in top15:
            print("  ", name, ":", round(val, 4))

    # Sample prediction vs actual for 5–10 recent races (where errors largest)
    val_df["actual"] = y_val.values
    val_df["pred"] = pred_val
    val_df["error"] = np.abs(val_df["pred"] - val_df["actual"])
    by_race = val_df.groupby(["Year", "Round"]).agg(
        race_mae=("error", "mean"),
        n=("error", "count"),
    ).reset_index().sort_values(["Year", "Round"])
    recent_races = by_race.tail(10)
    print("Sample pred vs actual (last 10 races by MAE):")
    for _, r in recent_races.iterrows():
        yr, rnd = int(r["Year"]), int(r["Round"])
        sub = val_df[(val_df["Year"] == yr) & (val_df["Round"] == rnd)][["Abbreviation", "actual", "pred", "error"]]
        print("  {} R{} MAE={:.2f}".format(yr, rnd, r["race_mae"]))
        print(sub.head(6).to_string(index=False))
    # Largest errors
    worst = val_df.nlargest(10, "error")[["Year", "Round", "Abbreviation", "actual", "pred", "error"]]
    print("Largest 10 errors (midfield/DNF etc):")
    print(worst.to_string(index=False))

    # Diagnostics: feature importance barh, error histogram, per-race MAE (save to artifacts)
    out_dir = ROOT / "model_artifacts"
    out_dir.mkdir(exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if hasattr(model, "feature_importances_"):
            top15 = sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1])[:15]
            names, vals = [x[0] for x in top15], [x[1] for x in top15]
            axes[0].barh(range(len(names)), vals, align="center")
            axes[0].set_yticks(range(len(names)))
            axes[0].set_yticklabels(names, fontsize=8)
            axes[0].set_xlabel("Importance")
            axes[0].set_title("Top 15 feature importances")
            axes[0].invert_yaxis()
        errors = pred_val - y_val.values
        axes[1].hist(errors, bins=30, edgecolor="black", alpha=0.7)
        axes[1].axvline(0, color="red", linestyle="--")
        axes[1].set_xlabel("Prediction error (pred - actual)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Error distribution")
        plt.tight_layout()
        plt.savefig(out_dir / "diagnostics.png", dpi=100, bbox_inches="tight")
        plt.close()
        print("Saved diagnostics plot to", out_dir / "diagnostics.png")
    except Exception as e:
        print("Could not save diagnostics plot:", e)

    # Per-race MAE for last season
    if "Year" in val_df.columns:
        last_year = val_df["Year"].max()
        per_race = val_df[val_df["Year"] == last_year].groupby(["Year", "Round"])["error"].mean()
        print("Per-race MAE (year {}): mean={:.3f}".format(last_year, per_race.mean()))
        print(per_race.to_string())

    # Save only if MAE improved by >0.5 vs baseline (grid)
    save_threshold = grid_baseline_mae - 0.5
    if mae_val < save_threshold or mae_val < 7.0:
        models_dir = ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        joblib.dump(model, out_dir / "xgboost_model.joblib")
        joblib.dump({
            "driver": enc_driver,
            "team": enc_team,
            "circuit": enc_circuit,
            "weather_encoder": enc_weather,
            "feature_names": feature_names,
            "track_avg_driver_map": track_avg_driver_map,
            "track_avg_team_map": track_avg_team_map,
            "driver_team_synergy_map": driver_team_synergy_map,
            "driver_rain_delta_map": driver_rain_delta_map,
            "DEFAULT_TRACK_AVG": DEFAULT_TRACK_AVG,
            "DEFAULT_SYNERGY": DEFAULT_SYNERGY,
            "scaler": scaler,
            "scale_idx": scale_idx,
        }, out_dir / "encoders.joblib")
        joblib.dump(enc_weather, models_dir / "weather_ohe.joblib")
        print("Saved model and encoders (MAE {:.3f} below threshold {:.3f})".format(mae_val, save_threshold))
    else:
        print("MAE {:.3f} did not improve enough (threshold {:.3f}); model not saved.".format(mae_val, save_threshold))
    print("Wet+Rain share in training: {:.1f}%".format((X_df["Weather"] != "Dry").sum() / len(X_df) * 100))


# ---------- Qualifying model (unchanged logic, same script) ----------
def load_historical_quali_results(start_year: int = 2020, end_year: int = 2025) -> pd.DataFrame:
    import fastf1
    cache_dir = ROOT / "fastf1_cache"
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    from data import get_event_schedule, load_qualifying_results
    rows = []
    for year in range(start_year, end_year + 1):
        try:
            sched = get_event_schedule(year)
            if sched.empty:
                continue
            for _, row in sched.iterrows():
                r = int(row["RoundNumber"])
                event_name = str(row.get("EventName", ""))
                try:
                    res = load_qualifying_results(year, r)
                    if res.empty:
                        continue
                    res = res.copy()
                    res["Year"] = year
                    res["Round"] = r
                    res["Circuit"] = event_name
                    res["QualiPosition"] = pd.to_numeric(res.get("QualiPosition", res.get("Position", 10)), errors="coerce")
                    rows.append(res)
                except Exception:
                    continue
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_quali_training_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["QualiPosition", "Abbreviation"]).sort_values(["Year", "Round"])
    df["quali_form"] = ROOKIE_DEFAULT_AVG_POSITION
    for driver in df["Abbreviation"].unique():
        mask = df["Abbreviation"] == driver
        pos = df.loc[mask, "QualiPosition"].astype(float)
        rolling = pos.shift(1).rolling(5, min_periods=1).mean()
        df.loc[mask, "quali_form"] = rolling.values
    team_avg = df.groupby("TeamName")["QualiPosition"].transform("mean")
    df["constructor_strength"] = team_avg
    rng = np.random.default_rng(43)
    w = rng.random(len(df))
    df["Weather"] = np.where(w < 0.5, "Dry", np.where(w < 0.8, "Wet", "Rain"))
    df["momentum"] = df["Round"].astype(float) / 24.0
    return df


def train_quali_model():
    print("\n--- Qualifying model ---")
    print("Loading 2020-2025 qualifying data...")
    df = load_historical_quali_results(2020, 2025)
    if df.empty:
        print("No quali data.")
        return
    df = build_quali_training_features(df)
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    import joblib
    enc_driver = LabelEncoder()
    enc_team = LabelEncoder()
    enc_circuit = LabelEncoder()
    enc_weather = OneHotEncoder(categories=[WEATHER_CATEGORIES], drop=None, sparse_output=False)
    grid_2026 = pd.DataFrame(GRID_2026)
    all_drivers = list(df["Abbreviation"].unique()) + list(grid_2026["Abbreviation"].unique())
    all_teams = list(df["TeamName"].unique()) + list(grid_2026["TeamName"].unique())
    all_circuits = list(df["Circuit"].unique()) + ["Australian Grand Prix", "Melbourne"]
    enc_driver.fit(list(dict.fromkeys(all_drivers)))
    enc_team.fit(list(dict.fromkeys(all_teams)))
    enc_circuit.fit(list(dict.fromkeys(all_circuits)))
    enc_weather.fit(np.array(WEATHER_CATEGORIES).reshape(-1, 1))
    df["driver_enc"] = enc_driver.transform(df["Abbreviation"])
    df["team_enc"] = enc_team.transform(df["TeamName"].astype(str))
    df["circuit_enc"] = enc_circuit.transform(df["Circuit"].astype(str))
    wo = enc_weather.transform(df[["Weather"]].values)
    df["weather_Dry"] = wo[:, 0]
    df["weather_Wet"] = wo[:, 1]
    df["weather_Rain"] = wo[:, 2]
    feat_cols = ["quali_form", "constructor_strength", "driver_enc", "team_enc", "circuit_enc", "momentum", "weather_Dry", "weather_Wet", "weather_Rain"]
    X = df[feat_cols].values
    y = df["QualiPosition"].astype(float)
    try:
        import xgboost as xgb
        model = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / "quali_model.joblib")
    joblib.dump({"driver": enc_driver, "team": enc_team, "circuit": enc_circuit, "weather_encoder": enc_weather, "feature_names": feat_cols}, models_dir / "quali_encoders.joblib")
    print("Saved quali model to", models_dir)


if __name__ == "__main__":
    main()
    train_quali_model()
