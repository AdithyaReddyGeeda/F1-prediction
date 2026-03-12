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
    """Oversample Wet and Rain rows so model sees enough weather signal."""
    if rng is None:
        rng = np.random.default_rng(42)
    if "Weather" not in df.columns or len(df) == 0:
        return df
    wet_mask = df["Weather"] == "Wet"
    rain_mask = df["Weather"] == "Rain"
    pct_wet_rain = (wet_mask | rain_mask).sum() / len(df)
    if pct_wet_rain >= target_min_pct:
        return df
    extra = []
    for _, row in df[wet_mask | rain_mask].iterrows():
        r = row.to_dict()
        for k in ["RecentForm", "GridPosition", "ConstructorEwma", "track_avg_driver", "track_avg_team", "driver_team_synergy", "QualiPosition"]:
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
    wo = enc_weather.transform(X_df[["Weather"]])
    X_df["weather_Dry"] = wo[:, 0]
    X_df["weather_Wet"] = wo[:, 1]
    X_df["weather_Rain"] = wo[:, 2]

    # Feature columns (order must match inference)
    feat_cols = [
        "GridPosition", "QualiPosition", "RecentForm", "ConstructorEwma",
        "track_avg_driver", "track_avg_team", "driver_team_synergy", "momentum", "driver_rain_delta",
        "driver_enc", "team_enc", "circuit_enc",
        "circuit_type_street", "circuit_type_high_speed", "circuit_type_technical",
        "weather_Dry", "weather_Wet", "weather_Rain", "grid_pos_x_rain",
    ]
    feature_names = list(feat_cols)
    # Ensure all present
    for c in feat_cols:
        if c not in X_df.columns:
            X_df[c] = 0.0 if "enc" in c or "weather" in c or "circuit_type" in c else 10.0
    X_full = X_df[feat_cols].values

    # Time-series split: train 2020-2023, validate 2024
    train_mask = (X_df["Year"] >= 2020) & (X_df["Year"] <= 2023)
    val_mask = X_df["Year"] == 2024
    X_train, y_train = X_full[train_mask], y[train_mask]
    X_val, y_val = X_full[val_mask], y[val_mask]
    if X_val.size == 0:
        # Fallback: use 2025 as val if 2024 missing
        val_mask = X_df["Year"] == 2025
        X_val, y_val = X_full[val_mask], y[val_mask]
    if X_val.size == 0:
        X_val, y_val = X_train[-500:], y_train[-500:]  # last 500 as val

    print("Train size:", len(X_train), "Val size:", len(X_val))

    # Build lookup maps for inference (from training set only)
    train_df = X_df[train_mask]
    track_avg_driver_map = train_df.groupby(["Abbreviation", "Circuit"])["track_avg_driver"].mean().to_dict()
    track_avg_team_map = train_df.groupby(["TeamName", "Circuit"])["track_avg_team"].mean().to_dict()
    driver_team_synergy_map = train_df.groupby(["Abbreviation", "TeamName"])["driver_team_synergy"].mean().to_dict()
    driver_rain_delta_map = train_df.groupby("Abbreviation")["driver_rain_delta"].mean().to_dict()

    try:
        import xgboost as xgb
        # MAE objective
        model = xgb.XGBRegressor(
            objective="reg:absoluteerror",
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            min_child_weight=3,
            random_state=42,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        # Optional: RandomizedSearchCV (commented out for speed; uncomment for tuning)
        # from sklearn.model_selection import RandomizedSearchCV
        # param_dist = {"n_estimators": [300, 500, 700], "max_depth": [4, 6, 8], "learning_rate": [0.03, 0.05, 0.1],
        #              "subsample": [0.7, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "reg_alpha": [0.1, 0.5, 2], "reg_lambda": [0.5, 1, 3]}
        # search = RandomizedSearchCV(model, param_dist, n_iter=30, cv=3, scoring="neg_mean_absolute_error", random_state=42)
        # search.fit(X_train, y_train)
        # model = search.best_estimator_
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42,
        )
        model.fit(X_train, y_train)

    # Validation MAE
    pred_val = model.predict(X_val)
    pred_val = np.clip(pred_val, 1, 22)
    mae_val = np.abs(pred_val - y_val.values).mean()
    print("Validation MAE (after training): {:.3f}".format(mae_val))

    # Feature selection: drop very low importance
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        keep = [i for i, v in enumerate(imp) if v >= 0.01]
        if len(keep) < len(feat_cols):
            feat_cols = [feat_cols[i] for i in keep]
            feature_names = list(feat_cols)
            X_full = X_df[feat_cols].values
            # Retrain with selected features (optional; comment out to keep full model)
            # X_train, X_val = X_full[train_mask], X_full[val_mask]
            # model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            # pred_val = model.predict(X_val)
            # mae_val = np.abs(pred_val - y_val.values).mean()
            # print("Validation MAE (after feature selection): {:.3f}".format(mae_val))
        top10 = sorted(zip(feature_names, imp), key=lambda x: -x[1])[:10]
        print("Top 10 feature importances:")
        for name, val in top10:
            print("  ", name, ":", round(val, 4))

    # Sample predictions vs actuals
    sample = pd.DataFrame({"actual": y_val.values[:15], "pred": pred_val[:15]})
    sample["pred"] = sample["pred"].clip(1, 22)
    print("Sample (first 15 val): actual vs pred")
    print(sample.to_string())

    out_dir = ROOT / "model_artifacts"
    out_dir.mkdir(exist_ok=True)
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
    }, out_dir / "encoders.joblib")
    joblib.dump(enc_weather, models_dir / "weather_ohe.joblib")
    print("Saved model and encoders to", out_dir)
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
    wo = enc_weather.transform(df[["Weather"]])
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
