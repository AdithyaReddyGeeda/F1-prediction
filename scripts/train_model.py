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

from config import GRID_2026, ROOKIE_DEFAULT_AVG_POSITION, ENGINE_BY_TEAM
from utils.race_features import (
    build_race_feature_df,
    DEFAULT_TRACK_AVG,
    DEFAULT_SYNERGY,
    _circuit_to_type,
)

WEATHER_CATEGORIES = ["Dry", "Wet", "Rain"]


def load_historical_results(start_year: int = 2020, end_year: int = 2025) -> pd.DataFrame:
    """Load all race results from FastF1 (with Status when available for DNF)."""
    import fastf1
    cache_dir = ROOT / "fastf1_cache"
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    from data import get_event_schedule, load_race_results, load_race_weather, load_race_tyre_proxy
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


def load_historical_fp_deltas(start_year: int = 2020, end_year: int = 2025) -> pd.DataFrame:
    """Load FP1/FP2/FP3 deltas for all races. Returns merged DataFrame with Year, Round, Abbreviation, FP1_delta, FP2_delta, FP3_delta."""
    from data import get_event_schedule, load_fp_deltas
    import fastf1
    cache_dir = ROOT / "fastf1_cache"
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    rows = []
    for year in range(start_year, end_year + 1):
        try:
            sched = get_event_schedule(year)
            if sched.empty:
                continue
            for _, row in sched.iterrows():
                r = int(row["RoundNumber"])
                try:
                    fp = load_fp_deltas(year, r)
                    if fp.empty:
                        continue
                    rows.append(fp)
                except Exception:
                    continue
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_historical_tyre_proxy(start_year: int = 2020, end_year: int = 2025) -> pd.DataFrame:
    """Load stint-based tyre proxy (avg laps per stint) for all races. Empty DataFrame on failure."""
    from data import get_event_schedule, load_race_tyre_proxy
    rows = []
    for year in range(start_year, end_year + 1):
        try:
            sched = get_event_schedule(year)
            if sched.empty:
                continue
            for _, row in sched.iterrows():
                r = int(row["RoundNumber"])
                try:
                    tp = load_race_tyre_proxy(year, r)
                    if not tp.empty:
                        rows.append(tp)
                except Exception:
                    continue
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_historical_quali_gaps(start_year: int = 2020, end_year: int = 2025) -> pd.DataFrame:
    """Load quali gap to pole (seconds) for all races. Returns Year, Round, Abbreviation, quali_gap_to_pole."""
    from data import get_event_schedule, load_quali_gaps
    import fastf1
    cache_dir = ROOT / "fastf1_cache"
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    rows = []
    for year in range(start_year, end_year + 1):
        try:
            sched = get_event_schedule(year)
            if sched.empty:
                continue
            for _, row in sched.iterrows():
                r = int(row["RoundNumber"])
                try:
                    qg = load_quali_gaps(year, r)
                    if not qg.empty:
                        rows.append(qg)
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
    import argparse
    import joblib
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder

    parser = argparse.ArgumentParser(description="Train F1 race prediction model (XGBoost + optional LightGBM).")
    parser.add_argument("--start", type=int, default=2020, help="First season year to load (default: 2020). Use e.g. 2023 to reduce API calls.")
    parser.add_argument("--end", type=int, default=2025, help="Last season year to load (default: 2025).")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from latest checkpoint if present (default: True).")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Ignore checkpoints and train from scratch.")
    parser.add_argument("--keep-checkpoints", action="store_true", help="Keep checkpoint files after successful training (default: delete them).")
    args = parser.parse_args()
    start_year = args.start
    end_year = args.end
    if start_year > end_year:
        start_year, end_year = end_year, start_year
    resume = args.resume
    keep_checkpoints = args.keep_checkpoints
    ckpt_dir = ROOT / "model_artifacts" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_1 = ckpt_dir / "checkpoint_1_data.joblib"
    ckpt_3 = ckpt_dir / "checkpoint_3_features.joblib"
    ckpt_4 = ckpt_dir / "checkpoint_4_model.joblib"

    # --- Phase 1: Load data (or resume from checkpoint 1) ---
    if resume and ckpt_1.exists():
        print("Resuming: loading checkpoint 1 (data)...")
        c1 = joblib.load(ckpt_1)
        race_df = c1["race_df"]
        quali_df = c1.get("quali_df")
        fp_df = c1.get("fp_df")
        tyre_proxy_df = c1.get("tyre_proxy_df")
        quali_gap_df = c1.get("quali_gap_df")
        weather_per_race = c1["weather_per_race"]
        start_year = c1["start_year"]
        end_year = c1["end_year"]
        print("Resumed from checkpoint 1 (seasons {}-{}). Skipping data load.".format(start_year, end_year))
    else:
        print("Training on seasons {}-{} (FastF1 limit 500 calls/h; fewer years = fewer calls).".format(start_year, end_year))
        print("Loading {}-{} race data from FastF1...".format(start_year, end_year))
        race_df = load_historical_results(start_year, end_year)
        if race_df.empty:
            print("No race data. Run app once to populate FastF1 cache, or use --start/--end to pick years.")
            return

        print("Loading qualifying data for quali-position feature...")
        quali_df = load_historical_quali_for_merge(start_year, end_year)
        if quali_df.empty:
            quali_df = None

        print("Loading FP deltas (FP1/FP2/FP3) for practice signals...")
        fp_df = load_historical_fp_deltas(start_year, end_year)
        if fp_df.empty:
            fp_df = None

        print("Loading stint-based tyre proxy (optional)...")
        tyre_proxy_df = load_historical_tyre_proxy(start_year, end_year)
        if tyre_proxy_df.empty:
            tyre_proxy_df = None

        print("Loading quali gaps to pole (optional)...")
        quali_gap_df = load_historical_quali_gaps(start_year, end_year)
        if quali_gap_df.empty:
            quali_gap_df = None

        rng = np.random.default_rng(42)
        keys = race_df[["Year", "Round"]].drop_duplicates()
        weather_per_race = {}
        real_count = 0
        for _, row in keys.iterrows():
            y, r = int(row["Year"]), int(row["Round"])
            real = load_race_weather(y, r)
            if real is not None:
                weather_per_race[(y, r)] = real
                real_count += 1
            else:
                weather_per_race[(y, r)] = rng.choice(["Dry", "Wet", "Rain"], p=[0.5, 0.3, 0.2])
        print("Weather: using real FastF1 for {}/{} races (rest random)".format(real_count, len(keys)))
        joblib.dump({
            "race_df": race_df, "quali_df": quali_df, "fp_df": fp_df, "tyre_proxy_df": tyre_proxy_df,
            "quali_gap_df": quali_gap_df, "weather_per_race": weather_per_race, "start_year": start_year, "end_year": end_year,
        }, ckpt_1)
        print("Checkpoint 1 saved (data load complete).")
    rng = np.random.default_rng(42)

    # --- Phase 2 & 3: Alpha search + feature build ---
    print("EWMA alpha grid search (0.3, 0.4, 0.5, 0.6)...")
    best_alpha, best_alpha_mae = 0.4, np.inf
    for alpha in [0.3, 0.4, 0.5, 0.6]:
        X_alpha = build_race_feature_df(
                race_df,
                quali_df=quali_df,
                weather_per_race=weather_per_race,
                fp_df=fp_df,
                tyre_proxy_df=tyre_proxy_df,
                ewma_alpha=alpha,
        )
        X_alpha = oversample_wet_rain(X_alpha, target_min_pct=0.25, rng=rng)
        if len(X_alpha) < 100:
            continue
        # Quick encode (reuse logic below but minimal)
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        enc_d = LabelEncoder()
        enc_t = LabelEncoder()
        enc_c = LabelEncoder()
        enc_w = OneHotEncoder(categories=[WEATHER_CATEGORIES], drop=None, sparse_output=False)
        grid_2026 = pd.DataFrame(GRID_2026)
        enc_d.fit(list(X_alpha["Abbreviation"].unique()) + list(grid_2026["Abbreviation"].unique()))
        enc_t.fit(list(X_alpha["TeamName"].unique()) + list(grid_2026["TeamName"].unique()))
        enc_e = LabelEncoder()
        enc_e.fit(list(X_alpha["engine_supplier"].unique()) + [ENGINE_BY_TEAM.get(t, "Other") for t in grid_2026["TeamName"].unique()])
        enc_c.fit(list(X_alpha["Circuit"].unique()) + ["Australian Grand Prix"])
        enc_w.fit(np.array(WEATHER_CATEGORIES).reshape(-1, 1))
        for c in ["driver_enc", "team_enc", "engine_enc", "circuit_enc", "weather_Dry", "weather_Wet", "weather_Rain",
                      "FP1_delta", "FP2_delta", "FP3_delta", "circuit_abrasion_proxy", "tyre_life_penalty_proxy",
                      "driver_dnf_rate", "driver_tyre_management_proxy"]:
                if c not in X_alpha.columns and "enc" in c:
                    continue
                if c == "driver_enc":
                    X_alpha["driver_enc"] = enc_d.transform(X_alpha["Abbreviation"].astype(str))
                elif c == "team_enc":
                    X_alpha["team_enc"] = enc_t.transform(X_alpha["TeamName"].astype(str))
                elif c == "engine_enc":
                    X_alpha["engine_enc"] = enc_e.transform(X_alpha["engine_supplier"].astype(str))
                elif c == "circuit_enc":
                    X_alpha["circuit_enc"] = enc_c.transform(X_alpha["Circuit"].astype(str))
                elif c == "weather_Dry":
                    wo = enc_w.transform(X_alpha[["Weather"]].values)
                    X_alpha["weather_Dry"], X_alpha["weather_Wet"], X_alpha["weather_Rain"] = wo[:, 0], wo[:, 1], wo[:, 2]
        feat_cols_alpha = [
            "GridPosition", "QualiPosition", "quali_gap_to_pole", "RecentForm", "ConstructorEwma",
            "track_avg_driver", "track_avg_team", "driver_team_synergy", "teammate_delta", "constructor_dnf_rate",
            "constructor_dnf_rate_at_circuit", "driver_dnf_rate", "circuit_abrasion_proxy", "tyre_life_penalty_proxy", "driver_tyre_management_proxy",
            "form_x_teammate_delta", "momentum", "driver_rain_delta",
            "FP1_delta", "FP2_delta", "FP3_delta",
            "driver_enc", "team_enc", "engine_enc", "circuit_enc",
            "circuit_type_street", "circuit_type_high_speed", "circuit_type_technical",
            "weather_Dry", "weather_Wet", "weather_Rain", "grid_pos_x_rain",
        ]
        if "quali_gap_to_pole" not in X_alpha.columns:
            X_alpha["quali_gap_to_pole"] = 2.0
        for c in feat_cols_alpha:
            if c not in X_alpha.columns:
                if c == "constructor_dnf_rate_at_circuit":
                    X_alpha[c] = 0.0
                else:
                    X_alpha[c] = 0.0 if "enc" in c or "weather" in c or "circuit_type" in c else (10.0 if "track" in c or "synergy" in c or "Constructor" in c or "Quali" in c or "Grid" in c else 0.0)
        X_mat = X_alpha[feat_cols_alpha].values
        train_m = (X_alpha["Year"] >= start_year) & (X_alpha["Year"] < end_year)
        val_m = X_alpha["Year"] == end_year
        if val_m.sum() == 0:
            val_m = X_alpha["Year"] == end_year - 1
        if val_m.sum() == 0:
            continue
        from sklearn.preprocessing import StandardScaler
        scale_idx_a = [i for i, n in enumerate(feat_cols_alpha) if n not in ("driver_enc", "team_enc", "engine_enc", "circuit_enc") and not n.startswith("weather_") and not n.startswith("circuit_type_")]
        scaler_a = StandardScaler()
        X_t = X_mat[train_m].copy()
        X_v = X_mat[val_m].copy()
        X_t[:, scale_idx_a] = scaler_a.fit_transform(X_t[:, scale_idx_a])
        X_v[:, scale_idx_a] = scaler_a.transform(X_v[:, scale_idx_a])
        y_t = X_alpha.loc[train_m, "Position"].astype(float)
        y_v = X_alpha.loc[val_m, "Position"].astype(float)
        try:
            import xgboost as xgb
            m = xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=80, max_depth=5, random_state=42)
            m.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
            p = np.clip(m.predict(X_v), 1, 22)
            mae_a = np.abs(p - y_v.values).mean()
            if mae_a < best_alpha_mae:
                best_alpha_mae = mae_a
                best_alpha = alpha
        except Exception:
            pass
    print("Best EWMA alpha: {:.1f} (val MAE: {:.3f})".format(best_alpha, best_alpha_mae))

    print("Building race features (EWMA alpha={}, track, synergy, tyre, practice, engine)...".format(best_alpha))
    X_df = build_race_feature_df(
        race_df,
        quali_df=quali_df,
        weather_per_race=weather_per_race,
        fp_df=fp_df,
        tyre_proxy_df=tyre_proxy_df,
        ewma_alpha=best_alpha,
    )
    if quali_gap_df is not None and not quali_gap_df.empty:
        X_df = X_df.merge(
            quali_gap_df[["Year", "Round", "Abbreviation", "quali_gap_to_pole"]].drop_duplicates(
                subset=["Year", "Round", "Abbreviation"], keep="last"
            ),
            on=["Year", "Round", "Abbreviation"],
            how="left",
        )
        X_df["quali_gap_to_pole"] = X_df["quali_gap_to_pole"].fillna(2.0)
    else:
        X_df["quali_gap_to_pole"] = 2.0
    X_df = oversample_wet_rain(X_df, target_min_pct=0.25, rng=rng)
    y = X_df["Position"].astype(float)

    # Encoders
    grid_2026 = pd.DataFrame(GRID_2026)
    all_drivers = list(X_df["Abbreviation"].unique()) + list(grid_2026["Abbreviation"].unique())
    all_teams = list(X_df["TeamName"].unique()) + list(grid_2026["TeamName"].unique())
    all_circuits = list(X_df["Circuit"].unique()) + ["Australian Grand Prix", "Melbourne"]
    all_engines = list(X_df["engine_supplier"].unique()) + [ENGINE_BY_TEAM.get(t, "Other") for t in grid_2026["TeamName"].unique()]
    enc_driver = LabelEncoder()
    enc_team = LabelEncoder()
    enc_engine = LabelEncoder()
    enc_circuit = LabelEncoder()
    enc_weather = OneHotEncoder(categories=[WEATHER_CATEGORIES], drop=None, sparse_output=False)
    enc_driver.fit(list(dict.fromkeys(all_drivers)))
    enc_team.fit(list(dict.fromkeys(all_teams)))
    enc_engine.fit(list(dict.fromkeys(all_engines)))
    enc_circuit.fit(list(dict.fromkeys(all_circuits)))
    enc_weather.fit(np.array(WEATHER_CATEGORIES).reshape(-1, 1))
    
    X_df["driver_enc"] = enc_driver.transform(X_df["Abbreviation"].astype(str))
    X_df["team_enc"] = enc_team.transform(X_df["TeamName"].astype(str))
    X_df["engine_enc"] = enc_engine.transform(X_df["engine_supplier"].astype(str))
    X_df["circuit_enc"] = enc_circuit.transform(X_df["Circuit"].astype(str))
    wo = enc_weather.transform(X_df[["Weather"]].values)
    X_df["weather_Dry"] = wo[:, 0]
    X_df["weather_Wet"] = wo[:, 1]
    X_df["weather_Rain"] = wo[:, 2]
    
    # Feature columns (order must match inference); include engine_enc, tyre, driver_dnf, practice deltas, quali_gap, circuit DNF
    feat_cols = [
    "GridPosition", "QualiPosition", "quali_gap_to_pole", "RecentForm", "ConstructorEwma",
    "track_avg_driver", "track_avg_team", "driver_team_synergy", "teammate_delta", "constructor_dnf_rate",
    "constructor_dnf_rate_at_circuit", "driver_dnf_rate", "circuit_abrasion_proxy", "tyre_life_penalty_proxy", "driver_tyre_management_proxy",
    "form_x_teammate_delta", "momentum", "driver_rain_delta",
        "FP1_delta", "FP2_delta", "FP3_delta",
        "driver_enc", "team_enc", "engine_enc", "circuit_enc",
        "circuit_type_street", "circuit_type_high_speed", "circuit_type_technical",
        "weather_Dry", "weather_Wet", "weather_Rain", "grid_pos_x_rain",
    ]
    feature_names = list(feat_cols)
    for c in feat_cols:
        if c not in X_df.columns:
            if c == "quali_gap_to_pole":
                X_df[c] = 2.0
            elif c == "constructor_dnf_rate_at_circuit":
                X_df[c] = 0.0
            else:
                X_df[c] = 0.0 if "enc" in c or "weather" in c or "circuit_type" in c else (10.0 if "track" in c or "synergy" in c or "Constructor" in c or "Quali" in c or "Grid" in c else 0.0)
    X_full = X_df[feat_cols].values
    
    # Time-series split: train on start_year..end_year-1, validate on end_year (walk-forward)
    train_mask = (X_df["Year"] >= start_year) & (X_df["Year"] < end_year)
    val_mask = X_df["Year"] == end_year
    if val_mask.sum() == 0:
        val_mask = X_df["Year"] == end_year - 1
    if val_mask.sum() == 0:
        val_mask = pd.Series(False, index=X_df.index)
        val_mask.iloc[-500:] = True
    X_train, y_train = X_full[train_mask], y[train_mask]
    X_val, y_val = X_full[val_mask], y[val_mask]
    val_df = X_df[val_mask].copy()
    
    print("Train size:", len(X_train), "Val size:", len(X_val))
    
    # StandardScaler on numeric features (same order as feat_cols)
    from sklearn.preprocessing import StandardScaler
    scale_idx = [i for i, n in enumerate(feat_cols) if n not in ("driver_enc", "team_enc", "engine_enc", "circuit_enc") and not n.startswith("weather_") and not n.startswith("circuit_type_")]
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
    circuit_dnf_rate_map = train_df.groupby(["TeamName", "Circuit"])["constructor_dnf_rate_at_circuit"].mean().to_dict()
    
    joblib.dump({
        "X_df": X_df, "y": y, "enc_driver": enc_driver, "enc_team": enc_team, "enc_engine": enc_engine,
        "enc_circuit": enc_circuit, "enc_weather": enc_weather, "feat_cols": feat_cols, "feature_names": feature_names,
        "scale_idx": scale_idx, "scaler": scaler, "train_mask": train_mask, "val_mask": val_mask,
        "X_train_scaled": X_train_scaled, "X_val_scaled": X_val_scaled, "y_train": y_train, "y_val": y_val,
        "val_df": val_df, "track_avg_driver_map": track_avg_driver_map, "track_avg_team_map": track_avg_team_map,
        "driver_team_synergy_map": driver_team_synergy_map, "driver_rain_delta_map": driver_rain_delta_map,
        "circuit_dnf_rate_map": circuit_dnf_rate_map, "grid_baseline_mae": grid_baseline_mae, "best_alpha": best_alpha,
        "start_year": start_year, "end_year": end_year,
    }, ckpt_3)
    print("Checkpoint 3 saved (features and splits).")

    # --- Phase 4: XGBoost hyperparameter search (or resume from checkpoint 4) ---
    skip_phase_4 = False
    if resume and ckpt_4.exists() and ckpt_3.exists():
        print("Resuming: loading checkpoint 4 (XGB model) and checkpoint 3...")
        model = joblib.load(ckpt_4)
        c3 = joblib.load(ckpt_3)
        X_df = c3["X_df"]
        y = c3["y"]
        enc_driver, enc_team = c3["enc_driver"], c3["enc_team"]
        enc_engine, enc_circuit, enc_weather = c3["enc_engine"], c3["enc_circuit"], c3["enc_weather"]
        feat_cols, feature_names = c3["feat_cols"], c3["feature_names"]
        scale_idx, scaler = c3["scale_idx"], c3["scaler"]
        train_mask, val_mask = c3["train_mask"], c3["val_mask"]
        X_train_scaled = c3["X_train_scaled"]
        X_val_scaled = c3["X_val_scaled"]
        y_train, y_val = c3["y_train"], c3["y_val"]
        val_df = c3["val_df"]
        track_avg_driver_map = c3["track_avg_driver_map"]
        track_avg_team_map = c3["track_avg_team_map"]
        driver_team_synergy_map = c3["driver_team_synergy_map"]
        driver_rain_delta_map = c3["driver_rain_delta_map"]
        circuit_dnf_rate_map = c3["circuit_dnf_rate_map"]
        grid_baseline_mae = c3["grid_baseline_mae"]
        best_alpha = c3["best_alpha"]
        skip_phase_4 = True
        pred_val = model.predict(X_val_scaled)
        pred_val = np.clip(pred_val, 1, 22)
        mae_before_blend = np.abs(pred_val - y_val.values).mean()
        print("Resumed from checkpoint 4. Skipping XGB search.")

    if not skip_phase_4:
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
        mae_before_blend = np.abs(pred_val - y_val.values).mean()
        joblib.dump(model, ckpt_4)
        print("Checkpoint 4 saved (XGB model).")
    # ---- Optimise blend ratio per circuit type (street / high_speed / technical) ----
    quali_col = feat_cols.index("QualiPosition")
    quali_vals = X_val[:, quali_col]
    raw_preds = np.clip(pred_val, 1, 22)
    val_df["circuit_type"] = val_df["Circuit"].map(_circuit_to_type)
    best_blend_by_type: dict[str, float] = {}
    base_blend_ratio = 0.7
    for ctype in ["street", "high_speed", "technical"]:
        type_mask = val_df["circuit_type"].values == ctype
        if type_mask.sum() < 20:
            best_blend_by_type[ctype] = 0.65
            continue
        best_r, best_m = 0.65, np.inf
        for r in np.arange(0.3, 0.85, 0.05):
            blended = r * raw_preds[type_mask] + (1.0 - r) * quali_vals[type_mask]
            mae_r = np.abs(blended - y_val.values[type_mask]).mean()
            if mae_r < best_m:
                best_m = mae_r
                best_r = float(r)
        best_blend_by_type[ctype] = float(best_r)
        print(f"Best blend for {ctype}: {best_r:.2f} (MAE {best_m:.3f})")
    print("Blend ratios by circuit type:", best_blend_by_type)

    # Apply type-specific blend to build validation prediction used for metrics
    pred_blend = np.empty_like(raw_preds)
    for ctype, r in best_blend_by_type.items():
        m = val_df["circuit_type"].values == ctype
        if m.any():
            pred_blend[m] = np.clip(r * raw_preds[m] + (1.0 - r) * quali_vals[m], 1, 22)
    # Any leftover rows (e.g. unknown type) use base blend
    leftover = ~np.isfinite(pred_blend)
    if leftover.any():
        pred_blend[leftover] = np.clip(
            base_blend_ratio * raw_preds[leftover] + (1.0 - base_blend_ratio) * quali_vals[leftover],
            1,
            22,
        )

    val_df["_pred_raw"] = pred_blend
    val_df["_rank"] = val_df.groupby(["Year", "Round"])["_pred_raw"].rank(method="first", ascending=True)
    pred_val = val_df["_rank"].values.astype(float)
    mae_val = np.abs(pred_val - y_val.values).mean()
    print("MAE before blend: {:.3f}  |  after blend + rank: {:.3f}".format(mae_before_blend, mae_val))

    # Optional: LightGBM LambdaRank ensemble (saves LGB if ensemble MAE is >0.1 better than XGB blend)
    lgb_ensemble = False
    try:
        import lightgbm as lgb

        def _make_groups(df_sub):
            return df_sub.groupby(["Year", "Round"]).size().values.astype(np.int32)

        train_groups = _make_groups(X_df[train_mask])
        val_groups = _make_groups(X_df[val_mask])

        lgb_train = lgb.Dataset(
            X_train_scaled, label=y_train.values,
            group=train_groups, free_raw_data=False,
        )
        lgb_val = lgb.Dataset(
            X_val_scaled, label=y_val.values,
            group=val_groups, reference=lgb_train, free_raw_data=False,
        )
        lgb_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5, 10],
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 10,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }
        lgb_model = lgb.train(
            lgb_params, lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        lgb_preds = lgb_model.predict(X_val_scaled)
        ensemble_raw = 0.6 * raw_preds + 0.4 * lgb_preds
        ensemble_raw = np.clip(ensemble_raw, 1, 22)
        val_df["_pred_raw"] = ensemble_raw
        val_df["_rank"] = val_df.groupby(["Year", "Round"])["_pred_raw"].rank(method="first", ascending=True)
        ensemble_mae = (np.abs(val_df["_rank"].values.astype(float) - y_val.values)).mean()
        print("Ensemble MAE (XGB+LGB): {:.3f}  vs XGB blend only: {:.3f}".format(ensemble_mae, mae_val))
        if ensemble_mae < mae_val - 0.1:
            print("Ensemble is better — saving LightGBM model too")
            joblib.dump(lgb_model, ROOT / "model_artifacts" / "lgb_model.joblib")
            lgb_ensemble = True
            mae_val = ensemble_mae
            pred_val = val_df["_rank"].values.astype(float)
        else:
            # Restore blend-based prediction for rest of script
            val_df["_pred_raw"] = pred_blend
            val_df["_rank"] = val_df.groupby(["Year", "Round"])["_pred_raw"].rank(method="first", ascending=True)
            pred_val = val_df["_rank"].values.astype(float)
    except ImportError:
        print("lightgbm not installed — skipping. Run: pip install lightgbm")
    except Exception as e:
        print("LightGBM ensemble failed:", e)

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

    # Stratified MAE: top-5 vs midfield vs back, dry vs wet
    top5 = val_df[val_df["actual"] <= 5]
    midfield = val_df[(val_df["actual"] >= 6) & (val_df["actual"] <= 15)]
    back = val_df[val_df["actual"] >= 16]
    dry_races = val_df[val_df["Weather"] == "Dry"]
    wet_races = val_df[val_df["Weather"].isin(["Wet", "Rain"])]
    print("Stratified MAE:")
    if len(top5) > 0:
        print("  Top-5 (actual 1–5): {:.3f}".format(top5["error"].mean()))
    if len(midfield) > 0:
        print("  Midfield (6–15): {:.3f}".format(midfield["error"].mean()))
    if len(back) > 0:
        print("  Back (16–22): {:.3f}".format(back["error"].mean()))
    if len(dry_races) > 0:
        print("  Dry races: {:.3f}".format(dry_races["error"].mean()))
    if len(wet_races) > 0:
        print("  Wet/rain races: {:.3f}".format(wet_races["error"].mean()))

    # Sample pred vs actual for recent 2025/2026 races
    recent_val = val_df[val_df["Year"].isin([2025, 2026])]
    if not recent_val.empty:
        print("Sample pred vs actual (2025/2026 races):")
        for (yr, rnd), g in recent_val.groupby(["Year", "Round"]):
            print("  {} R{}".format(int(yr), int(rnd)))
            print(g[["Abbreviation", "actual", "pred", "error"]].head(8).to_string(index=False))

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
            "engine": enc_engine,
            "circuit": enc_circuit,
            "weather_encoder": enc_weather,
            "feature_names": feature_names,
            "track_avg_driver_map": track_avg_driver_map,
            "track_avg_team_map": track_avg_team_map,
            "driver_team_synergy_map": driver_team_synergy_map,
            "driver_rain_delta_map": driver_rain_delta_map,
            "circuit_dnf_rate_map": circuit_dnf_rate_map,
            "DEFAULT_TRACK_AVG": DEFAULT_TRACK_AVG,
            "DEFAULT_SYNERGY": DEFAULT_SYNERGY,
            "scaler": scaler,
            "scale_idx": scale_idx,
            # Inference-time knobs (kept in sync with training)
            "ewma_alpha": float(best_alpha),
            "blend_ratio": base_blend_ratio,
            "blend_by_type": best_blend_by_type,
            "lgb_ensemble": lgb_ensemble,
        }, out_dir / "encoders.joblib")
        joblib.dump(enc_weather, models_dir / "weather_ohe.joblib")
        print("Saved model and encoders (MAE {:.3f} below threshold {:.3f})".format(mae_val, save_threshold))
        if not keep_checkpoints and ckpt_dir.exists():
            for f in (ckpt_1, ckpt_3, ckpt_4):
                if f.exists():
                    f.unlink()
                    print("Removed checkpoint:", f.name)
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


def build_quali_training_features(df: pd.DataFrame, weather_per_race: dict | None = None) -> pd.DataFrame:
    df = df.dropna(subset=["QualiPosition", "Abbreviation"]).sort_values(["Year", "Round"])
    df["quali_form"] = ROOKIE_DEFAULT_AVG_POSITION
    for driver in df["Abbreviation"].unique():
        mask = df["Abbreviation"] == driver
        pos = df.loc[mask, "QualiPosition"].astype(float)
        rolling = pos.shift(1).rolling(5, min_periods=1).mean()
        df.loc[mask, "quali_form"] = rolling.values
    team_avg = df.groupby("TeamName")["QualiPosition"].transform("mean")
    df["constructor_strength"] = team_avg
    if weather_per_race is not None:
        df["Weather"] = df.apply(
            lambda r: weather_per_race.get((int(r["Year"]), int(r["Round"])), "Dry"),
            axis=1,
        )
    else:
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
    # Real weather per race when available, fallback to random distribution
    from data import load_race_weather
    rng = np.random.default_rng(99)
    keys = df[["Year", "Round"]].drop_duplicates()
    weather_per_race: dict[tuple[int, int], str] = {}
    for _, row in keys.iterrows():
        y, r = int(row["Year"]), int(row["Round"])
        try:
            w = load_race_weather(y, r)
        except Exception:
            w = None
        if w is not None:
            weather_per_race[(y, r)] = w
        else:
            weather_per_race[(y, r)] = rng.choice(["Dry", "Wet", "Rain"], p=[0.5, 0.3, 0.2])
    print("Quali weather: using real FastF1 for {}/{} races (rest random)".format(
        sum(1 for (y, r) in weather_per_race if load_race_weather(y, r) is not None),
        len(weather_per_race),
    ))
    df = build_quali_training_features(df, weather_per_race=weather_per_race)
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
    try:
        main()
        train_quali_model()
    except Exception as e:
        if "RateLimitExceeded" in type(e).__name__ or "RateLimit" in str(e):
            print("\n*** Rate limit exceeded (500 API calls/hour). ***")
            print("Wait ~1 hour and run again; more data will be served from cache.")
            print("Or use fewer years:  python scripts/train_model.py --start 2024 --end 2025")
        raise
