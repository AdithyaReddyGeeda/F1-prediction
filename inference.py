"""
Prediction inference: XGBoost model (if available) or heuristic fallback.
Features: grid_pos, driver, team, circuit, recent_form, weather one-hot (Dry/Wet/Rain), grid_pos_x_rain.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import ROOKIE_DEFAULT_AVG_POSITION

# Heuristic fallback (no xgboost required)
from model import build_prediction_for_event

_MODEL_DIR = Path(__file__).resolve().parent / "model_artifacts"
_MODEL_PATH = _MODEL_DIR / "xgboost_model.joblib"
_ENCODERS_PATH = _MODEL_DIR / "encoders.joblib"

ROOKIE_AVG_POS = ROOKIE_DEFAULT_AVG_POSITION

# Must match train_model.py
WEATHER_CATEGORIES = ["Dry", "Wet", "Rain"]


def _load_model_and_encoders():
    """Load XGBoost and encoders. Load weather OHE from encoders or models/weather_ohe.joblib."""
    try:
        import joblib
        if _MODEL_PATH.exists() and _ENCODERS_PATH.exists():
            model = joblib.load(_MODEL_PATH)
            encoders = joblib.load(_ENCODERS_PATH)
            if encoders.get("weather_encoder") is None:
                weather_path = _MODEL_DIR.parent / "models" / "weather_ohe.joblib"
                if weather_path.exists():
                    encoders["weather_encoder"] = joblib.load(weather_path)
            return model, encoders
    except Exception:
        pass
    return None, None


def get_recent_form(
    year: int,
    round_number: int,
    driver_abbrevs: list,
    history_races: int = 5,
    seasons_back: int = 2,
) -> dict:
    """
    For each driver, get average finishing position over last N races (cross-season).
    Drivers not found get ROOKIE_AVG_POS.
    """
    from data import get_event_schedule, load_race_results
    from features import restrict_to_current_grid

    driver_form = {d: ROOKIE_AVG_POS for d in driver_abbrevs}
    history_frames = []
    races_collected = 0

    for y in range(year, max(year - seasons_back, 2014) - 1, -1):
        if y == year:
            start = round_number - 1
            if start < 1:
                continue
            rounds = range(start, 0, -1)
        else:
            try:
                sched = get_event_schedule(y)
                if sched.empty:
                    continue
                max_r = int(sched["RoundNumber"].max())
                rounds = range(max_r, 0, -1)
            except Exception:
                continue
        for r in rounds:
            if races_collected >= history_races:
                break
            try:
                df_r = load_race_results(y, r)
                if df_r.empty:
                    continue
                df_r["Round"] = r
                df_r["Year"] = y
                history_frames.append((y, r, df_r))
                races_collected += 1
            except Exception:
                continue
        if races_collected >= history_races:
            break

    if not history_frames:
        return driver_form

    # Build grid meta from last race in window
    last_y, last_r = max((y, r) for y, r, _ in history_frames)
    last_df = next(df for y, r, df in history_frames if y == last_y and r == last_r)
    meta = last_df[["DriverNumber", "Abbreviation", "TeamName"]].drop_duplicates("Abbreviation")
    history = restrict_to_current_grid(history_frames, meta)
    if history.empty:
        return driver_form

    agg = history.groupby("Abbreviation")["Position"].mean()
    for ab in driver_abbrevs:
        if ab in agg.index:
            driver_form[ab] = float(agg[ab])
    return driver_form


def predict_finishing_order(
    grid_df: pd.DataFrame,
    circuit: str,
    weather_str: str = "Dry",
    year: int = 2026,
    round_number: int = 1,
    use_xgboost: bool = True,
    force_heuristic: bool = False,
    return_debug: bool = False,
):
    """
    grid_df: DataFrame with DriverName/Abbreviation, TeamName, GridPosition.
    weather_str: "Dry" | "Wet" | "Rain" (one-hot encoded for model).
    Returns (pred_df, debug_info) if return_debug else pred_df only.
    """
    if grid_df.empty:
        out = pd.DataFrame(columns=["PredictedRank", "Driver", "Team"])
        return (out, None) if return_debug else out

    grid_df = grid_df.copy()
    if "Driver" not in grid_df.columns:
        grid_df["Driver"] = grid_df.get("DriverName", grid_df.get("Abbreviation", ""))
    if "Team" not in grid_df.columns:
        grid_df["Team"] = grid_df.get("TeamName", "")
    if "GridPosition" not in grid_df.columns:
        grid_df["GridPosition"] = np.arange(1, len(grid_df) + 1)

    abbrevs = grid_df.get("Abbreviation", grid_df["Driver"].str[:3].str.upper()).tolist()
    form = get_recent_form(year, round_number, abbrevs)

    model, encoders = _load_model_and_encoders() if use_xgboost and not force_heuristic else (None, None)
    debug_info = None

    if model is not None and encoders is not None:
        try:
            X, debug_info = _build_features(
                grid_df, circuit, weather_str, form, encoders, return_debug=return_debug
            )
            if X is not None and len(X) == len(grid_df):
                pred_pos = model.predict(X)
                pred_pos = np.clip(pred_pos, 1.0, 22.0)
                grid_df = grid_df.copy()
                grid_df["PredictedPosition"] = pred_pos
                grid_df = grid_df.sort_values("PredictedPosition").reset_index(drop=True)
                grid_df["PredictedRank"] = np.arange(1, len(grid_df) + 1)
                out = grid_df[["PredictedRank", "Driver", "Team"]]
                if return_debug and debug_info is not None and hasattr(model, "feature_importances_"):
                    fn = debug_info.get("feature_names", [])
                    imp = model.feature_importances_.tolist()
                    debug_info["feature_importances"] = dict(zip(fn, imp)) if fn else dict(zip(range(len(imp)), imp))
                return (out, debug_info) if return_debug else out
        except Exception:
            pass

    try:
        pred = build_prediction_for_event(year, round_number, history_races=5, seasons_back=2)
        if not pred.empty:
            out = pred[["PredictedRank", "Driver", "Team"]].head(22)
            # Weather-aware shuffle when no XGBoost: wet/rain add variance so order can change
            if weather_str != "Dry" and len(out) > 1:
                rng = np.random.default_rng(hash(weather_str) % (2**32))
                noise = rng.uniform(-1.5, 1.5, size=len(out))
                out = out.copy()
                out["_sort"] = out["PredictedRank"].values.astype(float) + noise
                out = out.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
                out["PredictedRank"] = np.arange(1, len(out) + 1)
            return (out, None) if return_debug else out
    except Exception:
        pass

    grid_df["Form"] = [form.get(a, ROOKIE_AVG_POS) for a in abbrevs]
    # Weather-aware: add noise for Wet/Rain so same grid can yield different order
    if weather_str != "Dry":
        rng = np.random.default_rng(hash(weather_str) % (2**32))
        grid_df["_noise"] = rng.uniform(-1.2, 1.2, size=len(grid_df))
        grid_df["Form"] = grid_df["Form"] + grid_df["_noise"]
    grid_df = grid_df.sort_values(["Form", "GridPosition"]).reset_index(drop=True)
    if "_noise" in grid_df.columns:
        grid_df = grid_df.drop(columns=["_noise"])
    grid_df["PredictedRank"] = np.arange(1, len(grid_df) + 1)
    out = grid_df[["PredictedRank", "Driver", "Team"]]
    return (out, None) if return_debug else out


def _circuit_type_dummies(circuit: str) -> tuple[float, float, float]:
    """Return (street, high_speed, technical) 0/1 dummies. Matches utils.race_features."""
    c = str(circuit).lower()
    street = 1.0 if any(x in c for x in ["monaco", "baku", "singapore", "miami", "vegas", "australian", "canadian", "saudi", "bahrain", "abu dhabi", "qatar", "azerbaijan"]) else 0.0
    high = 1.0 if any(x in c for x in ["monza", "spa", "sakhir", "jeddah", "silverstone", "japanese", "suzuka", "british", "belgian", "italian"]) else 0.0
    technical = 1.0 if (street == 0.0 and high == 0.0) else 0.0
    return street, high, technical


def _build_features(
    grid_df: pd.DataFrame,
    circuit: str,
    weather_str: str,
    form: dict,
    encoders: dict,
    return_debug: bool = False,
):
    """
    Build feature matrix. Supports legacy (9 cols) or full (19 cols) from training.
    Full: GridPosition, QualiPosition, RecentForm, ConstructorEwma, track_avg_*, driver_team_synergy,
    momentum, driver_rain_delta, driver_enc, team_enc, circuit_enc, circuit_type_*, weather_*, grid_pos_x_rain.
    """
    try:
        n = len(grid_df)
        abbrevs = grid_df.get("Abbreviation", grid_df["Driver"].astype(str).str[:3].str.upper()).tolist()
        teams = grid_df["TeamName"].astype(str).tolist()
        enc_driver = encoders.get("driver")
        enc_team = encoders.get("team")
        enc_circuit = encoders.get("circuit")
        enc_weather = encoders.get("weather_encoder")
        feature_names = encoders.get("feature_names") or []

        if enc_driver is None or enc_team is None or enc_circuit is None:
            return None, None
        if enc_weather is None:
            weather_onehot = np.array([[1, 0, 0]] * n) if weather_str == "Dry" else np.array([[0, 0, 1]] * n)
        else:
            w = weather_str if weather_str in WEATHER_CATEGORIES else "Dry"
            weather_onehot = enc_weather.transform(np.array([[w]])).reshape(1, -1)
            weather_onehot = np.repeat(weather_onehot, n, axis=0)

        grid_pos = grid_df["GridPosition"].values.astype(float)
        is_rain = 1.0 if weather_str == "Rain" else 0.0
        grid_pos_x_rain = grid_pos * is_rain
        def _safe_transform(enc, vals, default_idx=0):
            out = []
            classes = list(enc.classes_) if hasattr(enc, "classes_") else []
            for v in vals:
                v_str = str(v)
                if v_str in classes:
                    out.append(enc.transform([v_str])[0])
                else:
                    out.append(default_idx)
            return np.array(out)
        driver_enc = _safe_transform(enc_driver, abbrevs)
        team_enc = _safe_transform(enc_team, teams)
        circuit_enc = _safe_transform(enc_circuit, [circuit] * n)
        recent_form = np.array([form.get(a, ROOKIE_AVG_POS) for a in abbrevs], dtype=float)

        # Full feature set (new training pipeline)
        if encoders.get("track_avg_driver_map") is not None and "QualiPosition" in (feature_names or []):
            default_track = encoders.get("DEFAULT_TRACK_AVG", 10.0)
            default_synergy = encoders.get("DEFAULT_SYNERGY", 10.0)
            track_driver_map = encoders.get("track_avg_driver_map") or {}
            track_team_map = encoders.get("track_avg_team_map") or {}
            synergy_map = encoders.get("driver_team_synergy_map") or {}
            rain_delta_map = encoders.get("driver_rain_delta_map") or {}
            quali_pos = grid_pos  # use grid as quali proxy at inference
            constructor_ewma = recent_form  # proxy with driver form
            track_avg_driver = np.array([track_driver_map.get((a, circuit), default_track) for a in abbrevs], dtype=float)
            track_avg_team = np.array([track_team_map.get((t, circuit), default_track) for t in teams], dtype=float)
            driver_team_synergy = np.array([synergy_map.get((a, t), default_synergy) for a, t in zip(abbrevs, teams)], dtype=float)
            momentum = np.zeros(n, dtype=float)
            driver_rain_delta = np.array([rain_delta_map.get(a, 0.0) for a in abbrevs], dtype=float)
            street, high, tech = _circuit_type_dummies(circuit)
            circuit_type_street = np.full(n, street)
            circuit_type_high_speed = np.full(n, high)
            circuit_type_technical = np.full(n, tech)
            X = np.column_stack([
                grid_pos,
                quali_pos,
                recent_form,
                constructor_ewma,
                track_avg_driver,
                track_avg_team,
                driver_team_synergy,
                momentum,
                driver_rain_delta,
                driver_enc,
                team_enc,
                circuit_enc,
                circuit_type_street,
                circuit_type_high_speed,
                circuit_type_technical,
                weather_onehot[:, 0],
                weather_onehot[:, 1],
                weather_onehot[:, 2],
                grid_pos_x_rain,
            ])
        else:
            # Legacy 9-feature matrix
            X = np.column_stack([
                grid_pos,
                driver_enc,
                team_enc,
                circuit_enc,
                recent_form,
                weather_onehot[:, 0],
                weather_onehot[:, 1],
                weather_onehot[:, 2],
                grid_pos_x_rain,
            ])
            if not feature_names:
                feature_names = [
                    "GridPosition", "driver_enc", "team_enc", "circuit_enc", "RecentForm",
                    "weather_Dry", "weather_Wet", "weather_Rain", "grid_pos_x_rain",
                ]

        debug_info = None
        if return_debug:
            names = feature_names or [f"f{i}" for i in range(X.shape[1])]
            debug_info = {
                "weather": weather_str,
                "weather_encoded": weather_onehot[0].tolist(),
                "weather_encoded_labels": ["weather_Dry", "weather_Wet", "weather_Rain"],
                "X_sample": dict(zip(names, X[0].tolist())) if len(X) > 0 else {},
                "feature_names": names,
                "X_shape": X.shape,
            }
        return X, debug_info
    except Exception:
        return None, None
