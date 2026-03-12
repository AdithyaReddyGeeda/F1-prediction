"""
Qualifying position prediction: XGBoost model (1–22) or heuristic from quali form.
Features: driver quali form (avg last 5), constructor quali strength, circuit, weather one-hot, season momentum.
Auto fallback to 2025 when no 2026 quali data. Rookies default to ~12–15.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data import get_event_schedule, load_qualifying_results

ROOKIE_QUALI_BASELINE = 13.0  # default avg quali pos for new drivers
QUALI_MODEL_DIR = Path(__file__).resolve().parent / "model_artifacts"
QUALI_MODEL_PATH = Path(__file__).resolve().parent / "models" / "quali_model.joblib"
QUALI_ENCODERS_PATH = Path(__file__).resolve().parent / "models" / "quali_encoders.joblib"
WEATHER_CATEGORIES = ["Dry", "Wet", "Rain"]


def _load_quali_model_and_encoders():
    try:
        import joblib
        if QUALI_MODEL_PATH.exists() and QUALI_ENCODERS_PATH.exists():
            model = joblib.load(QUALI_MODEL_PATH)
            encoders = joblib.load(QUALI_ENCODERS_PATH)
            return model, encoders
    except Exception:
        pass
    return None, None


def fetch_quali_form(
    year: int,
    round_number: int,
    driver_abbrevs: list,
    last_n: int = 5,
    seasons_back: int = 2,
) -> dict:
    """
    Per-driver average qualifying position over last N quali sessions (current year then 2025 fallback).
    Returns dict driver_abbrev -> avg_quali_pos (float). Missing/rookies get ROOKIE_QUALI_BASELINE.
    """
    out = {d: ROOKIE_QUALI_BASELINE for d in driver_abbrevs}
    collected = []
    # Current year then previous
    for y in range(year, max(year - seasons_back, 2014) - 1, -1):
        if y == year:
            start_r = round_number - 1
            if start_r < 1:
                continue
            rounds = range(start_r, 0, -1)
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
            if len(collected) >= last_n:
                break
            try:
                q = load_qualifying_results(y, r)
                if q.empty:
                    continue
                q = q.rename(columns={"QualiPosition": "Position"})
                q["Year"] = y
                q["Round"] = r
                collected.append(q)
            except Exception:
                continue
        if len(collected) >= last_n:
            break

    if not collected:
        return out
    combined = pd.concat(collected, ignore_index=True).dropna(subset=["Position", "Abbreviation"])
    if combined.empty:
        return out
    agg = combined.groupby("Abbreviation")["Position"].mean()
    for ab in driver_abbrevs:
        if ab in agg.index:
            out[ab] = float(agg[ab])
    return out


def fetch_constructor_quali_strength(
    year: int,
    round_number: int,
    team_names: list,
    last_n: int = 5,
    seasons_back: int = 2,
) -> dict:
    """Per-constructor average best quali position in session (lower = stronger). Missing -> 10."""
    out = {t: 10.0 for t in team_names}
    collected = []
    for y in range(year, max(year - seasons_back, 2014) - 1, -1):
        if y == year:
            start_r = round_number - 1
            if start_r < 1:
                continue
            rounds = range(start_r, 0, -1)
        else:
            try:
                sched = get_event_schedule(y)
                if sched.empty:
                    continue
                rounds = range(int(sched["RoundNumber"].max()), 0, -1)
            except Exception:
                continue
        for r in rounds:
            if len(collected) >= last_n:
                break
            try:
                q = load_qualifying_results(y, r)
                if q.empty:
                    continue
                collected.append(q)
            except Exception:
                continue
        if len(collected) >= last_n:
            break

    if not collected:
        return out
    combined = pd.concat(collected, ignore_index=True)
    pos_col = "QualiPosition" if "QualiPosition" in combined.columns else "Position"
    combined["Position"] = pd.to_numeric(combined[pos_col], errors="coerce")
    team_avg = combined.groupby("TeamName")["Position"].mean()
    for t in team_names:
        if t in team_avg.index:
            out[t] = float(team_avg[t])
    return out


def predict_quali_order(
    drivers_df: pd.DataFrame,
    circuit: str,
    weather_str: str = "Dry",
    year: int = 2026,
    round_number: int = 1,
    return_debug: bool = False,
):
    """
    Predict qualifying order 1–22. drivers_df must have DriverName/Abbreviation, TeamName.
    Returns (quali_df with PredictedQualiPos, Driver, Team[, optional columns], debug_info or None).
    """
    if drivers_df.empty:
        out = pd.DataFrame(columns=["PredictedQualiPos", "Driver", "Team"])
        return (out, None) if return_debug else out

    drivers_df = drivers_df.copy()
    if "Driver" not in drivers_df.columns:
        drivers_df["Driver"] = drivers_df.get("DriverName", drivers_df.get("Abbreviation", ""))
    if "Team" not in drivers_df.columns:
        drivers_df["Team"] = drivers_df.get("TeamName", "")
    abbrevs = drivers_df.get("Abbreviation", drivers_df["Driver"].astype(str).str[:3].str.upper()).tolist()
    teams = drivers_df["TeamName"].astype(str).tolist()

    quali_form = fetch_quali_form(year, round_number, abbrevs, last_n=5, seasons_back=2)
    constructor_strength = fetch_constructor_quali_strength(year, round_number, list(dict.fromkeys(teams)), last_n=5, seasons_back=2)
    # Season momentum: round/24 so early rounds slightly lower
    momentum = round_number / 24.0 if year >= 2020 else 0.5

    model, encoders = _load_quali_model_and_encoders()
    debug_info = None

    if model is not None and encoders is not None:
        try:
            X, debug_info = _build_quali_features(
                drivers_df, circuit, weather_str, quali_form, constructor_strength, momentum, encoders, return_debug
            )
            if X is not None and len(X) == len(drivers_df):
                pred_pos = model.predict(X)
                drivers_df = drivers_df.copy()
                drivers_df["PredictedQualiPos"] = pred_pos
                drivers_df = drivers_df.sort_values("PredictedQualiPos").reset_index(drop=True)
                drivers_df["PredictedQualiPos"] = np.arange(1, len(drivers_df) + 1)
                out = drivers_df[["PredictedQualiPos", "Driver", "Team"]].copy()
                if return_debug and debug_info is not None and hasattr(model, "feature_importances_"):
                    fn = debug_info.get("feature_names", [])
                    imp = model.feature_importances_.tolist()
                    debug_info["feature_importances"] = dict(zip(fn, imp)) if fn else dict(zip(range(len(imp)), imp))
                return (out, debug_info) if return_debug else out
        except Exception:
            pass

    # Heuristic: sort by quali form then constructor strength
    drivers_df["QualiForm"] = [quali_form.get(a, ROOKIE_QUALI_BASELINE) for a in abbrevs]
    drivers_df["ConstStrength"] = [constructor_strength.get(t, 10.0) for t in teams]
    drivers_df = drivers_df.sort_values(["QualiForm", "ConstStrength"]).reset_index(drop=True)
    drivers_df["PredictedQualiPos"] = np.arange(1, len(drivers_df) + 1)
    out = drivers_df[["PredictedQualiPos", "Driver", "Team"]].copy()
    return (out, None) if return_debug else out


def _build_quali_features(
    drivers_df: pd.DataFrame,
    circuit: str,
    weather_str: str,
    quali_form: dict,
    constructor_strength: dict,
    momentum: float,
    encoders: dict,
    return_debug: bool = False,
):
    """Build feature matrix for quali model. Returns (X, debug_info or None)."""
    try:
        n = len(drivers_df)
        abbrevs = drivers_df.get("Abbreviation", drivers_df["Driver"].str[:3].str.upper()).tolist()
        teams = drivers_df["TeamName"].astype(str).tolist()

        enc_driver = encoders.get("driver")
        enc_team = encoders.get("team")
        enc_circuit = encoders.get("circuit")
        enc_weather = encoders.get("weather_encoder")
        feature_names = encoders.get("feature_names", [])

        if enc_driver is None or enc_team is None or enc_circuit is None:
            return None, None

        quali_form_arr = np.array([quali_form.get(a, ROOKIE_QUALI_BASELINE) for a in abbrevs], dtype=float)
        const_strength_arr = np.array([constructor_strength.get(t, 10.0) for t in teams], dtype=float)
        momentum_arr = np.full(n, momentum, dtype=float)

        if enc_weather is None:
            weather_onehot = np.array([[1, 0, 0]] * n) if weather_str == "Dry" else np.array([[0, 0, 1]] * n)
        else:
            w = weather_str if weather_str in WEATHER_CATEGORIES else "Dry"
            weather_onehot = enc_weather.transform(np.array([[w]])).reshape(1, -1)
            weather_onehot = np.repeat(weather_onehot, n, axis=0)

        driver_enc = enc_driver.transform(abbrevs)
        team_enc = enc_team.transform(teams)
        circuit_enc = enc_circuit.transform([circuit] * n)

        X = np.column_stack([
            quali_form_arr,
            const_strength_arr,
            driver_enc,
            team_enc,
            circuit_enc,
            momentum_arr,
            weather_onehot[:, 0],
            weather_onehot[:, 1],
            weather_onehot[:, 2],
        ])

        debug_info = None
        if return_debug:
            names = feature_names or [
                "quali_form", "constructor_strength", "driver_enc", "team_enc", "circuit_enc",
                "momentum", "weather_Dry", "weather_Wet", "weather_Rain",
            ]
            debug_info = {
                "weather": weather_str,
                "weather_encoded": weather_onehot[0].tolist(),
                "feature_names": names,
                "X_sample": dict(zip(names, X[0].tolist())) if len(X) > 0 else {},
                "X_shape": X.shape,
            }
        return X, debug_info
    except Exception:
        return None, None
