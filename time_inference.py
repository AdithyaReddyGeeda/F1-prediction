"""
Predict total race time (winner) and fastest lap (driver + time).

Race time model: XGBoost regression on winner time in seconds.
    Features: circuit_enc, total_laps, pole_time_sec, weather_dry/wet/rain,
              avg_winner_time_at_circuit (historical), sc_proxy

Fastest lap model:
    - Regressor: predict fastest lap time in seconds
      Features: pole_time_sec, circuit_enc, weather
    - Driver scoring: team FL rate, grid position → probabilities
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_ARTIFACTS = Path(__file__).resolve().parent / "model_artifacts"
_TIME_MODEL_PATH = _ARTIFACTS / "race_time_model.joblib"
_FL_MODEL_PATH = _ARTIFACTS / "fastest_lap_model.joblib"
_TIME_ENC_PATH = _ARTIFACTS / "race_time_encoders.joblib"
_FL_ENC_PATH = _ARTIFACTS / "fastest_lap_encoders.joblib"


def _load(path):
    try:
        import joblib
        if path.exists():
            return joblib.load(path)
    except Exception:
        pass
    return None


def _sec_to_str(sec: float, include_hours: bool = True) -> str:
    """Convert seconds to H:MM:SS.mmm or M:SS.mmm"""
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    if include_hours:
        return f"{h}:{m:02d}:{s:06.3f}"
    return f"{m}:{s:06.3f}"


def predict_race_time(
    circuit: str,
    total_laps: int,
    pole_time_sec: Optional[float],
    weather_str: str = "Dry",
) -> dict:
    """
    Predict winner's total race time and confidence interval.
    Returns dict with:
        predicted_sec  : float  - predicted winner time in seconds
        predicted_str  : str    - formatted as "H:MM:SS.mmm"
        low_sec        : float  - lower bound (optimistic / no SC)
        high_sec       : float  - upper bound (SC deployment)
        method         : str    - "model" or "heuristic"
    """
    model = _load(_TIME_MODEL_PATH)
    encoders = _load(_TIME_ENC_PATH)

    if model is not None and encoders is not None and pole_time_sec is not None:
        try:
            enc_circuit = encoders["circuit"]
            circ_classes = list(enc_circuit.classes_) if hasattr(enc_circuit, "classes_") else []
            circ_enc = enc_circuit.transform([circuit])[0] if circuit in circ_classes else 0

            weather_map = {"Dry": [1, 0, 0], "Wet": [0, 1, 0], "Rain": [0, 0, 1]}
            w = weather_map.get(weather_str, [1, 0, 0])

            avg_time = encoders.get("avg_winner_time_map", {}).get(
                circuit, pole_time_sec * total_laps * 1.02
            )
            sc_proxy = encoders.get("sc_proxy_map", {}).get(circuit, 0.4)

            X = np.array([[circ_enc, total_laps, pole_time_sec, avg_time, sc_proxy] + w])
            scaler = encoders.get("scaler")
            if scaler is not None:
                X = scaler.transform(X)
            pred = float(model.predict(X)[0])
            weather_mult = 1.0 if weather_str == "Dry" else (1.12 if weather_str == "Rain" else 1.06)
            pred_wet = pred * weather_mult if weather_str != "Dry" else pred
            return {
                "predicted_sec": pred_wet,
                "predicted_str": _sec_to_str(pred_wet),
                "low_sec": pred_wet - 45,
                "high_sec": pred_wet + 90,
                "method": "model",
            }
        except Exception:
            pass

    # Heuristic fallback
    if pole_time_sec is not None and total_laps > 0:
        weather_mult = 1.0 if weather_str == "Dry" else (1.13 if weather_str == "Rain" else 1.07)
        est = pole_time_sec * total_laps * 1.034 * weather_mult
        return {
            "predicted_sec": est,
            "predicted_str": _sec_to_str(est),
            "low_sec": est - 60,
            "high_sec": est + 120,
            "method": "heuristic",
        }

    return {}


def predict_fastest_lap(
    circuit: str,
    grid_df: pd.DataFrame,
    pole_time_sec: Optional[float],
    weather_str: str = "Dry",
) -> dict:
    """
    Predict fastest lap time and most likely driver to set it.
    grid_df must have: Abbreviation, TeamName, GridPosition columns.

    Returns dict with:
        predicted_time_sec  : float
        predicted_time_str  : str    - "M:SS.mmm"
        driver              : str    - Abbreviation most likely to set FL
        driver_name         : str    - full driver name if available
        probabilities       : dict   - {abbreviation: probability} for top 5
        method              : str
    """
    fl_model = _load(_FL_MODEL_PATH)
    fl_enc = _load(_FL_ENC_PATH)

    # --- Fastest lap TIME ---
    fl_time_sec = None
    if pole_time_sec is not None:
        ratio = {"Dry": 0.983, "Wet": 0.998, "Rain": 1.005}.get(weather_str, 0.983)
        fl_time_sec = pole_time_sec * ratio

        if fl_model is not None and fl_enc is not None:
            try:
                enc_c = fl_enc.get("circuit")
                circ_classes = list(enc_c.classes_) if hasattr(enc_c, "classes_") else []
                circ_enc = enc_c.transform([circuit])[0] if circuit in circ_classes else 0
                w = {"Dry": [1, 0, 0], "Wet": [0, 1, 0], "Rain": [0, 0, 1]}.get(
                    weather_str, [1, 0, 0]
                )
                X_time = np.array([[circ_enc, pole_time_sec] + w])
                scaler = fl_enc.get("scaler")
                if scaler is not None:
                    X_time = scaler.transform(X_time)
                fl_time_sec = float(fl_model.predict(X_time)[0])
            except Exception:
                pass

    # --- Fastest lap DRIVER (heuristic: grid position + team FL rate) ---
    enc_map = {} if fl_enc is None else fl_enc.get("global_team_fl_rate", {})
    probabilities = {}
    if not grid_df.empty:
        scores = []
        for _, row in grid_df.iterrows():
            abbrev = str(row.get("Abbreviation", ""))
            team = str(row.get("TeamName", ""))
            gpos = float(row.get("GridPosition", 10))

            base = max(0, 23 - gpos)
            team_fl_boost = enc_map.get(team, 0.15) * 30
            if gpos > 7:
                base *= 0.6
            score = base + team_fl_boost
            scores.append((abbrev, score, row.get("DriverName", abbrev)))

        total = sum(s for _, s, _ in scores)
        if total > 0:
            probabilities = {a: round(s / total, 3) for a, s, _ in scores}

        top_driver = max(scores, key=lambda x: x[1]) if scores else ("", 0, "")
        fl_driver = top_driver[0]
        fl_name = top_driver[2]
    else:
        fl_driver, fl_name = "", ""

    result = {
        "driver": fl_driver,
        "driver_name": fl_name,
        "probabilities": dict(sorted(probabilities.items(), key=lambda x: -x[1])[:5]),
        "method": "model" if fl_model is not None else "heuristic",
    }
    if fl_time_sec is not None:
        result["predicted_time_sec"] = fl_time_sec
        result["predicted_time_str"] = _sec_to_str(fl_time_sec, include_hours=False)
    return result
