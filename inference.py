"""
Prediction inference: XGBoost model (if available) or heuristic fallback.
Features: grid_pos, driver, team, circuit, recent_form, weather one-hot (Dry/Wet/Rain), grid_pos_x_rain.
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import ROOKIE_DEFAULT_AVG_POSITION, ENGINE_BY_TEAM
from utils.race_features import get_circuit_abrasion_proxy, _circuit_to_type

# Heuristic fallback (no xgboost required)
from model import build_prediction_for_event

# Surface inference errors instead of swallowing them (see get_inference_warnings).
_INFERENCE_WARNINGS: list[str] = []

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
        _INFERENCE_WARNINGS.append(traceback.format_exc())
    return None, None


def get_inference_warnings() -> list[str]:
    """Return and clear the list of inference warnings (e.g. for display in UI)."""
    global _INFERENCE_WARNINGS
    out = list(_INFERENCE_WARNINGS)
    _INFERENCE_WARNINGS = []
    return out


def get_recent_form(
    year: int,
    round_number: int,
    driver_abbrevs: list,
    history_races: int = 5,
    seasons_back: int = 2,
    alpha: float = 0.4,
) -> dict:
    """
    For each driver, get EWMA finishing position over last N races (cross-season),
    matching the training pipeline (utils.race_features.compute_ewma_form).
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

    # EWMA over recent races, matching training pipeline (higher weight on latest races).
    history = history.sort_values(["Abbreviation", "Year", "Round"])
    alpha = float(alpha)
    for ab in driver_abbrevs:
        sub = history[history["Abbreviation"] == ab]
        if sub.empty:
            continue
        pos = sub["Position"].astype(float)
        if len(pos) == 0:
            continue
        weights = np.array([(1 - alpha) ** i for i in range(len(pos) - 1, -1, -1)], dtype=float)
        if weights.sum() == 0:
            continue
        weights /= weights.sum()
        driver_form[ab] = float(np.dot(weights, pos.values))

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
    Returns (pred_df, debug_info, source) if return_debug else pred_df only.
    source is "xgboost" | "heuristic_build" | "heuristic_form".
    """
    if grid_df.empty:
        out = pd.DataFrame(columns=["PredictedRank", "Driver", "Team"])
        return (out, None, "heuristic_form") if return_debug else out

    grid_df = grid_df.copy()
    if "Driver" not in grid_df.columns:
        grid_df["Driver"] = grid_df.get("DriverName", grid_df.get("Abbreviation", ""))
    if "Team" not in grid_df.columns:
        grid_df["Team"] = grid_df.get("TeamName", "")
    if "GridPosition" not in grid_df.columns:
        grid_df["GridPosition"] = np.arange(1, len(grid_df) + 1)

    # Load model/encoders first so we can read ewma_alpha / blend info.
    model, encoders = _load_model_and_encoders() if use_xgboost and not force_heuristic else (None, None)
    ewma_alpha = 0.4
    if encoders is not None:
        try:
            ewma_alpha = float(encoders.get("ewma_alpha", ewma_alpha))
        except Exception:
            pass

    abbrevs = grid_df.get("Abbreviation", grid_df["Driver"].str[:3].str.upper()).tolist()
    form = get_recent_form(year, round_number, abbrevs, alpha=ewma_alpha)
    debug_info = None

    if model is not None and encoders is not None:
        try:
            # Pass current context into encoders so feature builder can align GridPosition semantics.
            encoders = dict(encoders)
            encoders["current_year"] = year
            encoders["current_round"] = round_number
            X, debug_info = _build_features(
                grid_df, circuit, weather_str, form, encoders, return_debug=return_debug
            )
            if X is not None and len(X) == len(grid_df):
                pred_pos = model.predict(X)
                pred_pos = np.clip(pred_pos, 1.0, 22.0)
                if encoders.get("lgb_ensemble"):
                    try:
                        import joblib as _jl
                        _lgb_path = _MODEL_DIR / "lgb_model.joblib"
                        if _lgb_path.exists():
                            lgb_model = _jl.load(_lgb_path)
                            lgb_preds = lgb_model.predict(X)
                            pred_pos = 0.6 * pred_pos + 0.4 * lgb_preds
                            pred_pos = np.clip(pred_pos, 1.0, 22.0)
                    except Exception:
                        pass

                # Blend with QualiPosition (or GridPosition fallback) and assign per-race ranks,
                # matching training-time post-processing.
                feature_names = encoders.get("feature_names") or []
                quali_pos = grid_df["GridPosition"].values.astype(float)
                if feature_names and "QualiPosition" in feature_names:
                    try:
                        q_idx = feature_names.index("QualiPosition")
                        quali_pos = X[:, q_idx]
                    except Exception:
                        pass

                # Circuit-type-specific blend if available, else global blend_ratio, else 0.7.
                blend_ratio = 0.7
                circuit_type = _circuit_to_type(circuit)
                blend_by_type = encoders.get("blend_by_type") or {}
                if circuit_type in blend_by_type:
                    blend_ratio = float(blend_by_type[circuit_type])
                else:
                    blend_ratio = float(encoders.get("blend_ratio", blend_ratio))

                pred_blended = blend_ratio * pred_pos + (1.0 - blend_ratio) * quali_pos
                pred_blended = np.clip(pred_blended, 1.0, 22.0)

                order = np.argsort(pred_blended)
                ranks = np.empty_like(order)
                ranks[order] = np.arange(1, len(order) + 1)

                grid_df = grid_df.copy()
                grid_df["PredictedPosition"] = ranks.astype(float)
                grid_df = grid_df.sort_values("PredictedPosition").reset_index(drop=True)
                grid_df["PredictedRank"] = np.arange(1, len(grid_df) + 1)
                out = grid_df[["PredictedRank", "Driver", "Team"]]
                if return_debug and debug_info is not None and hasattr(model, "feature_importances_"):
                    fn = debug_info.get("feature_names", [])
                    imp = model.feature_importances_.tolist()
                    debug_info["feature_importances"] = dict(zip(fn, imp)) if fn else dict(zip(range(len(imp)), imp))
                return (out, debug_info, "xgboost") if return_debug else out
        except Exception:
            _INFERENCE_WARNINGS.append(traceback.format_exc())

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
            return (out, None, "heuristic_build") if return_debug else out
    except Exception:
        _INFERENCE_WARNINGS.append(traceback.format_exc())

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
    return (out, None, "heuristic_form") if return_debug else out


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
        use_engine = "engine_enc" in (feature_names or [])
        enc_engine = encoders.get("engine")
        engines = [ENGINE_BY_TEAM.get(t, "Other") for t in teams]
        engine_enc = (_safe_transform(enc_engine, engines) if enc_engine is not None else np.zeros(n, dtype=int)) if use_engine else None
        circuit_enc = _safe_transform(enc_circuit, [circuit] * n)
        recent_form = np.array([form.get(a, ROOKIE_AVG_POS) for a in abbrevs], dtype=float)

        # Full feature set (new training pipeline with teammate_delta, constructor_dnf_rate, form_x_teammate_delta)
        if encoders.get("track_avg_driver_map") is not None and "QualiPosition" in (feature_names or []):
            default_track = encoders.get("DEFAULT_TRACK_AVG", 10.0)
            default_synergy = encoders.get("DEFAULT_SYNERGY", 10.0)
            track_driver_map = encoders.get("track_avg_driver_map") or {}
            track_team_map = encoders.get("track_avg_team_map") or {}
            synergy_map = encoders.get("driver_team_synergy_map") or {}
            rain_delta_map = encoders.get("driver_rain_delta_map") or {}
            # Qualifying position: actual grid slot
            quali_pos = grid_pos
            # Quali gap to pole (seconds); only if model was trained with this feature
            use_quali_gap = "quali_gap_to_pole" in (feature_names or [])
            quali_gap_to_pole = np.full(n, 2.0, dtype=float)
            if use_quali_gap:
                try:
                    from data import load_quali_gaps
                    _y = encoders.get("current_year", 2026)
                    _r = encoders.get("current_round", 1)
                    qg_df = load_quali_gaps(_y, _r)
                    if not qg_df.empty and "Abbreviation" in qg_df.columns and "quali_gap_to_pole" in qg_df.columns:
                        gap_map = qg_df.set_index("Abbreviation")["quali_gap_to_pole"].to_dict()
                        for i, ab in enumerate(abbrevs):
                            if ab in gap_map and pd.notna(gap_map[ab]):
                                quali_gap_to_pole[i] = float(gap_map[ab])
                except Exception:
                    pass
            # GridPosition in training is previous race finish; reuse get_recent_form with history_races=1
            # If last-race result is missing, fall back to current grid position.
            try:
                prev_form = get_recent_form(
                    year=encoders.get("current_year", 2026),
                    round_number=encoders.get("current_round", 1),
                    driver_abbrevs=abbrevs,
                    history_races=1,
                    seasons_back=2,
                    alpha=float(encoders.get("ewma_alpha", 0.4)),
                )
            except Exception:
                _INFERENCE_WARNINGS.append(traceback.format_exc())
                prev_form = {}
            # If last race result missing, fall back to current grid position
            grid_prev = np.array(
                [prev_form.get(a, gp) for a, gp in zip(abbrevs, grid_pos)],
                dtype=float,
            )
            constructor_ewma = recent_form
            track_avg_driver = np.array([track_driver_map.get((a, circuit), default_track) for a in abbrevs], dtype=float)
            track_avg_team = np.array([track_team_map.get((t, circuit), default_track) for t in teams], dtype=float)
            driver_team_synergy = np.array([synergy_map.get((a, t), default_synergy) for a, t in zip(abbrevs, teams)], dtype=float)
            teammate_delta = np.zeros(n, dtype=float)
            constructor_dnf_rate = np.zeros(n, dtype=float)
            use_circuit_dnf = "constructor_dnf_rate_at_circuit" in (feature_names or [])
            circuit_dnf_rate_map = encoders.get("circuit_dnf_rate_map") or {}
            constructor_dnf_rate_at_circuit = np.array(
                [circuit_dnf_rate_map.get((t, circuit), 0.0) for t in teams],
                dtype=float,
            )
            driver_dnf_rate = np.zeros(n, dtype=float)
            circuit_abrasion_proxy = np.full(n, get_circuit_abrasion_proxy(circuit))
            tyre_life_penalty_proxy = np.full(n, 0.5)
            driver_tyre_management_proxy = np.zeros(n, dtype=float)
            form_x_teammate_delta = np.zeros(n, dtype=float)
            momentum = np.zeros(n, dtype=float)
            driver_rain_delta = np.array([rain_delta_map.get(a, 0.0) for a in abbrevs], dtype=float)
            fp1_delta = np.zeros(n, dtype=float)
            fp2_delta = np.zeros(n, dtype=float)
            fp3_delta = np.zeros(n, dtype=float)

            # ── Clean air pace ──
            cap_map = encoders.get("cap_map", {})
            circuit_cap_map = encoders.get("circuit_cap_map", {})
            live_cap = {}
            try:
                from data import load_clean_air_pace
                _y = encoders.get("current_year", 2026)
                _r = encoders.get("current_round", 1)
                cap_live_df = load_clean_air_pace(_y, _r)
                if not cap_live_df.empty:
                    live_cap = cap_live_df.set_index("Abbreviation")["clean_air_pace_sec"].to_dict()
            except Exception:
                pass
            global_cap_fallback = float(np.mean(list(cap_map.values()))) if cap_map else 88.0
            circuit_cap_fallback = circuit_cap_map.get(circuit, global_cap_fallback)
            clean_air_pace = np.array([
                live_cap.get(a) or cap_map.get(a) or circuit_cap_fallback
                for a in abbrevs
            ], dtype=float)

            # ── Sector times ──
            sector_map = encoders.get("sector_map", {})
            live_sectors = {}
            try:
                from data import load_quali_sector_times
                _y = encoders.get("current_year", 2026)
                _r = encoders.get("current_round", 1)
                sec_live_df = load_quali_sector_times(_y, _r)
                if not sec_live_df.empty:
                    for col in ["s1_gap", "s2_gap", "s3_gap", "total_sector_gap", "s1_pct", "s2_pct", "s3_pct"]:
                        if col in sec_live_df.columns:
                            live_sectors[col] = sec_live_df.set_index("Abbreviation")[col].to_dict()
            except Exception:
                pass

            def _get_sector_feature(col, driver_abbrev, circuit_name):
                if col in live_sectors and driver_abbrev in live_sectors[col]:
                    return float(live_sectors[col][driver_abbrev])
                driver_map = sector_map.get("{}_by_driver".format(col), {})
                if driver_abbrev in driver_map:
                    return float(driver_map[driver_abbrev])
                circuit_med_map = sector_map.get("{}_by_circuit".format(col), {})
                return float(circuit_med_map.get(circuit_name, 0.0))

            s1_gap_arr = np.array([_get_sector_feature("s1_gap", a, circuit) for a in abbrevs])
            s2_gap_arr = np.array([_get_sector_feature("s2_gap", a, circuit) for a in abbrevs])
            s3_gap_arr = np.array([_get_sector_feature("s3_gap", a, circuit) for a in abbrevs])
            total_sector_gap_arr = np.array([_get_sector_feature("total_sector_gap", a, circuit) for a in abbrevs])
            s1_pct_arr = np.array([_get_sector_feature("s1_pct", a, circuit) for a in abbrevs])
            s2_pct_arr = np.array([_get_sector_feature("s2_pct", a, circuit) for a in abbrevs])
            s3_pct_arr = np.array([_get_sector_feature("s3_pct", a, circuit) for a in abbrevs])

            t = _circuit_to_type(circuit)
            street = 1.0 if t == "street" else 0.0
            high = 1.0 if t == "high_speed" else 0.0
            tech = 1.0 if t == "technical" else 0.0
            circuit_type_street = np.full(n, street)
            circuit_type_high_speed = np.full(n, high)
            circuit_type_technical = np.full(n, tech)
            stack_parts = [grid_prev, quali_pos]
            if use_quali_gap:
                stack_parts.append(quali_gap_to_pole)
            stack_parts.extend([
                recent_form, constructor_ewma,
                track_avg_driver, track_avg_team, driver_team_synergy, teammate_delta,
                constructor_dnf_rate,
            ])
            if use_circuit_dnf:
                stack_parts.append(constructor_dnf_rate_at_circuit)
            stack_parts.extend([
                driver_dnf_rate, circuit_abrasion_proxy, tyre_life_penalty_proxy,
                driver_tyre_management_proxy, form_x_teammate_delta, momentum, driver_rain_delta,
                fp1_delta, fp2_delta, fp3_delta,
            ])
            if "clean_air_pace_sec" in (feature_names or []):
                stack_parts.append(clean_air_pace)
            if "s1_gap" in (feature_names or []):
                stack_parts.extend([s1_gap_arr, s2_gap_arr, s3_gap_arr, total_sector_gap_arr, s1_pct_arr, s2_pct_arr, s3_pct_arr])
            stack_parts.extend([driver_enc, team_enc])
            if engine_enc is not None:
                stack_parts.append(engine_enc)
            stack_parts.extend([
                circuit_enc, circuit_type_street, circuit_type_high_speed, circuit_type_technical,
                weather_onehot[:, 0], weather_onehot[:, 1], weather_onehot[:, 2], grid_pos_x_rain,
            ])
            X = np.column_stack(stack_parts)
            scaler = encoders.get("scaler")
            scale_idx = encoders.get("scale_idx")
            if scaler is not None and scale_idx is not None:
                X = X.astype(float)
                X[:, scale_idx] = scaler.transform(X[:, scale_idx])
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
        _INFERENCE_WARNINGS.append(traceback.format_exc())
        return None, None
