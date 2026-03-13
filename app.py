"""
F1 Race Prediction Dashboard — Streamlit.
Qualifying prediction (1–22) feeds into race prediction. 2025 fallback automatic when no 2026 data.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import GRID_2026
from inference import predict_finishing_order
from quali_inference import predict_quali_order
from data import enable_fastf1_cache, get_event_schedule, load_race_results
from utils.data_fetch import safe_get_drivers, safe_get_schedule
from utils.circuit_geo import get_circuit_track_data


@st.cache_data(ttl=600)
def _cached_load_race_results(year: int, round_number: int):
    """Cached FastF1 race results for faster reruns/refreshes."""
    return load_race_results(year, round_number)


def init_session_state():
    if "drivers_df" not in st.session_state:
        st.session_state.drivers_df = pd.DataFrame()
    if "schedule_df" not in st.session_state:
        st.session_state.schedule_df = pd.DataFrame()
    if "year" not in st.session_state:
        st.session_state.year = 2026
    if "predicted_quali_grid" not in st.session_state:
        st.session_state.predicted_quali_grid = None  # DataFrame with PredictedQualiPos, Driver, Team, DriverName, Abbreviation, TeamName, GridPosition
    if "race_result_from_quali" not in st.session_state:
        st.session_state.race_result_from_quali = None  # (pred_df, grid_df) when "Use grid & run race" clicked from Quali tab


def quali_to_grid_df(quali_df: pd.DataFrame, drivers_df: pd.DataFrame) -> pd.DataFrame:
    """Build grid DataFrame from quali result for race model: DriverName, Abbreviation, TeamName, GridPosition."""
    grid_list = []
    for _, row in quali_df.iterrows():
        driver_name = row.get("Driver", row.get("DriverName", ""))
        match = drivers_df[drivers_df["DriverName"].astype(str) == driver_name]
        if match.empty and "Driver" in drivers_df.columns:
            match = drivers_df[drivers_df["Driver"].astype(str) == driver_name]
        if not match.empty:
            r = match.iloc[0]
            grid_list.append({
                "DriverName": r["DriverName"],
                "Abbreviation": r.get("Abbreviation", str(driver_name)[:3]),
                "TeamName": r["TeamName"],
                "GridPosition": int(row.get("PredictedQualiPos", 10)),
            })
    if not grid_list:
        return pd.DataFrame()
    out = pd.DataFrame(grid_list)
    out["PredictedQualiPos"] = out["GridPosition"]
    out["Driver"] = out["DriverName"]
    out["Team"] = out["TeamName"]
    return out


def compute_rolling_mae(year: int, round_number: int, schedule_df: pd.DataFrame, weather_str: str, last_n: int = 5) -> tuple[list[tuple[int, float]], float]:
    """
    Compute MAE for the last N completed races (before current round).
    Returns (list of (round, mae)), mean_mae. Uses previous race finish order as grid for each race.
    """
    if schedule_df.empty or round_number <= 1:
        return [], 0.0
    rounds_to_eval = [r for r in range(max(2, round_number - last_n), round_number) if r >= 2]
    if not rounds_to_eval:
        return [], 0.0
    maes = []
    for r in rounds_to_eval:
        try:
            prev_race = _cached_load_race_results(year, r - 1)
            if prev_race.empty or "Abbreviation" not in prev_race.columns:
                continue
            grid_df = _race_result_to_grid_df(prev_race)
            if grid_df.empty:
                continue
            event_r = schedule_df.loc[schedule_df["RoundNumber"].astype(int) == r, "EventName"]
            event_r = event_r.iloc[0] if not event_r.empty else "Australian Grand Prix"
            pred = predict_finishing_order(
                grid_df,
                circuit=event_r,
                weather_str=weather_str,
                year=year,
                round_number=r,
                use_xgboost=True,
                force_heuristic=False,
                return_debug=False,
            )
            pred_df = pred[0] if isinstance(pred, tuple) else pred
            actual = _cached_load_race_results(year, r)
            if pred_df.empty or actual.empty:
                continue
            actual = actual.rename(columns={"Position": "ActualPosition"})
            abbrev_map = dict(zip(grid_df["DriverName"].astype(str), grid_df["Abbreviation"]))
            pred_df = pred_df.copy()
            pred_df["Abbreviation"] = pred_df["Driver"].astype(str).map(abbrev_map)
            merged = pred_df.merge(actual[["Abbreviation", "ActualPosition"]], on="Abbreviation", how="inner")
            if "PredictedRank" not in merged.columns or "ActualPosition" not in merged.columns or merged.empty:
                continue
            mae = (merged["PredictedRank"] - merged["ActualPosition"]).abs().mean()
            maes.append((r, float(mae)))
        except Exception:
            continue
    if not maes:
        return [], 0.0
    mean_mae = sum(m[1] for m in maes) / len(maes)
    return maes, mean_mae


def _race_result_to_grid_df(race_df: pd.DataFrame) -> pd.DataFrame:
    """Build grid DataFrame from race results (Position -> GridPosition)."""
    if race_df.empty or "Abbreviation" not in race_df.columns:
        return pd.DataFrame()
    race_df = race_df.sort_values("Position")
    grid_list = []
    for _, row in race_df.iterrows():
        grid_list.append({
            "DriverName": row.get("DriverName", row["Abbreviation"]),
            "Abbreviation": row["Abbreviation"],
            "TeamName": row["TeamName"],
            "GridPosition": int(row["Position"]),
        })
    return pd.DataFrame(grid_list)


@st.cache_data(ttl=600)
def load_drivers_and_schedule(year: int):
    used_2025_fallback = False
    with st.spinner("Loading drivers and schedule..."):
        drivers = safe_get_drivers(year, use_fallback_year=False)
        schedule = safe_get_schedule(year, use_fallback_year=False)
        if year >= 2026 and (drivers.empty or schedule.empty or len(drivers) < 22):
            used_2025_fallback = True
            drivers = safe_get_drivers(year, use_fallback_year=True)
            schedule = safe_get_schedule(year, use_fallback_year=True)
        if drivers.empty and year >= 2026:
            drivers = pd.DataFrame(GRID_2026)
        # 2026 schedule fallback is handled in safe_get_schedule (config.SCHEDULE_2026)
    return drivers, schedule, used_2025_fallback


def main():
    st.set_page_config(page_title="F1 Race Predictor", page_icon="🏎️", layout="wide")
    init_session_state()

    st.title("F1 Race Prediction Dashboard")
    st.markdown(
        "**Qualifying** prediction (1–22) can be used as the **race** starting grid. "
        "2025 fallback is automatic when 2026 data is missing."
    )

    enable_fastf1_cache()

    with st.sidebar:
        st.header("Configuration")
        year = st.number_input("Season year", min_value=2020, max_value=2030, value=2026, step=1)
        weather_options = ["Dry", "Wet", "Rain"]
        weather_str = st.radio("Weather", options=weather_options, index=0)

    drivers_df, schedule_df, used_2025_fallback = load_drivers_and_schedule(year)
    if drivers_df.empty:
        st.error("Could not load drivers.")
        return
    if used_2025_fallback:
        st.info("No usable 2026 results/form yet → using adjusted 2025 baseline.")

    drivers_df = drivers_df.copy()
    if "DriverName" not in drivers_df.columns:
        drivers_df["DriverName"] = drivers_df.get("Abbreviation", "?")
    driver_options = drivers_df["DriverName"].astype(str).tolist()

    if schedule_df.empty:
        event_name = "Australian Grand Prix"
        round_number = 1
    else:
        event_labels = [
            f"Round {int(r['RoundNumber'])} — {r['EventName']} ({r['EventDate']})"
            for _, r in schedule_df.iterrows()
        ]
        selected = st.selectbox("Select race", event_labels, index=0)
        round_number = int(selected.split()[1])
        event_name = schedule_df.loc[schedule_df["RoundNumber"] == round_number, "EventName"].iloc[0]

    # Track map (circuit-specific when in cache)
    track_data = get_circuit_track_data(event_name)
    if track_data:
        try:
            import folium
            m = folium.Map(location=track_data["center"], zoom_start=14, tiles="OpenStreetMap")
            if "outline" in track_data and track_data["outline"]:
                folium.PolyLine(track_data["outline"], color="red", weight=4, opacity=0.8).add_to(m)
            folium.Marker(track_data["center"], popup=event_name, tooltip="Circuit").add_to(m)
            with st.expander("Track map", expanded=False):
                st.components.v1.html(m._repr_html_(), height=350, scrolling=False)
        except Exception:
            st.caption("Track map unavailable for this circuit.")
    else:
        with st.expander("Track map", expanded=False):
            st.caption("No circuit map data for this venue. Add it in utils/circuit_geo.py.")

    tab_quali, tab_race = st.tabs(["Qualifying", "Race"])

    # ---------- Qualifying tab ----------
    with tab_quali:
        st.subheader("Predicted qualifying order (1–22)")
        if st.button("Predict Qualifying", key="predict_quali"):
            with st.spinner("Predicting qualifying..."):
                result = predict_quali_order(
                    drivers_df,
                    circuit=event_name,
                    weather_str=weather_str,
                    year=year,
                    round_number=round_number,
                    return_debug=True,
                )
            quali_df = result[0] if isinstance(result, tuple) else result
            quali_debug = result[1] if isinstance(result, tuple) and len(result) > 1 else None
            if quali_df is not None and not quali_df.empty:
                st.session_state.predicted_quali_grid = quali_to_grid_df(quali_df, drivers_df)
                st.session_state.race_result_from_quali = None  # reset when new quali run
                st.dataframe(quali_df, use_container_width=True, hide_index=True)
                _quali_export_df = quali_df[["PredictedQualiPos", "Driver", "Team"]].copy()
                _quali_export_df.columns = ["Position", "Driver", "Team"]
                st.download_button(
                    "Download qualifying prediction (CSV)",
                    data=_quali_export_df.to_csv(index=False),
                    file_name=f"quali_prediction_{event_name.replace(' ', '_')}_R{round_number}.csv",
                    mime="text/csv",
                    key="dl_quali",
                )
                if quali_debug:
                    with st.expander("Debug: quali features and sample"):
                        st.write("**Weather:**", quali_debug.get("weather", "—"))
                        st.write("**Weather encoded:**", quali_debug.get("weather_encoded", []))
                        st.write("**Feature names:**", quali_debug.get("feature_names", []))
                        st.write("**X sample (first row):**", quali_debug.get("X_sample", {}))
                        st.write("**X shape:**", quali_debug.get("X_shape", "—"))
                        if quali_debug.get("feature_importances"):
                            st.write("**Feature importances:**", quali_debug.get("feature_importances"))
            else:
                st.warning("No qualifying prediction produced.")
        else:
            if st.session_state.predicted_quali_grid is not None:
                _pq = st.session_state.predicted_quali_grid
                _disp = _pq[["PredictedQualiPos", "Driver", "Team"]].rename(columns={"PredictedQualiPos": "P"}) if "PredictedQualiPos" in _pq.columns else _pq
                st.dataframe(_disp, use_container_width=True, hide_index=True)
                _export_q = _pq[["PredictedQualiPos", "Driver", "Team"]].copy() if "PredictedQualiPos" in _pq.columns else _pq.copy()
                _export_q.columns = ["Position", "Driver", "Team"]
                st.download_button("Download qualifying (CSV)", data=_export_q.to_csv(index=False), file_name=f"quali_grid_{event_name.replace(' ', '_')}_R{round_number}.csv", mime="text/csv", key="dl_quali_cached")
                st.caption("Using last predicted qualifying grid for race (see Race tab).")
                # One-click: use this grid & run race prediction in same view
                if st.button("Use this grid & run race prediction", key="use_grid_run_race"):
                    with st.spinner("Running race prediction..."):
                        grid_for_race = st.session_state.predicted_quali_grid.copy()
                        res = predict_finishing_order(
                            grid_for_race,
                            circuit=event_name,
                            weather_str=weather_str,
                            year=year,
                            round_number=round_number,
                            use_xgboost=True,
                            force_heuristic=False,
                            return_debug=False,
                        )
                        pred_from_quali = res[0] if isinstance(res, tuple) else res
                        st.session_state.race_result_from_quali = (pred_from_quali, grid_for_race)
                if st.session_state.race_result_from_quali is not None:
                    pred_from_quali, _ = st.session_state.race_result_from_quali
                    if not pred_from_quali.empty:
                        st.subheader("Race prediction (from qualifying grid)")
                        st.dataframe(pred_from_quali, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download race prediction (CSV)",
                            data=pred_from_quali.to_csv(index=False),
                            file_name=f"race_prediction_{event_name.replace(' ', '_')}_R{round_number}.csv",
                            mime="text/csv",
                            key="dl_race_from_quali",
                        )
            else:
                st.info("Click **Predict Qualifying** to get an order and use it as the race grid.")

    # ---------- Race tab ----------
    with tab_race:
        override_manual = st.checkbox("Override with manual grid", value=False, key="override_grid")
        if override_manual or st.session_state.predicted_quali_grid is None:
            st.subheader("Grid order (starting positions 1–22)")
            st.caption("Assign each position to a driver, or run Qualifying prediction first to auto-fill.")
            grid_assignments = []
            n_drivers = min(22, len(driver_options))
            cols = st.columns(4)
            for pos in range(1, n_drivers + 1):
                with cols[(pos - 1) % 4]:
                    default_idx = (pos - 1) % len(driver_options)
                    chosen = st.selectbox(f"P{pos}", options=driver_options, index=default_idx, key=f"grid_{pos}")
                    grid_assignments.append((pos, chosen))
            grid_list = []
            for pos, driver_name in grid_assignments:
                row = drivers_df[drivers_df["DriverName"].astype(str) == driver_name].iloc[0]
                grid_list.append({
                    "DriverName": row["DriverName"],
                    "Abbreviation": row.get("Abbreviation", driver_name[:3]),
                    "TeamName": row["TeamName"],
                    "GridPosition": pos,
                })
            grid_df = pd.DataFrame(grid_list)
        else:
            grid_df = st.session_state.predicted_quali_grid.copy()
            st.subheader("Grid order (from predicted qualifying)")
            st.dataframe(
                grid_df[["GridPosition", "DriverName", "TeamName"]].rename(columns={"DriverName": "Driver", "GridPosition": "P"}),
                use_container_width=True,
                hide_index=True,
            )

        run_predict = st.button("Run Race Prediction", key="run_race")

        if not run_predict:
            st.info("Click **Run Race Prediction** to get predicted finishing order.")
        else:
            with st.spinner("Predicting finishing order..."):
                result = predict_finishing_order(
                    grid_df,
                    circuit=event_name,
                    weather_str=weather_str,
                    year=year,
                    round_number=round_number,
                    use_xgboost=True,
                    force_heuristic=False,
                    return_debug=True,
                )
            pred_df = result[0] if isinstance(result, tuple) else result
            debug_info = result[1] if isinstance(result, tuple) and len(result) > 1 else None

            if pred_df.empty:
                st.warning("No prediction produced.")
            else:
                st.subheader("Predicted finishing order")
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download race prediction (CSV)",
                    data=pred_df.to_csv(index=False),
                    file_name=f"race_prediction_{event_name.replace(' ', '_')}_R{round_number}.csv",
                    mime="text/csv",
                    key="dl_race_pred",
                )
                if debug_info:
                    with st.expander("Debug: race weather encoding and features"):
                        st.write("**Selected weather:**", debug_info.get("weather", "—"))
                        st.write("**Weather one-hot:**", debug_info.get("weather_encoded", []))
                        st.write("**Feature names:**", debug_info.get("feature_names", []))
                        st.write("**X sample (first row):**", debug_info.get("X_sample", {}))
                        st.write("**X shape:**", debug_info.get("X_shape", "—"))
                        imp = debug_info.get("feature_importances")
                        if imp:
                            if isinstance(imp, dict):
                                top10 = sorted(imp.items(), key=lambda x: -x[1])[:10]
                                st.write("**Top 10 feature importances:**", {k: round(v, 4) for k, v in top10})
                            else:
                                st.write("**Feature importances:**", imp)

                pred_df = pred_df.copy()
                pred_df["Strength"] = (len(pred_df) + 1) - pred_df["PredictedRank"]
                chart = alt.Chart(pred_df).mark_bar().encode(
                    x=alt.X("Driver:N", sort="-y"),
                    y="Strength:Q",
                    color="Team:N",
                    tooltip=["Driver", "Team", "PredictedRank"],
                ).properties(height=350)
                st.altair_chart(chart, use_container_width=True)

            today = dt.date.today()
            if "EventDate" in schedule_df.columns:
                schedule_df["EventDate"] = pd.to_datetime(schedule_df["EventDate"]).dt.date
            event_dates = schedule_df[schedule_df["RoundNumber"] == round_number]["EventDate"] if not schedule_df.empty else pd.Series(dtype=object)
            if not event_dates.empty and event_dates.iloc[0] < today:
                try:
                    with st.spinner("Loading actual results..."):
                        actual = _cached_load_race_results(year, round_number)
                    if not actual.empty:
                        actual = actual.rename(columns={"Position": "ActualPosition"})
                        pred_with_abbrev = pred_df.merge(
                            grid_df[["DriverName", "Abbreviation"]],
                            left_on="Driver",
                            right_on="DriverName",
                            how="left",
                        )
                        merged = pred_with_abbrev.merge(actual[["Abbreviation", "ActualPosition"]], on="Abbreviation", how="left")
                        merged["Error"] = merged["ActualPosition"] - merged["PredictedRank"]
                        st.subheader("Prediction vs actual")
                        mae = merged["Error"].abs().mean()
                        col_mae, col_roll = st.columns(2)
                        with col_mae:
                            st.metric("MAE (this race)", f"{mae:.2f} positions")
                        with col_roll:
                            if st.checkbox("Show model accuracy (last 5 races)", value=False, key="show_rolling_mae"):
                                with st.spinner("Computing rolling MAE..."):
                                    _maes, _mean = compute_rolling_mae(year, round_number, schedule_df, weather_str, last_n=5)
                                if _maes:
                                    st.metric("Rolling MAE (last 5 races)", f"{_mean:.2f} positions")
                                    st.caption(f"Rounds: {[r for r, _ in _maes]}")
                                else:
                                    st.caption("No past rounds to compute.")
                        st.dataframe(
                            merged[["PredictedRank", "Driver", "Team", "ActualPosition", "Error"]],
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.download_button(
                            "Download prediction vs actual (CSV)",
                            data=merged[["PredictedRank", "Driver", "Team", "ActualPosition", "Error"]].to_csv(index=False),
                            file_name=f"pred_vs_actual_{event_name.replace(' ', '_')}_R{round_number}.csv",
                            mime="text/csv",
                            key="dl_pred_actual",
                        )
                except Exception as e:
                    st.caption(f"Could not load actual results: {e}")

    st.caption(
        "Retrain race model: `python scripts/train_model.py` (trains both race and quali models)."
    )


if __name__ == "__main__":
    main()
