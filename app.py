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
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import GRID_2026, CIRCUIT_METADATA
from inference import predict_finishing_order, get_inference_warnings
from quali_inference import predict_quali_order
from data import enable_fastf1_cache, get_event_schedule, load_race_results, load_race_time_and_fastlap
from time_inference import predict_race_time, predict_fastest_lap
from utils.data_fetch import safe_get_drivers, safe_get_schedule
from utils.circuit_geo import get_circuit_track_data

TEAM_COLORS = {
    "Mercedes": "#00D2BE", "Ferrari": "#E8002D", "Red Bull Racing": "#3671C6",
    "McLaren": "#FF8000", "Aston Martin": "#358C75", "Alpine": "#FF87BC",
    "Williams": "#64C4FF", "Racing Bulls": "#6692FF", "Haas": "#B6BABD",
    "Audi": "#B6BABD", "Cadillac": "#FFFFFF", "Sauber": "#52E252",
}

TEAM_NAME_ALIASES = {
    "Red Bull": "Red Bull Racing",
    "Kick Sauber": "Sauber",
    "Stake F1 Team Kick Sauber": "Sauber",
    "Alpine F1 Team": "Alpine",
    "Haas F1 Team": "Haas",
    "RB F1 Team": "Racing Bulls",
    "Visa Cash App RB": "Racing Bulls",
    "MoneyGram Haas F1 Team": "Haas",
    "BWT Alpine F1 Team": "Alpine",
}


def normalize_team_name(name: str) -> str:
    return TEAM_NAME_ALIASES.get(str(name).strip(), str(name).strip())


def _sec_to_str(sec: float, include_hours: bool = True) -> str:
    """Format seconds as H:MM:SS.mmm or M:SS.mmm."""
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    if include_hours:
        return f"{h}:{m:02d}:{s:06.3f}"
    return f"{m}:{s:06.3f}"


@st.cache_data(ttl=600)
def _cached_load_race_results(year: int, round_number: int, refresh_ts: float = 0):
    """Cached race results. Pass refresh_ts from session state to force refresh."""
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


def render_prediction_table(pred_df, grid_df=None):
    """Render prediction table with team color dots and medal-style positions."""
    rows_html = ""
    for _, row in pred_df.iterrows():
        team = str(row.get("Team", ""))
        color = TEAM_COLORS.get(team, "#888888")
        rank = int(row["PredictedRank"])
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"P{rank}"
        rows_html += f"""
        <tr style='border-bottom:1px solid #222;'>
            <td style='padding:8px 12px; font-weight:bold; color:{"#ffd700" if rank<=3 else "#fff"};
                       font-size:1.1rem;'>{medal}</td>
            <td style='padding:8px 4px;'>
                <span style='display:inline-block; width:12px; height:12px; border-radius:50%;
                             background:{color}; margin-right:8px;'></span>
                {row.get("Driver", "")}
            </td>
            <td style='padding:8px 12px; color:#aaa;'>{team}</td>
        </tr>"""
    st.markdown(f"""
    <table style='width:100%; border-collapse:collapse; background:#111; border-radius:8px;
                  overflow:hidden; font-family:Arial,sans-serif;'>
        <thead>
            <tr style='background:#1a1a1a;'>
                <th style='padding:10px 12px; color:#e10600; text-align:left; text-transform:uppercase;
                           letter-spacing:1px; font-size:0.8rem;'>POS</th>
                <th style='padding:10px 12px; color:#e10600; text-align:left; text-transform:uppercase;
                           letter-spacing:1px; font-size:0.8rem;'>DRIVER</th>
                <th style='padding:10px 12px; color:#e10600; text-align:left; text-transform:uppercase;
                           letter-spacing:1px; font-size:0.8rem;'>TEAM</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)


def render_vs_actual_table(merged_df):
    """Render Prediction vs Actual table with color-coded errors and team dots."""
    rows_html = ""
    for _, row in merged_df.iterrows():
        err = row.get("Error", 0)
        err_val = float(err) if pd.notna(err) else 0
        color = "#09ab3b" if abs(err_val) <= 2 else "#ffd600" if abs(err_val) <= 5 else "#e10600"
        arrow = "▲" if err_val < 0 else "▼" if err_val > 0 else "●"
        team = str(row.get("Team", ""))
        dot_color = TEAM_COLORS.get(team, "#888")
        actual_pos = row.get("ActualPosition")
        actual_str = f"P{int(actual_pos)}" if pd.notna(actual_pos) else "–"
        rows_html += f"""
        <tr style='border-bottom:1px solid #222;'>
            <td style='padding:8px 12px; color:#fff;'>P{int(row["PredictedRank"])}</td>
            <td style='padding:8px 4px;'>
                <span style='display:inline-block;width:10px;height:10px;border-radius:50%;
                             background:{dot_color};margin-right:8px;'></span>
                {row.get("Driver","")}
            </td>
            <td style='padding:8px 12px; color:#aaa;'>{team}</td>
            <td style='padding:8px 12px; color:#fff;'>{actual_str}</td>
            <td style='padding:8px 12px; color:{color}; font-weight:bold;'>
                {arrow} {abs(err_val):.0f}
            </td>
        </tr>"""
    st.markdown(f"""
    <table style='width:100%;border-collapse:collapse;background:#111;border-radius:8px;
                  overflow:hidden;font-family:Arial,sans-serif;'>
        <thead>
            <tr style='background:#1a1a1a;'>
                <th style='padding:10px 12px;color:#e10600;text-align:left;font-size:0.8rem;
                           text-transform:uppercase;letter-spacing:1px;'>PRED</th>
                <th style='padding:10px 12px;color:#e10600;text-align:left;font-size:0.8rem;
                           text-transform:uppercase;letter-spacing:1px;'>DRIVER</th>
                <th style='padding:10px 12px;color:#e10600;text-align:left;font-size:0.8rem;
                           text-transform:uppercase;letter-spacing:1px;'>TEAM</th>
                <th style='padding:10px 12px;color:#e10600;text-align:left;font-size:0.8rem;
                           text-transform:uppercase;letter-spacing:1px;'>ACTUAL</th>
                <th style='padding:10px 12px;color:#e10600;text-align:left;font-size:0.8rem;
                           text-transform:uppercase;letter-spacing:1px;'>DIFF</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)


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

    st.markdown("""
<style>
/* Dark F1-style theme */
[data-testid="stAppViewContainer"] { background-color: #0e0e0e; }
[data-testid="stSidebar"] { background-color: #1a1a1a; border-right: 2px solid #e10600; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] label { color: #ffffff !important; }

/* Titles */
h1 { color: #e10600 !important; font-family: 'Formula1', 'Arial Black', sans-serif !important;
     letter-spacing: 2px; text-transform: uppercase; }
h2, h3 { color: #ffffff !important; font-family: Arial, sans-serif; }

/* Tabs */
[data-baseweb="tab"] { color: #aaaaaa !important; font-weight: bold;
                        text-transform: uppercase; letter-spacing: 1px; }
[aria-selected="true"] { color: #e10600 !important;
                          border-bottom: 3px solid #e10600 !important; }

/* Buttons */
[data-testid="baseButton-primary"], .stButton > button {
    background-color: #e10600 !important; color: white !important;
    border: none !important; border-radius: 4px !important;
    font-weight: bold !important; text-transform: uppercase !important;
    letter-spacing: 1px !important; padding: 0.5rem 1.5rem !important; }
.stButton > button:hover { background-color: #ff2a1f !important; }

/* DataFrames */
[data-testid="stDataFrame"] { border: 1px solid #333 !important; border-radius: 6px; }
thead tr th { background-color: #1a1a1a !important; color: #e10600 !important;
              font-weight: bold !important; text-transform: uppercase !important; }
tbody tr:nth-child(even) { background-color: #161616 !important; }
tbody tr:hover { background-color: #252525 !important; }

/* Metric cards */
[data-testid="metric-container"] { background-color: #1a1a1a; border-left: 4px solid #e10600;
    border-radius: 6px; padding: 1rem; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 2rem !important; }
[data-testid="stMetricLabel"] { color: #aaaaaa !important; text-transform: uppercase !important;
    letter-spacing: 1px !important; }

/* Info / warning boxes */
[data-testid="stAlert"] { border-radius: 6px !important; }
.stInfo { border-left: 4px solid #0068c9 !important; background-color: #0a1a2e !important; }
.stSuccess { border-left: 4px solid #09ab3b !important; background-color: #061a0e !important; }
.stWarning { border-left: 4px solid #ffd600 !important; background-color: #1a1600 !important; }

/* Selectbox & radio */
[data-baseweb="select"] { background-color: #1a1a1a !important; }
[data-testid="stRadio"] label { color: #ffffff !important; }

/* Expander */
[data-testid="stExpander"] { background-color: #1a1a1a !important;
    border: 1px solid #333 !important; border-radius: 6px !important; }
summary { color: #e10600 !important; font-weight: bold !important; }

/* Spinner */
[data-testid="stSpinner"] > div { border-top-color: #e10600 !important; }
</style>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div style='display:flex; align-items:center; gap:16px; margin-bottom:8px;'>
    <span style='font-size:2.8rem;'>🏎️</span>
    <div>
        <h1 style='margin:0; font-size:2rem;'>F1 RACE PREDICTOR</h1>
        <p style='color:#888; margin:0; font-size:0.9rem; letter-spacing:2px;'>
            POWERED BY XGBOOST + FASTF1 · {year} SEASON
        </p>
    </div>
</div>
<hr style='border-color:#e10600; margin-bottom:1.5rem;'>
""", unsafe_allow_html=True)

    enable_fastf1_cache()

    with st.sidebar:
        st.markdown("## ⚙️ CONFIG")
        st.markdown("---")
        year = st.number_input("📅 Season", min_value=2020, max_value=2030, value=2026, step=1)
        st.markdown("**🌦️ Weather condition**")
        weather_str = st.radio("", options=["☀️ Dry", "🌧️ Wet", "⛈️ Rain"], index=0, label_visibility="collapsed")
        weather_str = weather_str.split()[-1]  # strip emoji
        st.markdown("---")
        st.markdown("""
<div style='color:#888; font-size:0.75rem; text-align:center; margin-top:1rem;'>
F1 data via FastF1 API<br>Model: XGBoost · MAE ~6 pos
</div>""", unsafe_allow_html=True)

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
        today = dt.date.today()
        event_labels = []
        for _, r in schedule_df.iterrows():
            date_val = r["EventDate"]
            if not isinstance(date_val, dt.date):
                date_val = pd.to_datetime(date_val).date()
            days_away = (date_val - today).days
            status = " ✅" if date_val < today else " 🔜" if days_away <= 7 else ""
            event_labels.append(
                f"Round {int(r['RoundNumber'])} — {r['EventName']} ({r['EventDate']}){status}"
            )
        selected = st.selectbox("Select race", event_labels, index=0)
        round_number = int(selected.split()[1])
        event_name = schedule_df.loc[schedule_df["RoundNumber"] == round_number, "EventName"].iloc[0]
        event_row = schedule_df[schedule_df["RoundNumber"].astype(int) == round_number].iloc[0]
        event_date = event_row["EventDate"]
        event_date_d = event_date if isinstance(event_date, dt.date) else pd.to_datetime(event_date).date()
        today_card = dt.date.today()
        is_past = event_date_d < today_card
        status_badge = (
            "<span style='background:#09ab3b;color:white;padding:2px 8px;border-radius:10px;"
            "font-size:0.75rem;font-weight:bold;'>COMPLETED</span>"
            if is_past else
            "<span style='background:#e10600;color:white;padding:2px 8px;border-radius:10px;"
            "font-size:0.75rem;font-weight:bold;'>UPCOMING</span>"
        )
        st.markdown(f"""
<div style='background:#1a1a1a; border:1px solid #333; border-radius:8px;
            padding:1rem; margin:0.5rem 0;'>
    <div style='display:flex; justify-content:space-between; align-items:center;'>
        <div>
            <div style='color:#e10600; font-weight:bold; font-size:1.1rem;'>
                {event_name}
            </div>
            <div style='color:#888; font-size:0.85rem;'>Round {round_number} · {event_date}</div>
        </div>
        {status_badge}
    </div>
</div>
""", unsafe_allow_html=True)

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
                quali_show = quali_df.copy()
                quali_show["PredictedRank"] = quali_show["PredictedQualiPos"]
                render_prediction_table(quali_show)
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
                _pq_show = _pq.copy()
                _pq_show["PredictedRank"] = _pq_show["PredictedQualiPos"] if "PredictedQualiPos" in _pq.columns else _pq_show["GridPosition"]
                _pq_show["Driver"] = _pq_show.get("DriverName", _pq_show.get("Driver", ""))
                _pq_show["Team"] = _pq_show.get("TeamName", _pq_show.get("Team", ""))
                render_prediction_table(_pq_show)
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
                        render_prediction_table(pred_from_quali)
                        # Include extra useful columns in CSV when available
                        export_cols = []
                        for col in ["PredictedRank", "Driver", "Team", "GridPosition", "QualiPosition"]:
                            if col in pred_from_quali.columns:
                                export_cols.append(col)
                        export_df = pred_from_quali[export_cols] if export_cols else pred_from_quali
                        st.download_button(
                            "Download race prediction (CSV)",
                            data=export_df.to_csv(index=False),
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
            model_source = result[2] if isinstance(result, tuple) and len(result) > 2 else "xgboost"

            if pred_df.empty:
                st.warning("No prediction produced.")
            else:
                if model_source == "xgboost":
                    st.success("🤖 XGBoost model")
                elif model_source == "heuristic_build":
                    st.warning("⚠️ Heuristic fallback (build_prediction)")
                else:
                    st.warning("⚠️ Heuristic fallback (form only)")
                st.subheader("Predicted finishing order")
                render_prediction_table(pred_df, grid_df)
                # Include extra useful columns in CSV when available
                export_cols = []
                for col in ["PredictedRank", "Driver", "Team", "GridPosition", "QualiPosition"]:
                    if col in pred_df.columns:
                        export_cols.append(col)
                export_df = pred_df[export_cols] if export_cols else pred_df
                st.download_button(
                    "Download race prediction (CSV)",
                    data=export_df.to_csv(index=False),
                    file_name=f"race_prediction_{event_name.replace(' ', '_')}_R{round_number}.csv",
                    mime="text/csv",
                    key="dl_race_pred",
                )
                if debug_info:
                    with st.expander("Debug: race weather encoding and features"):
                        for w in get_inference_warnings():
                            st.warning(w)
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
                fig = go.Figure()
                for _, row in pred_df.sort_values("PredictedRank").iterrows():
                    team = str(row.get("Team", ""))
                    color = TEAM_COLORS.get(team, "#888888")
                    fig.add_trace(go.Bar(
                        y=[row["Driver"]],
                        x=[23 - int(row["PredictedRank"])],
                        orientation="h",
                        marker_color=color,
                        name=team,
                        showlegend=False,
                        text=f"P{int(row['PredictedRank'])}",
                        textposition="inside",
                        hovertemplate=f"<b>{row['Driver']}</b><br>Predicted: P{int(row['PredictedRank'])}<br>Team: {team}<extra></extra>"
                    ))
                fig.update_layout(
                    paper_bgcolor="#0e0e0e", plot_bgcolor="#111111",
                    font=dict(color="#ffffff", family="Arial"),
                    height=500, margin=dict(l=120, r=20, t=40, b=20),
                    title=dict(text="Predicted Race Order", font=dict(color="#e10600", size=16)),
                    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                    yaxis=dict(autorange="reversed", gridcolor="#222"),
                    bargap=0.3,
                )
                st.plotly_chart(fig, use_container_width=True)

                # ── Race time and fastest lap predictions ──────────────────────────
                st.markdown("---")
                st.markdown("### ⏱️ Race Time & Fastest Lap")
                circuit_meta = CIRCUIT_METADATA.get(event_name, {})
                total_laps_est = circuit_meta.get("laps", 60)
                pole_time_sec = None
                try:
                    fl_live = load_race_time_and_fastlap(year, round_number)
                    if fl_live.get("pole_time_sec"):
                        pole_time_sec = fl_live["pole_time_sec"]
                except Exception:
                    pass

                col_time, col_fl = st.columns(2)
                with col_time:
                    st.markdown("#### 🏁 Predicted Race Duration")
                    time_pred = predict_race_time(
                        circuit=event_name,
                        total_laps=total_laps_est,
                        pole_time_sec=pole_time_sec,
                        weather_str=weather_str,
                    )
                    if time_pred:
                        length_km = circuit_meta.get("length_km", "?")
                        total_km = round(total_laps_est * (length_km if isinstance(length_km, (int, float)) else 5), 1)
                        st.markdown(f"""
                        <div style='background:#1a1a1a; border-left:4px solid #e10600;
                                    border-radius:6px; padding:1rem;'>
                            <div style='font-size:2rem; font-weight:bold; color:#fff;'>
                                {time_pred.get("predicted_str", "—")}
                            </div>
                            <div style='color:#888; font-size:0.85rem; margin-top:4px;'>
                                Range: {_sec_to_str(time_pred.get("low_sec", 0))}
                                — {_sec_to_str(time_pred.get("high_sec", 0))}
                            </div>
                            <div style='color:#555; font-size:0.75rem; margin-top:4px;'>
                                {total_laps_est} laps ·
                                {length_km} km/lap ·
                                {total_km} km total
                                · via {time_pred.get("method", "—")}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if weather_str != "Dry":
                            pct = "13" if weather_str == "Rain" else "7"
                            st.caption(f"⚠️ {weather_str} conditions — estimate includes ~{pct}% pace reduction")
                    else:
                        st.info("Pole time not yet available — run Qualifying prediction first for better estimate.")

                with col_fl:
                    st.markdown("#### ⚡ Predicted Fastest Lap")
                    fl_pred = predict_fastest_lap(
                        circuit=event_name,
                        grid_df=grid_df,
                        pole_time_sec=pole_time_sec,
                        weather_str=weather_str,
                    )
                    fl_time_str = "—"
                    if fl_pred:
                        fl_driver = fl_pred.get("driver", "—")
                        fl_name = fl_pred.get("driver_name", fl_driver)
                        fl_time_str = fl_pred.get("predicted_time_str", "—")
                        fl_probs = fl_pred.get("probabilities", {})
                        st.markdown(f"""
                        <div style='background:#1a1a1a; border-left:4px solid #ffd700;
                                    border-radius:6px; padding:1rem;'>
                            <div style='font-size:1.4rem; font-weight:bold; color:#ffd700;'>
                                {fl_name}
                            </div>
                            <div style='font-size:1.8rem; font-weight:bold; color:#fff; margin-top:4px;'>
                                {fl_time_str}
                            </div>
                            <div style='color:#555; font-size:0.75rem; margin-top:4px;'>
                                via {fl_pred.get("method", "—")}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if fl_probs:
                            st.markdown("**Top 5 probabilities:**")
                            for drv, prob in list(fl_probs.items())[:5]:
                                bar_pct = int(prob * 100)
                                st.markdown(f"""
                                <div style='display:flex; align-items:center; gap:8px; margin:3px 0;'>
                                    <span style='color:#fff; width:40px; font-size:0.85rem;'>{drv}</span>
                                    <div style='background:#333; border-radius:3px; height:16px; flex:1;'>
                                        <div style='background:#e10600; height:100%; width:{bar_pct}%;
                                                    border-radius:3px;'></div>
                                    </div>
                                    <span style='color:#aaa; font-size:0.8rem; width:35px;'>{bar_pct}%</span>
                                </div>""", unsafe_allow_html=True)

            today = dt.date.today()
            if "EventDate" in schedule_df.columns:
                schedule_df["EventDate"] = pd.to_datetime(schedule_df["EventDate"]).dt.date
            event_dates = schedule_df[schedule_df["RoundNumber"] == round_number]["EventDate"] if not schedule_df.empty else pd.Series(dtype=object)
            if not event_dates.empty and event_dates.iloc[0] < today:
                col_refresh, _ = st.columns([1, 4])
                with col_refresh:
                    force_refresh = st.button("🔄 Refresh results", key="refresh_actual")
                if force_refresh:
                    st.session_state["actual_refresh_ts"] = dt.datetime.now().timestamp()
                refresh_ts = st.session_state.get("actual_refresh_ts", 0)
                try:
                    with st.spinner("Loading actual results..."):
                        actual = _cached_load_race_results(year, round_number, refresh_ts)
                    if not actual.empty:
                        if "TeamName" in actual.columns:
                            actual["TeamName"] = actual["TeamName"].map(normalize_team_name)
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
                                    mae_df = pd.DataFrame({"Round": [r for r, _ in _maes], "MAE": [m for _, m in _maes]})
                                    chart = alt.Chart(mae_df).mark_line(point=True).encode(
                                        x=alt.X("Round:O", title="Round"),
                                        y=alt.Y("MAE:Q", title="MAE (positions)", scale=alt.Scale(zero=False)),
                                        tooltip=["Round", alt.Tooltip("MAE:Q", format=".2f")],
                                    ).properties(title=f"Prediction accuracy — last {len(_maes)} races", height=200)
                                    st.altair_chart(chart, use_container_width=True)
                                    st.caption(f"Mean MAE over last {len(_maes)} races: {_mean:.2f} positions")
                                else:
                                    st.caption("No past rounds to compute.")
                        render_vs_actual_table(merged)
                        st.download_button(
                            "Download prediction vs actual (CSV)",
                            data=merged[["PredictedRank", "Driver", "Team", "ActualPosition", "Error"]].to_csv(index=False),
                            file_name=f"pred_vs_actual_{event_name.replace(' ', '_')}_R{round_number}.csv",
                            mime="text/csv",
                            key="dl_pred_actual",
                        )
                        # Actual vs predicted for race time and fastest lap
                        try:
                            actual_tfl = load_race_time_and_fastlap(year, round_number)
                            if actual_tfl and (time_pred or fl_pred):
                                st.markdown("**Actual vs Predicted (Race Time & Fastest Lap):**")
                                c1, c2 = st.columns(2)
                                with c1:
                                    actual_time = actual_tfl.get("winner_time_sec")
                                    if actual_time is not None and time_pred:
                                        err_s = abs(actual_time - time_pred["predicted_sec"])
                                        st.metric(
                                            "Race time error",
                                            f"{err_s:.1f}s",
                                            delta=f"Predicted: {time_pred['predicted_str']} · Actual: {_sec_to_str(actual_time)}"
                                        )
                                with c2:
                                    actual_fl_driver = actual_tfl.get("fastest_lap_driver", "")
                                    actual_fl_time = actual_tfl.get("fastest_lap_time_sec")
                                    correct = "✅" if actual_fl_driver == fl_pred.get("driver") else "❌"
                                    if actual_fl_time is not None:
                                        st.metric(
                                            f"Fastest lap {correct}",
                                            actual_fl_driver,
                                            delta=f"Time: {_sec_to_str(actual_fl_time, False)} · Predicted: {fl_time_str}"
                                        )
                        except Exception:
                            pass
                except Exception as e:
                    days_since = (today - pd.to_datetime(event_dates.iloc[0]).date()).days if not event_dates.empty else 99
                    if days_since == 0:
                        st.info(
                            "🏁 **Race just finished!** Results are usually available within 30–60 minutes. "
                            "Try refreshing the page shortly. FastF1 data can take up to 6 hours.",
                            icon="⏳"
                        )
                    elif days_since <= 1:
                        st.warning(
                            f"⚠️ Could not fetch results yet (FastF1 data lag). "
                            f"Try again in a few hours. Error: `{e}`"
                        )
                    else:
                        st.error(f"Could not load actual results: `{e}`")

    st.caption(
        "Retrain race model: `python scripts/train_model.py` (trains both race and quali models)."
    )


if __name__ == "__main__":
    main()
