from __future__ import annotations

from pathlib import Path
from typing import Optional

import fastf1
import pandas as pd

from config import get_manual_grid
from web_data import fetch_season_grid_from_web


def enable_fastf1_cache(cache_dir: Optional[str] = "./fastf1_cache") -> None:
    """Enable FastF1 HTTP cache in a local directory."""
    path = Path(cache_dir)
    path.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(path)


def get_event_schedule(year: int) -> pd.DataFrame:
    """Return a cleaned event schedule for a given year. Empty DataFrame on API failure."""
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        df = schedule[["RoundNumber", "EventName", "EventDate", "Country"]].copy()
        df["EventDate"] = pd.to_datetime(df["EventDate"]).dt.date
        return df
    except Exception:
        return pd.DataFrame()


def load_race_weather(year: int, round_number: int) -> Optional[str]:
    """
    Load real weather for a race from FastF1 session.weather_data.
    Returns "Dry", "Wet", or "Rain" when available; None if not available.
    """
    try:
        session = fastf1.get_session(year, round_number, "R")
        session.load()
        w = getattr(session, "weather_data", None)
        if w is None or (hasattr(w, "empty") and w.empty):
            return None
        # FastF1 weather_data can have 'Rain' column (bool) or similar
        if hasattr(w, "columns") and "Rain" in w.columns:
            if w["Rain"].any() if hasattr(w["Rain"], "any") else bool(w["Rain"].iloc[0]):
                return "Rain"
        if hasattr(w, "columns") and "Wet" in w.columns:
            if w["Wet"].any() if hasattr(w["Wet"], "any") else bool(w["Wet"].iloc[0]):
                return "Wet"
        # Some APIs use 'Rainfall' or numeric rain indicator
        for col in ("Rainfall", "Humidity", "Precipitation"):
            if hasattr(w, "columns") and col in w.columns:
                ser = w[col]
                if ser.dtype in ("bool", "int", "float") and (ser.astype(float) > 0).any():
                    return "Wet"
        return "Dry"
    except Exception:
        return None


def fetch_results_from_jolpica(year: int, round_number: int) -> pd.DataFrame:
    """
    Fetch race results from the Jolpica API (fast, available ~30 min after race).
    Returns DataFrame with: Abbreviation, DriverName, TeamName, Position
    or empty DataFrame on failure.

    API endpoint: https://api.jolpi.ca/ergast/f1/{year}/{round}/results/
    """
    import requests

    url = f"https://api.jolpi.ca/ergast/f1/{year}/{round_number}/results/"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not races:
            return pd.DataFrame()
        results = races[0].get("Results", [])
        if not results:
            return pd.DataFrame()
        rows = []
        for r in results:
            driver = r.get("Driver", {})
            constructor = r.get("Constructor", {})
            rows.append({
                "Position": int(r.get("position", 99)),
                "Abbreviation": driver.get("code", "UNK").upper(),
                "DriverName": f"{driver.get('givenName','')} {driver.get('familyName','')}".strip(),
                "TeamName": constructor.get("name", "Unknown"),
                "Status": r.get("status", "Finished"),
            })
        df = pd.DataFrame(rows)
        df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def load_race_results(year: int, round_number: int) -> pd.DataFrame:
    """
    Load race results. For recent races (within 3 days of today),
    tries Jolpica API first (faster), then falls back to FastF1.
    """
    import datetime as dt

    # For very recent races, try Jolpica first (FastF1 has a data lag)
    try:
        from config import get_schedule_fallback
        sched = get_schedule_fallback(year)
        if not sched.empty:
            row = sched[sched["RoundNumber"] == round_number]
            if not row.empty:
                event_date = pd.to_datetime(row.iloc[0]["EventDate"]).date()
                days_since = (dt.date.today() - event_date).days
                if 0 <= days_since <= 3:
                    jolpica_df = fetch_results_from_jolpica(year, round_number)
                    if not jolpica_df.empty:
                        return jolpica_df
    except Exception:
        pass

    # Original FastF1 logic
    try:
        session = fastf1.get_session(year, round_number, "R")
        session.load()
        results = session.results
        cols = ["DriverNumber", "Abbreviation", "TeamName", "Position"]
        if "Status" in results.columns:
            cols.append("Status")
        if "Laps" in results.columns:
            cols.append("Laps")
        df = results[[c for c in cols if c in results.columns]].copy()
        df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
        if "Laps" in df.columns:
            df["Laps"] = pd.to_numeric(df["Laps"], errors="coerce")
        total_laps = getattr(session, "total_laps", None)
        if total_laps is None and "Laps" in df.columns:
            total_laps = df["Laps"].max()
        if total_laps is not None and not pd.isna(total_laps):
            df["RaceLaps"] = int(total_laps)
        return df
    except Exception:
        return pd.DataFrame()


def load_fp_deltas(year: int, round_number: int, max_retries: int = 2) -> pd.DataFrame:
    """
    Load FP1/FP2/FP3 best lap deltas to session fastest (seconds).
    Retries each session up to max_retries on failure. Returns DataFrame with
    Year, Round, Abbreviation, FP1_delta, FP2_delta, FP3_delta (no data → 0 in feature pipeline).
    """
    import time
    out = []
    for session_name in ("FP1", "FP2", "FP3"):
        laps = None
        session = None
        for attempt in range(max_retries + 1):
            try:
                session = fastf1.get_session(year, round_number, session_name)
                session.load()
                laps = session.laps
                break
            except Exception:
                if attempt < max_retries:
                    time.sleep(1.0)
                else:
                    laps = None
        if laps is None or (hasattr(laps, "empty") and laps.empty):
            continue
        try:
            # Best lap per driver
            best = laps.dropna(subset=["LapTime"]).groupby("DriverNumber").agg(
                LapTime=("LapTime", "min")
            ).reset_index()
            if best.empty:
                continue
            fastest = best["LapTime"].min()
            if fastest is None or pd.isna(fastest):
                continue
            def _to_sec(t):
                if t is None or pd.isna(t):
                    return 0.0
                if hasattr(t, "total_seconds"):
                    return t.total_seconds()
                return float(t)
            fastest_sec = _to_sec(fastest)
            best["delta_sec"] = best["LapTime"].apply(lambda x: _to_sec(x) - fastest_sec)
            drivers = session.results if session is not None else None
            if drivers is not None and not drivers.empty and "Abbreviation" in drivers.columns:
                best = best.merge(
                    drivers[["DriverNumber", "Abbreviation"]].drop_duplicates("DriverNumber"),
                    on="DriverNumber",
                    how="left",
                )
            else:
                best["Abbreviation"] = best["DriverNumber"].astype(str)
            best["Year"] = year
            best["Round"] = round_number
            best[session_name + "_delta"] = best["delta_sec"]
            out.append(best[["Year", "Round", "Abbreviation", session_name + "_delta"]])
        except Exception:
            continue
    if not out:
        return pd.DataFrame()
    merged = out[0]
    for d in out[1:]:
        merged = merged.merge(
            d,
            on=["Year", "Round", "Abbreviation"],
            how="outer",
        )
    return merged


def load_race_tyre_proxy(year: int, round_number: int) -> pd.DataFrame:
    """
    Load race lap data and compute per-driver tyre proxy (e.g. avg laps per stint).
    Returns DataFrame with Year, Round, Abbreviation, tyre_proxy (float). Empty on failure.
    """
    try:
        session = fastf1.get_session(year, round_number, "R")
        session.load()
        laps = getattr(session, "laps", None)
        if laps is None or (hasattr(laps, "empty") and laps.empty):
            return pd.DataFrame()
        if "Stint" not in laps.columns or "DriverNumber" not in laps.columns:
            return pd.DataFrame()
        stint_laps = laps.groupby(["DriverNumber", "Stint"]).size().reset_index(name="LapsInStint")
        avg_per_driver = stint_laps.groupby("DriverNumber")["LapsInStint"].mean().reset_index()
        avg_per_driver.columns = ["DriverNumber", "tyre_proxy"]
        results = getattr(session, "results", None)
        if results is not None and not results.empty and "Abbreviation" in results.columns:
            avg_per_driver = avg_per_driver.merge(
                results[["DriverNumber", "Abbreviation"]].drop_duplicates("DriverNumber"),
                on="DriverNumber",
                how="left",
            )
        else:
            avg_per_driver["Abbreviation"] = avg_per_driver["DriverNumber"].astype(str)
        avg_per_driver["Year"] = year
        avg_per_driver["Round"] = round_number
        return avg_per_driver[["Year", "Round", "Abbreviation", "tyre_proxy"]]
    except Exception:
        return pd.DataFrame()


def load_qualifying_results(year: int, round_number: int) -> pd.DataFrame:
    """Load qualifying results (positions 1–22) for a specific event. Empty DataFrame on failure."""
    try:
        session = fastf1.get_session(year, round_number, "Q")
        session.load()
        results = session.results
        df = results[["DriverNumber", "Abbreviation", "TeamName", "Position"]].copy()
        df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
        df = df.rename(columns={"Position": "QualiPosition"})
        return df
    except Exception:
        return pd.DataFrame()


def load_quali_gaps(year: int, round_number: int) -> pd.DataFrame:
    """
    Load each driver's best qualifying lap gap to pole (seconds).
    Uses best lap across all Q segments. Returns DataFrame:
    Year, Round, Abbreviation, quali_gap_to_pole (float seconds).
    Missing drivers get NaN (to be filled downstream).
    """
    try:
        session = fastf1.get_session(year, round_number, "Q")
        session.load()
        laps = session.laps
        if laps is None or (hasattr(laps, "empty") and laps.empty):
            return pd.DataFrame()
        best = laps.dropna(subset=["LapTime"]).groupby("DriverNumber").agg(
            BestLap=("LapTime", "min")
        ).reset_index()
        if best.empty:
            return pd.DataFrame()

        def _to_sec(t):
            return t.total_seconds() if hasattr(t, "total_seconds") else float(t)

        best["best_sec"] = best["BestLap"].apply(_to_sec)
        pole_time = best["best_sec"].min()
        best["quali_gap_to_pole"] = best["best_sec"] - pole_time

        results = session.results
        if results is not None and not results.empty and "Abbreviation" in results.columns:
            best = best.merge(
                results[["DriverNumber", "Abbreviation"]].drop_duplicates("DriverNumber"),
                on="DriverNumber",
                how="left",
            )
        else:
            best["Abbreviation"] = best["DriverNumber"].astype(str)
        best["Year"] = year
        best["Round"] = round_number
        return best[["Year", "Round", "Abbreviation", "quali_gap_to_pole"]]
    except Exception:
        return pd.DataFrame()


def get_current_season_grid(year: int, up_to_round: int, seasons_back: int = 2) -> pd.DataFrame:
    """
    Infer the current season grid (drivers + teams).

    Preference order:
    1. If an external web source is available and no race has run yet, use that.
    2. If a manual grid override exists for this year and no race has run yet, use that.
    3. Use the most recent completed race in the selected year (up to `up_to_round`).
    4. If no race has been run yet, fall back to the last race of a recent past season.
    """
    # 0) For 2026 (and future), prefer hardcoded grid when no race run yet — real 22 drivers/11 teams
    if year >= 2026 and up_to_round < 1:
        manual = get_manual_grid(2026)
        if not manual.empty:
            return manual
    if up_to_round < 1:
        # 1) Try external web data (official F1 site)
        try:
            web_grid = fetch_season_grid_from_web(year)
            if not web_grid.empty:
                if "DriverName" not in web_grid.columns:
                    web_grid = web_grid.copy()
                    web_grid["DriverName"] = web_grid.get("Abbreviation", "")
                return web_grid
        except Exception:
            pass
        # 2) Fallback: manual grid if configured
        manual = get_manual_grid(year)
        if not manual.empty:
            return manual
    # 2) Try this season up to the selected round
    if up_to_round >= 1:
        for r in range(up_to_round, 0, -1):
            try:
                df = load_race_results(year, r)
            except Exception:
                continue

            if df.empty:
                continue

            meta = (
                df[["DriverNumber", "Abbreviation", "TeamName"]]
                .dropna(subset=["Abbreviation"])
                .drop_duplicates(subset=["Abbreviation"], keep="last")
            )
            if not meta.empty:
                return meta

    # 3) Fall back to most recent race of previous seasons
    for prev_year in range(year - 1, max(year - seasons_back, 2014) - 1, -1):
        try:
            sched = get_event_schedule(prev_year)
        except Exception:
            continue

        if sched.empty:
            continue

        max_round = int(sched["RoundNumber"].max())
        try:
            df_prev = load_race_results(prev_year, max_round)
        except Exception:
            continue

        if df_prev.empty:
            continue

        meta_prev = (
            df_prev[["DriverNumber", "Abbreviation", "TeamName"]]
            .dropna(subset=["Abbreviation"])
            .drop_duplicates(subset=["Abbreviation"], keep="last")
        )
        if not meta_prev.empty:
            return meta_prev

    return pd.DataFrame(columns=["DriverNumber", "Abbreviation", "TeamName"])



