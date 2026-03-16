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


def load_clean_air_pace(year: int, round_number: int, min_gap_ahead: float = 3.0) -> pd.DataFrame:
    """
    Per-driver average lap time when running in clean air
    (gap to car ahead > min_gap_ahead seconds).
    Excludes pit laps, safety car laps, and lapped traffic.
    Returns: Year, Round, Abbreviation, clean_air_pace_sec (float)
    Empty DataFrame on failure.
    """
    try:
        session = fastf1.get_session(year, round_number, "R")
        session.load()
        laps = session.laps.copy()

        def to_sec(t):
            if t is None or not hasattr(t, "total_seconds"):
                return None
            return t.total_seconds()

        if "PitInTime" in laps.columns and "PitOutTime" in laps.columns:
            laps = laps[laps["PitInTime"].isna() & laps["PitOutTime"].isna()]
        laps = laps.dropna(subset=["LapTime"])
        laps["lap_sec"] = laps["LapTime"].apply(to_sec)
        laps = laps.dropna(subset=["lap_sec"])
        laps = laps[laps["lap_sec"] > 40]

        if "GapToLeader" in laps.columns:
            laps = laps[laps["GapToLeader"] > min_gap_ahead]

        if laps.empty:
            return pd.DataFrame()

        best_per_driver = laps.groupby("DriverNumber")["lap_sec"].min()
        laps = laps.merge(
            best_per_driver.rename("best_lap"),
            left_on="DriverNumber", right_index=True, how="left",
        )
        laps = laps[laps["lap_sec"] <= laps["best_lap"] * 1.07]

        avg = (
            laps.groupby("DriverNumber")["lap_sec"]
            .mean()
            .reset_index()
            .rename(columns={"lap_sec": "clean_air_pace_sec"})
        )

        results = session.results
        if results is not None and not results.empty and "Abbreviation" in results.columns:
            avg = avg.merge(
                results[["DriverNumber", "Abbreviation"]].drop_duplicates("DriverNumber"),
                on="DriverNumber", how="left",
            )
        else:
            avg["Abbreviation"] = avg["DriverNumber"].astype(str)

        avg["Year"] = year
        avg["Round"] = round_number
        return avg[["Year", "Round", "Abbreviation", "clean_air_pace_sec"]]

    except Exception:
        return pd.DataFrame()


def load_quali_sector_times(year: int, round_number: int) -> pd.DataFrame:
    """
    Per-driver best sector times from qualifying (gap to session best, seconds).
    Uses best sector across all Q1/Q2/Q3 attempts independently.
    Returns: Year, Round, Abbreviation,
             s1_gap, s2_gap, s3_gap, total_sector_gap,
             s1_pct, s2_pct, s3_pct
    Empty DataFrame on failure.
    """
    try:
        session = fastf1.get_session(year, round_number, "Q")
        session.load()
        laps = session.laps.copy()
        laps = laps.dropna(subset=["Sector1Time", "Sector2Time", "Sector3Time"])

        def to_sec(t):
            return t.total_seconds() if hasattr(t, "total_seconds") else None

        laps["s1_sec"] = laps["Sector1Time"].apply(to_sec)
        laps["s2_sec"] = laps["Sector2Time"].apply(to_sec)
        laps["s3_sec"] = laps["Sector3Time"].apply(to_sec)
        laps = laps.dropna(subset=["s1_sec", "s2_sec", "s3_sec"])
        laps = laps[(laps["s1_sec"] > 5) & (laps["s2_sec"] > 5) & (laps["s3_sec"] > 5)]

        best = laps.groupby("DriverNumber").agg(
            best_s1=("s1_sec", "min"),
            best_s2=("s2_sec", "min"),
            best_s3=("s3_sec", "min"),
        ).reset_index()

        best["s1_gap"] = best["best_s1"] - best["best_s1"].min()
        best["s2_gap"] = best["best_s2"] - best["best_s2"].min()
        best["s3_gap"] = best["best_s3"] - best["best_s3"].min()
        best["total_sector_gap"] = best["s1_gap"] + best["s2_gap"] + best["s3_gap"]

        best["lap_total"] = best["best_s1"] + best["best_s2"] + best["best_s3"]
        best["s1_pct"] = best["best_s1"] / best["lap_total"]
        best["s2_pct"] = best["best_s2"] / best["lap_total"]
        best["s3_pct"] = best["best_s3"] / best["lap_total"]

        results = session.results
        if results is not None and not results.empty and "Abbreviation" in results.columns:
            best = best.merge(
                results[["DriverNumber", "Abbreviation"]].drop_duplicates("DriverNumber"),
                on="DriverNumber", how="left",
            )
        else:
            best["Abbreviation"] = best["DriverNumber"].astype(str)

        best["Year"] = year
        best["Round"] = round_number

        cols = ["Year", "Round", "Abbreviation", "s1_gap", "s2_gap", "s3_gap", "total_sector_gap", "s1_pct", "s2_pct", "s3_pct"]
        return best[cols]

    except Exception:
        return pd.DataFrame()


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


def load_race_time_and_fastlap(year: int, round_number: int) -> dict:
    """
    Extract winner total race time, fastest lap time, and fastest lap driver
    from a completed race session.

    Returns dict with keys:
        winner_time_sec     : float  - winner's total race time in seconds
        fastest_lap_time_sec: float  - fastest lap time in seconds
        fastest_lap_driver  : str    - Abbreviation of fastest lap setter
        fastest_lap_no      : int    - lap number fastest lap was set on
        total_laps          : int    - total race laps
        pole_time_sec       : float  - pole position lap time in seconds (from Q session)
    Returns empty dict on failure.
    """
    result = {}
    try:
        session = fastf1.get_session(year, round_number, "R")
        session.load()
        res = session.results

        # Winner time
        winner_row = res[pd.to_numeric(res["Position"], errors="coerce") == 1]
        if not winner_row.empty:
            t = winner_row.iloc[0].get("Time")
            if t is not None and hasattr(t, "total_seconds"):
                result["winner_time_sec"] = t.total_seconds()

        # Fastest lap: try FastestLapTime column first
        if "FastestLapTime" in res.columns:
            fl_valid = res["FastestLapTime"].notna()
            if fl_valid.any():
                fl_row = res[fl_valid].iloc[0]
                flt = fl_row.get("FastestLapTime")
                if flt is not None and hasattr(flt, "total_seconds"):
                    result["fastest_lap_time_sec"] = flt.total_seconds()
                result["fastest_lap_driver"] = str(fl_row.get("Abbreviation", ""))
                fln = fl_row.get("FastestLapNo") or fl_row.get("FastestLap")
                if fln is not None and not isinstance(fln, str):
                    result["fastest_lap_no"] = int(fln)
        else:
            # Fallback: compute from laps data
            laps = session.laps
            if laps is not None and not laps.empty:
                laps = laps.dropna(subset=["LapTime"])
            if laps is not None and not laps.empty:
                fastest_idx = laps["LapTime"].idxmin()
                fl_lap = laps.loc[fastest_idx]
                result["fastest_lap_time_sec"] = fl_lap["LapTime"].total_seconds()
                drv_num = fl_lap.get("DriverNumber")
                if res is not None and "Abbreviation" in res.columns and "DriverNumber" in res.columns:
                    abbrev_map = res.set_index("DriverNumber")["Abbreviation"].to_dict()
                    result["fastest_lap_driver"] = str(abbrev_map.get(drv_num, ""))
                else:
                    result["fastest_lap_driver"] = str(drv_num)
                result["fastest_lap_no"] = int(fl_lap.get("LapNumber", 0))

        # Total laps
        total = getattr(session, "total_laps", None)
        if total is None and "Laps" in res.columns:
            total = int(res["Laps"].max())
        if total:
            result["total_laps"] = int(total)

    except Exception:
        pass

    # Pole time from qualifying
    try:
        q_session = fastf1.get_session(year, round_number, "Q")
        q_session.load()
        q_laps = q_session.laps.dropna(subset=["LapTime"])
        if not q_laps.empty:
            pole_time = q_laps["LapTime"].min()
            result["pole_time_sec"] = pole_time.total_seconds()
    except Exception:
        pass

    return result


def load_historical_race_times(start_year: int = 2020, end_year: int = 2025) -> pd.DataFrame:
    """
    Load winner race time, fastest lap time, fastest lap driver for all historical races.
    Returns DataFrame: Year, Round, Circuit, winner_time_sec, fastest_lap_time_sec,
                       fastest_lap_driver, fastest_lap_no, total_laps, pole_time_sec
    """
    rows = []
    for year in range(start_year, end_year + 1):
        try:
            sched = get_event_schedule(year)
            if sched.empty:
                from config import get_schedule_fallback
                sched = get_schedule_fallback(year)
            if sched.empty:
                continue
            for _, row in sched.iterrows():
                r = int(row["RoundNumber"])
                circuit = str(row.get("EventName", ""))
                d = load_race_time_and_fastlap(year, r)
                if d:
                    d["Year"] = year
                    d["Round"] = r
                    d["Circuit"] = circuit
                    rows.append(d)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


