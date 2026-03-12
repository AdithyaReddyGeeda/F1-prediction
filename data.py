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


def load_race_results(year: int, round_number: int) -> pd.DataFrame:
    """Load race results (finish positions) for a specific event. Empty DataFrame on failure.
    Includes Status when available for DNF handling in training."""
    try:
        session = fastf1.get_session(year, round_number, "R")
        session.load()
        results = session.results
        cols = ["DriverNumber", "Abbreviation", "TeamName", "Position"]
        if "Status" in results.columns:
            cols.append("Status")
        df = results[[c for c in cols if c in results.columns]].copy()
        df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def load_fp_deltas(year: int, round_number: int) -> pd.DataFrame:
    """
    Load FP1/FP2/FP3 best lap deltas to session fastest (seconds).
    Returns DataFrame with Year, Round, Abbreviation, FP1_delta, FP2_delta, FP3_delta (NaN if no data).
    """
    out = []
    for session_name in ("FP1", "FP2", "FP3"):
        try:
            session = fastf1.get_session(year, round_number, session_name)
            session.load()
            laps = session.laps
            if laps is None or laps.empty:
                continue
            # Best lap per driver
            best = laps.dropna(subset=["LapTime"]).groupby("DriverNumber").agg(
                LapTime=("LapTime", "min")
            ).reset_index()
            if best.empty:
                continue
            # Fastest in session
            fastest = best["LapTime"].min()
            if fastest is None or pd.isna(fastest):
                continue
            # Delta in seconds (timedelta -> float)
            def _to_sec(t):
                if t is None or pd.isna(t):
                    return 0.0
                if hasattr(t, "total_seconds"):
                    return t.total_seconds()
                return float(t)
            fastest_sec = _to_sec(fastest)
            best["delta_sec"] = best["LapTime"].apply(lambda x: _to_sec(x) - fastest_sec)
            drivers = session.results
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



