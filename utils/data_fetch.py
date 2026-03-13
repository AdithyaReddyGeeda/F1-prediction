"""
Safe data fetching with fallback chain: 2026 FastF1 → 2025 FastF1 → hardcoded 2026 grid.
Handles first-race and early-season when current-year data is empty/sparse.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

# Import app-level modules (run from project root)
import sys
_path = Path(__file__).resolve().parent.parent
if str(_path) not in sys.path:
    sys.path.insert(0, str(_path))

from config import get_manual_grid, get_schedule_fallback, GRID_2026
from data import enable_fastf1_cache, get_event_schedule, load_race_results
from web_data import fetch_season_grid_from_web


def _ensure_cache():
    """Ensure FastF1 cache is enabled once."""
    cache_dir = _path / "fastf1_cache"
    cache_dir.mkdir(exist_ok=True)
    try:
        import fastf1
        fastf1.Cache.enable_cache(cache_dir)
    except Exception:
        pass


def safe_get_schedule(year: int, use_fallback_year: bool = False) -> pd.DataFrame:
    """
    Get event schedule for year. If use_fallback_year, try previous year when current fails/empty.
    For 2026 (and future seasons), use config fallback when FastF1 returns empty.
    """
    _ensure_cache()
    try:
        df = get_event_schedule(year)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    # Config fallback for 2026 / future when official calendar is in config
    fallback = get_schedule_fallback(year)
    if not fallback.empty:
        return fallback
    if use_fallback_year and year > 2014:
        try:
            return get_event_schedule(year - 1)
        except Exception:
            pass
    return pd.DataFrame()


def safe_get_drivers(year: int, use_fallback_year: bool = False) -> pd.DataFrame:
    """
    Get driver list (with TeamName, Abbreviation, DriverName) for year.
    Fallback: 1) web scrape, 2) last race of year from FastF1, 3) last race of year-1, 4) hardcoded 2026.
    """
    _ensure_cache()
    # 1) Try hardcoded 2026 first for 2026 (so we always have 22 drivers with real names)
    if year == 2026:
        manual = get_manual_grid(2026)
        if not manual.empty:
            return manual
    # 2) Web scrape (current grid)
    try:
        web = fetch_season_grid_from_web(year)
        if web is not None and not web.empty:
            if "DriverName" not in web.columns:
                web["DriverName"] = web.get("Abbreviation", pd.Series(dtype=object))
            return web
    except Exception:
        pass
    # 3) From last completed race of this year
    try:
        sched = get_event_schedule(year)
        if not sched.empty:
            last_round = int(sched["RoundNumber"].max())
            for r in range(last_round, 0, -1):
                try:
                    res = load_race_results(year, r)
                    if res.empty:
                        continue
                    meta = res[["DriverNumber", "Abbreviation", "TeamName"]].drop_duplicates("Abbreviation")
                    meta["DriverName"] = meta["Abbreviation"]  # FastF1 may not give full name
                    return meta
                except Exception:
                    continue
    except Exception:
        pass
    # 4) Fallback to previous year's last race
    if use_fallback_year and year > 2014:
        try:
            sched_prev = get_event_schedule(year - 1)
            if not sched_prev.empty:
                last_round = int(sched_prev["RoundNumber"].max())
                res = load_race_results(year - 1, last_round)
                if not res.empty:
                    meta = res[["DriverNumber", "Abbreviation", "TeamName"]].drop_duplicates("Abbreviation")
                    meta["DriverName"] = meta["Abbreviation"]
                    return meta
        except Exception:
            pass
    # 5) Hardcoded 2026
    if year >= 2026:
        return get_manual_grid(2026)
    return pd.DataFrame()


def safe_get_teams(year: int, use_fallback_year: bool = False) -> list:
    """Get list of team names for year (from drivers table)."""
    df = safe_get_drivers(year, use_fallback_year=use_fallback_year)
    if df.empty or "TeamName" not in df.columns:
        return []
    return df["TeamName"].dropna().unique().tolist()


def safe_get_race_data(
    year: int,
    round_number: Optional[int] = None,
    circuit_name: Optional[str] = None,
    use_fallback_year: bool = False,
) -> pd.DataFrame:
    """
    Load race results for year + round (or infer round from circuit).
    If use_fallback_year and no data for year, load same round from year-1.
    """
    _ensure_cache()
    if round_number is None and circuit_name:
        sched = safe_get_schedule(year, use_fallback_year=False)
        if sched.empty and use_fallback_year:
            sched = safe_get_schedule(year - 1, use_fallback_year=False)
        if not sched.empty:
            mask = sched["EventName"].str.contains(circuit_name, case=False, na=False)
            if mask.any():
                round_number = int(sched.loc[mask, "RoundNumber"].iloc[0])
    if round_number is None:
        return pd.DataFrame()
    try:
        res = load_race_results(year, round_number)
        if res is not None and not res.empty:
            return res
    except Exception:
        pass
    if use_fallback_year and year > 2014:
        try:
            return load_race_results(year - 1, round_number)
        except Exception:
            pass
    return pd.DataFrame()


def get_grid_for_prediction(
    year: int,
    up_to_round: int,
    force_2025_baseline: bool = False,
) -> pd.DataFrame:
    """
    Get the grid (22 drivers, 11 teams) to use for prediction.
    - If force_2025_baseline: use 2025 grid (from last 2025 race) then map to 2026 names if year=2026.
    - Else: current-year FastF1 / web / hardcoded 2026.
    """
    if force_2025_baseline and year >= 2026:
        # Use 2025 grid as baseline; we'll still show 2026 names from get_manual_grid(2026)
        drivers_2025 = safe_get_drivers(2025, use_fallback_year=False)
        if not drivers_2025.empty:
            grid_2026 = get_manual_grid(2026)
            if not grid_2026.empty:
                # Prefer 2026 list for display; 2025 has history for form
                return grid_2026
        return get_manual_grid(2026)
    # Normal path: current year (or 2026 hardcoded)
    if year == 2026 and up_to_round < 1:
        manual = get_manual_grid(2026)
        if not manual.empty:
            return manual
    try:
        from data import get_current_season_grid
        grid = get_current_season_grid(year, up_to_round=up_to_round, seasons_back=2)
        if not grid.empty:
            if "DriverName" not in grid.columns:
                grid["DriverName"] = grid.get("Abbreviation", pd.Series(dtype=object))
            return grid
    except Exception:
        pass
    return get_manual_grid(2026) if year >= 2026 else pd.DataFrame()
