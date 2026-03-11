from __future__ import annotations

from pathlib import Path
from typing import Optional

import fastf1
import pandas as pd


def enable_fastf1_cache(cache_dir: Optional[str] = "./fastf1_cache") -> None:
    """Enable FastF1 HTTP cache in a local directory."""
    path = Path(cache_dir)
    path.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(path)


def get_event_schedule(year: int) -> pd.DataFrame:
    """Return a cleaned event schedule for a given year."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    df = schedule[["RoundNumber", "EventName", "EventDate", "Country"]].copy()
    df["EventDate"] = pd.to_datetime(df["EventDate"]).dt.date
    return df


def load_race_results(year: int, round_number: int) -> pd.DataFrame:
    """Load race results (finish positions) for a specific event."""
    session = fastf1.get_session(year, round_number, "R")
    session.load()
    results = session.results
    df = results[["DriverNumber", "Abbreviation", "TeamName", "Position"]].copy()
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    return df

