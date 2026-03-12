from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from data import get_current_season_grid, get_event_schedule, load_race_results
from features import aggregate_driver_stats, restrict_to_current_grid


def _collect_recent_races(
    year: int,
    round_number: int,
    history_races: int,
    seasons_back: int,
) -> List[Tuple[int, int, pd.DataFrame]]:
    """
    Collect up to `history_races` most recent completed races,
    going backwards from the selected race across this season and previous ones.
    """
    history_frames: List[Tuple[int, int, pd.DataFrame]] = []
    races_collected = 0

    for y in range(year, max(year - seasons_back, 2014) - 1, -1):
        if y == year:
            start_round = round_number - 1
            if start_round < 1:
                continue
            candidate_rounds = range(start_round, 0, -1)
        else:
            sched = get_event_schedule(y)
            if sched.empty:
                continue
            max_round = int(sched["RoundNumber"].max())
            candidate_rounds = range(max_round, 0, -1)

        for r in candidate_rounds:
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

    return history_frames


def build_prediction_for_event(
    year: int,
    round_number: int,
    history_races: int,
    seasons_back: int = 2,
) -> pd.DataFrame:
    """
    Build predictions using up to `history_races` most recent completed races
    across this and recent seasons, but:
    - Restrict to the current season's grid (drivers present in this year).
    - Use the current season's driver/team pairing for metadata.
    """
    # Determine current grid, allowing fallback to previous seasons if no race yet
    current_grid_meta = get_current_season_grid(
        year,
        up_to_round=round_number - 1,
        seasons_back=seasons_back,
    )
    if current_grid_meta.empty:
        return pd.DataFrame(columns=["Driver", "Team", "AvgFinish", "RacesUsed"])

    history_frames = _collect_recent_races(
        year=year,
        round_number=round_number,
        history_races=history_races,
        seasons_back=seasons_back,
    )

    history = restrict_to_current_grid(history_frames, current_grid_meta=current_grid_meta)

    def _neutral_from_grid(meta: pd.DataFrame) -> pd.DataFrame:
        grid_sorted = meta.reset_index(drop=True)
        grid_sorted.insert(0, "PredictedRank", np.arange(1, len(grid_sorted) + 1))
        # Prefer full driver name when available
        if "DriverName" in grid_sorted.columns:
            grid_sorted["Driver"] = grid_sorted["DriverName"].fillna(grid_sorted["Abbreviation"])
        else:
            grid_sorted["Driver"] = grid_sorted["Abbreviation"]
        grid_sorted.rename(columns={"TeamName": "Team"}, inplace=True)
        grid_sorted["AvgFinish"] = np.nan
        grid_sorted["RacesUsed"] = 0
        return grid_sorted[
            ["PredictedRank", "Driver", "Team", "AvgFinish", "RacesUsed", "DriverNumber"]
        ]

    # If we have no historical data after filtering, fall back to a neutral
    # prediction based purely on the current grid order.
    if history.empty:
        return _neutral_from_grid(current_grid_meta)

    summary = aggregate_driver_stats(history, current_grid_meta=current_grid_meta)
    if summary.empty:
        return _neutral_from_grid(current_grid_meta)

    summary.insert(0, "PredictedRank", np.arange(1, len(summary) + 1))
    # Prefer full driver name when available
    if "DriverName" in summary.columns:
        summary["Driver"] = summary["DriverName"].fillna(summary["Abbreviation"])
    else:
        summary["Driver"] = summary["Abbreviation"]
    summary.rename(columns={"TeamName": "Team"}, inplace=True)
    return summary

