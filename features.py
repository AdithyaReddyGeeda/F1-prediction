from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from config import ROOKIE_DEFAULT_AVG_POSITION


def restrict_to_current_grid(
    history_frames: List[Tuple[int, int, pd.DataFrame]],
    current_grid_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Given a list of (year, round, df) frames and a current‑season grid,
    return a single DataFrame filtered to drivers present in that grid.
    """
    if not history_frames or current_grid_meta.empty:
        return pd.DataFrame()

    combined = pd.concat([df for _, _, df in history_frames], ignore_index=True)
    combined = combined.dropna(subset=["Position"])

    if combined.empty:
        return pd.DataFrame()

    current_drivers = set(current_grid_meta["Abbreviation"].dropna().unique())
    return combined[combined["Abbreviation"].isin(current_drivers)]


def aggregate_driver_stats(history: pd.DataFrame, current_grid_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate race history into per‑driver summary stats and merge with
    current‑season metadata (driver number, team).
    Expects history columns: Abbreviation, Position.
    Expects meta columns: Abbreviation, DriverNumber, TeamName.
    """
    if history.empty or current_grid_meta.empty:
        return pd.DataFrame()

    stats = (
        history.groupby("Abbreviation")
        .agg(
            AvgFinish=("Position", "mean"),
            RacesUsed=("Position", "count"),
        )
        .reset_index()
    )

    meta_cols = ["Abbreviation", "DriverNumber", "TeamName"]
    if "DriverName" in current_grid_meta.columns:
        meta_cols.append("DriverName")
    summary = current_grid_meta[meta_cols].merge(
        stats,
        on="Abbreviation",
        how="left",
    )
    summary["AvgFinish"] = summary["AvgFinish"].fillna(ROOKIE_DEFAULT_AVG_POSITION)
    summary["RacesUsed"] = summary["RacesUsed"].fillna(0)
    summary = summary.sort_values("AvgFinish", ascending=True).reset_index(drop=True)
    return summary


