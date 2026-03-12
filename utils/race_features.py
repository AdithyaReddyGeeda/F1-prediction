"""
Race feature engineering for F1 finishing position prediction.
All features use only past data (no leakage): EWMA form, track-specific avg, quali strength,
driver-team synergy, weather deltas, momentum, circuit type, DNF imputation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Defaults for rookies / missing data
DEFAULT_AVG_POS = 12.0
DEFAULT_TRACK_AVG = 10.0
DEFAULT_SYNERGY = 10.0
DNF_IMPUTE_POSITION = 21

# Circuit name -> type for track characteristic dummies (simplified mapping)
CIRCUIT_TYPE_MAP = {
    "street": [
        "Monaco", "Baku", "Singapore", "Miami", "Las Vegas", "Australian", "Canadian",
        "Saudi", "Bahrain", "Abu Dhabi", "Qatar", "Azerbaijan",
    ],
    "high_speed": [
        "Monza", "Spa", "Sakhir", "Jeddah", "Silverstone", "Japanese", "Suzuka",
        "British", "Belgian", "Italian", "Bahrain",
    ],
    "technical": [
        "Hungaroring", "Zandvoort", "Spanish", "Barcelona", "Monaco", "Singapore",
        "Hungarian", "Dutch", "Spanish", "Portuguese", "Emilia", "Imola",
    ],
}


def _circuit_to_type(circuit: str) -> str:
    """Map circuit name to street / high_speed / technical (default technical)."""
    c = str(circuit).lower()
    for t, keywords in CIRCUIT_TYPE_MAP.items():
        if any(kw.lower() in c for kw in keywords):
            return t
    return "technical"


def impute_dnf_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute finish position for DNFs so target is usable.
    Expects Position (numeric) and optional Status. Sets Position to DNF_IMPUTE_POSITION
    when Status indicates DNF or Position is null/invalid.
    """
    df = df.copy()
    if "Status" in df.columns:
        # Common DNF-like statuses (FastF1 may use various strings)
        dnf_flags = df["Status"].astype(str).str.upper().str.contains(
            "ACCIDENT|COLLISION|DNF|RETIRED|WHEEL|ENGINE|GEARBOX|BRAKE|SUSPENSION|POWER|SPUN|DAMAGE",
            na=False,
            regex=True,
        )
        mask = dnf_flags & (df["Position"].isna() | (df["Position"] < 1) | (df["Position"] > 22))
        df.loc[mask, "Position"] = DNF_IMPUTE_POSITION
    # Any remaining null/invalid positions (no Status) -> impute
    bad = df["Position"].isna() | (df["Position"] < 1) | (df["Position"] > 22)
    df.loc[bad, "Position"] = DNF_IMPUTE_POSITION
    return df


def compute_ewma_form(
    df: pd.DataFrame,
    position_col: str = "Position",
    alpha: float = 0.4,
    min_periods: int = 1,
) -> pd.Series:
    """
    Per-driver EWMA of finishing position over time (last N races).
    Uses only past data: shift(1) then ewm. Returns series indexed as df.index.
    """
    df = df.sort_values(["Abbreviation", "Year", "Round"])
    out = pd.Series(index=df.index, dtype=float)
    for driver in df["Abbreviation"].unique():
        mask = df["Abbreviation"] == driver
        pos = df.loc[mask, position_col].astype(float).shift(1)
        ewm = pos.ewm(alpha=alpha, min_periods=min_periods, adjust=False).mean()
        out.loc[mask] = ewm.values
    return out.fillna(DEFAULT_AVG_POS)


def compute_constructor_ewma(
    df: pd.DataFrame,
    position_col: str = "Position",
    alpha: float = 0.4,
) -> pd.Series:
    """Per-constructor EWMA of finishing position per race (per driver row, team's form). Indexed as df.index."""
    df = df.sort_values(["Year", "Round"])
    out = pd.Series(index=df.index, dtype=float)
    for team in df["TeamName"].unique():
        mask = df["TeamName"] == team
        sub = df.loc[mask].sort_values(["Year", "Round"])
        pos = sub[position_col].astype(float).shift(1)
        ewm = pos.ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        out.loc[sub.index] = ewm.values
    return out.fillna(DEFAULT_AVG_POS)


def get_track_specific_avg(
    df: pd.DataFrame,
    circuit_col: str = "Circuit",
    position_col: str = "Position",
    max_visits: int = 5,
) -> tuple[pd.Series, pd.Series]:
    """
    Driver and constructor average position at this circuit (past visits only).
    Returns (driver_track_avg, constructor_track_avg) indexed as df.index.
    """
    df = df.sort_values(["Year", "Round"])
    driver_avg = pd.Series(index=df.index, dtype=float)
    team_avg = pd.Series(index=df.index, dtype=float)

    for i, row in df.iterrows():
        circ = row[circuit_col]
        drv = row["Abbreviation"]
        team = row["TeamName"]
        yr, rnd = row["Year"], row["Round"]
        # Past races at same circuit: strictly before (Year, Round)
        past_d = df[
            (df[circuit_col] == circ)
            & (df["Abbreviation"] == drv)
            & ((df["Year"] < yr) | ((df["Year"] == yr) & (df["Round"] < rnd)))
        ].sort_values(["Year", "Round"])
        past_t = df[
            (df[circuit_col] == circ)
            & (df["TeamName"] == team)
            & ((df["Year"] < yr) | ((df["Year"] == yr) & (df["Round"] < rnd)))
        ].sort_values(["Year", "Round"])
        if len(past_d) > 0:
            driver_avg.loc[i] = past_d[position_col].tail(max_visits).mean()
        else:
            driver_avg.loc[i] = DEFAULT_TRACK_AVG
        if len(past_t) > 0:
            team_best_per_race = past_t.groupby(["Year", "Round"])[position_col].min().tail(max_visits)
            team_avg.loc[i] = team_best_per_race.mean()
        else:
            team_avg.loc[i] = DEFAULT_TRACK_AVG

    return driver_avg.fillna(DEFAULT_TRACK_AVG), team_avg.fillna(DEFAULT_TRACK_AVG)


def get_driver_team_synergy(df: pd.DataFrame, position_col: str = "Position") -> pd.Series:
    """Historical average finish for this (driver, team) pair — past races only. Indexed as df.index."""
    df = df.sort_values(["Year", "Round"])
    out = pd.Series(index=df.index, dtype=float)
    for (driver, team), g in df.groupby(["Abbreviation", "TeamName"]):
        pos = g[position_col].astype(float).shift(1).expanding(min_periods=1).mean()
        out.loc[g.index] = pos.values
    return out.fillna(DEFAULT_SYNERGY)


def get_momentum_position_change(df: pd.DataFrame, position_col: str = "Position") -> pd.Series:
    """Position change vs previous race (current - previous). Positive = worse. No leakage: use shift(1)."""
    df = df.sort_values(["Abbreviation", "Year", "Round"])
    out = pd.Series(index=df.index, dtype=float)
    for driver in df["Abbreviation"].unique():
        mask = df["Abbreviation"] == driver
        pos = df.loc[mask, position_col].astype(float)
        prev = pos.shift(1)
        curr = pos
        out.loc[mask] = (curr - prev).values
    return out.fillna(0.0)


def get_driver_rain_delta(
    df: pd.DataFrame,
    weather_col: str = "Weather",
    position_col: str = "Position",
    past_only: bool = True,
) -> pd.Series:
    """
    Per-driver delta: avg position in wet/rain - avg in dry.
    past_only: use only strictly earlier (Year, Round) rows per driver to avoid leakage.
    """
    df = df.sort_values(["Year", "Round"])
    out = pd.Series(index=df.index, dtype=float)
    for i, row in df.iterrows():
        drv = row["Abbreviation"]
        yr, rnd = row["Year"], row["Round"]
        past = df[
            (df["Abbreviation"] == drv)
            & ((df["Year"] < yr) | ((df["Year"] == yr) & (df["Round"] < rnd)))
        ]
        if len(past) == 0:
            out.loc[i] = 0.0
            continue
        dry = past[past[weather_col] == "Dry"][position_col].mean()
        wet = past[past[weather_col].isin(["Wet", "Rain"])][position_col].mean()
        if pd.notna(dry) and pd.notna(wet):
            out.loc[i] = wet - dry
        else:
            out.loc[i] = 0.0
    return out


def get_relative_teammate_delta(df: pd.DataFrame, form_col: str = "RecentForm") -> pd.Series:
    """
    Per-row: driver's form minus teammate's form (same team, same race).
    Positive = this driver has been finishing worse than teammate. Uses form_col (e.g. RecentForm).
    """
    out = pd.Series(index=df.index, dtype=float)
    df = df.sort_values(["Year", "Round"])
    for (yr, rnd, team), g in df.groupby(["Year", "Round", "TeamName"]):
        if len(g) < 2:
            out.loc[g.index] = 0.0
            continue
        drivers = g["Abbreviation"].tolist()
        forms = g[form_col].values
        for i, idx in enumerate(g.index):
            # Teammate = other driver(s) in same race/team; use mean form of others
            other_forms = [forms[j] for j in range(len(forms)) if j != i]
            teammate_form = np.mean(other_forms) if other_forms else forms[i]
            out.loc[idx] = forms[i] - teammate_form
    return out.fillna(0.0)


def get_constructor_dnf_rate(
    df: pd.DataFrame,
    last_n: int = 10,
    position_col: str = "Position",
    status_col: Optional[str] = "Status",
) -> pd.Series:
    """
    Per-row: constructor's DNF rate in last N race entries (past only).
    DNF = Status indicates DNF, or Position > 20 / null. Returns fraction 0–1.
    """
    df = df.sort_values(["Year", "Round"])
    is_dnf = pd.Series(0.0, index=df.index)
    if status_col and status_col in df.columns:
        dnf_flags = df[status_col].astype(str).str.upper().str.contains(
            "ACCIDENT|COLLISION|DNF|RETIRED|WHEEL|ENGINE|GEARBOX|BRAKE|SUSPENSION|POWER|SPUN|DAMAGE",
            na=False,
            regex=True,
        )
        is_dnf = dnf_flags.astype(float)
    pos = df[position_col].astype(float)
    is_dnf = is_dnf.where(is_dnf == 1.0, ((pos > 20) | pos.isna()).astype(float))
    out = pd.Series(index=df.index, dtype=float)
    for team in df["TeamName"].unique():
        mask = df["TeamName"] == team
        sub = df.loc[mask].sort_values(["Year", "Round"])
        for i, (idx, row) in enumerate(sub.iterrows()):
            past = sub.iloc[:i]
            if len(past) == 0:
                out.loc[idx] = 0.0
                continue
            last = past.tail(last_n)
            rate = is_dnf.loc[last.index].mean()
            out.loc[idx] = rate
    return out.fillna(0.0)


# Circuit abrasion proxy: high tyre degradation tracks (0=low, 0.5=med, 1=high)
CIRCUIT_ABRASION = {
    "high": ["Bahrain", "Abu Dhabi", "Barcelona", "Spanish", "Silverstone", "British", "Suzuka", "Japanese"],
    "medium": ["Monza", "Italian", "Spa", "Belgian", "Monaco", "Miami", "Hungaroring", "Hungarian", "Zandvoort", "Dutch"],
}
def get_circuit_abrasion_proxy(circuit: str) -> float:
    c = str(circuit).lower()
    for level, keywords in CIRCUIT_ABRASION.items():
        if any(kw.lower() in c for kw in keywords):
            return 1.0 if level == "high" else 0.5
    return 0.0


def get_tyre_life_penalty_proxy(laps_fraction: float = 0.5, decay_laps: float = 30.0) -> float:
    """Non-linear tyre life penalty: 1 - exp(-laps_fraction * 50 / decay_laps). Use default 0.5 if unknown."""
    import math
    x = laps_fraction * 50.0 / max(decay_laps, 1.0)
    return 1.0 - math.exp(-min(x, 5.0))


def get_driver_dnf_rate(
    df: pd.DataFrame,
    last_n: int = 10,
    position_col: str = "Position",
    status_col: Optional[str] = "Status",
) -> pd.Series:
    """Per-driver DNF rate in last N races (past only). Fraction 0–1."""
    df = df.sort_values(["Year", "Round"])
    is_dnf = pd.Series(0.0, index=df.index)
    if status_col and status_col in df.columns:
        dnf_flags = df[status_col].astype(str).str.upper().str.contains(
            "ACCIDENT|COLLISION|DNF|RETIRED|WHEEL|ENGINE|GEARBOX|BRAKE|SUSPENSION|POWER|SPUN|DAMAGE",
            na=False,
            regex=True,
        )
        is_dnf = dnf_flags.astype(float)
    pos = df[position_col].astype(float)
    is_dnf = is_dnf.where(is_dnf == 1.0, ((pos > 20) | pos.isna()).astype(float))
    out = pd.Series(index=df.index, dtype=float)
    for driver in df["Abbreviation"].unique():
        mask = df["Abbreviation"] == driver
        sub = df.loc[mask].sort_values(["Year", "Round"])
        for i, idx in enumerate(sub.index):
            past = sub.iloc[:i]
            if len(past) == 0:
                out.loc[idx] = 0.0
                continue
            last = past.tail(last_n)
            out.loc[idx] = is_dnf.loc[last.index].mean()
    return out.fillna(0.0)


def add_circuit_type_dummies(df: pd.DataFrame, circuit_col: str = "Circuit") -> pd.DataFrame:
    """Add circuit_type_street, circuit_type_high_speed, circuit_type_technical (0/1)."""
    df = df.copy()
    t = df[circuit_col].map(_circuit_to_type)
    df["circuit_type_street"] = (t == "street").astype(float)
    df["circuit_type_high_speed"] = (t == "high_speed").astype(float)
    df["circuit_type_technical"] = (t == "technical").astype(float)
    return df


def build_race_feature_df(
    race_df: pd.DataFrame,
    quali_df: Optional[pd.DataFrame] = None,
    weather_per_race: Optional[dict] = None,
    fp_df: Optional[pd.DataFrame] = None,
    ewma_alpha: float = 0.4,
    default_avg_pos: float = DEFAULT_AVG_POS,
) -> pd.DataFrame:
    """
    Build full feature DataFrame for race model from race_df (Year, Round, Circuit, Abbreviation, TeamName, Position).
    Optionally merge quali from quali_df (Year, Round, Abbreviation, QualiPosition).
    weather_per_race: optional dict (Year, Round) -> "Dry"|"Wet"|"Rain". If None, random 50/30/20 per race.
    Returns race_df with: GridPosition, RecentForm, ConstructorEwma, track_avg_*, driver_team_synergy,
    momentum, Weather, driver_rain_delta, circuit_type_*, QualiPosition, is_rain, grid_pos_x_rain.
    """
    df = race_df.dropna(subset=["Position", "Abbreviation"]).copy()
    df = df.sort_values(["Year", "Round"])
    df = impute_dnf_positions(df)

    # Grid position: previous round finish or 10
    df["GridPosition"] = 10.0
    for (y, r), g in df.groupby(["Year", "Round"]):
        prev = df[(df["Year"] == y) & (df["Round"] == r - 1)]
        if not prev.empty:
            prev_pos = prev.set_index("Abbreviation")["Position"]
            idx = g.index
            df.loc[idx, "GridPosition"] = df.loc[idx, "Abbreviation"].map(prev_pos).fillna(10).values

    # EWMA form (replaces simple rolling recent form)
    df["RecentForm"] = compute_ewma_form(df, position_col="Position", alpha=ewma_alpha)
    df["ConstructorEwma"] = compute_constructor_ewma(df, position_col="Position", alpha=ewma_alpha)

    # Track-specific
    track_d, track_t = get_track_specific_avg(df, circuit_col="Circuit", position_col="Position", max_visits=5)
    df["track_avg_driver"] = track_d.values
    df["track_avg_team"] = track_t.values

    # Driver-team synergy
    df["driver_team_synergy"] = get_driver_team_synergy(df, position_col="Position").values

    # Relative teammate: driver form minus teammate form (same race/team)
    df["teammate_delta"] = get_relative_teammate_delta(df, form_col="RecentForm").values

    # Constructor DNF rate (past last_n races)
    df["constructor_dnf_rate"] = get_constructor_dnf_rate(
        df, last_n=10, position_col="Position", status_col="Status" if "Status" in df.columns else None
    ).values

    # Driver DNF rate (past last_n races)
    df["driver_dnf_rate"] = get_driver_dnf_rate(
        df, last_n=10, position_col="Position", status_col="Status" if "Status" in df.columns else None
    ).values

    # Tyre degradation proxies: circuit abrasion, tyre life penalty (default 0.5), driver tyre management placeholder
    df["circuit_abrasion_proxy"] = df["Circuit"].map(get_circuit_abrasion_proxy).values
    df["tyre_life_penalty_proxy"] = get_tyre_life_penalty_proxy(0.5)
    df["driver_tyre_management_proxy"] = 0.0  # placeholder; can be filled from stint data later

    # Interaction: form * teammate_delta (captures under/overperformance vs teammate)
    df["form_x_teammate_delta"] = (df["RecentForm"].astype(float) * df["teammate_delta"].astype(float)).values

    # Momentum (position change)
    df["momentum"] = get_momentum_position_change(df, position_col="Position").values

    # Weather assignment: one per race (Year, Round) to avoid leakage when computing driver_rain_delta
    if weather_per_race is not None:
        df["Weather"] = df.apply(lambda r: weather_per_race.get((int(r["Year"]), int(r["Round"])), "Dry"), axis=1)
    else:
        rng = np.random.default_rng(42)
        keys = df[["Year", "Round"]].drop_duplicates()
        w = rng.random(len(keys))
        assign = dict(zip(zip(keys["Year"], keys["Round"]), np.where(w < 0.50, "Dry", np.where(w < 0.80, "Wet", "Rain"))))
        df["Weather"] = df.apply(lambda r: assign.get((int(r["Year"]), int(r["Round"])), "Dry"), axis=1)

    # Driver rain delta (after weather assigned)
    df["driver_rain_delta"] = get_driver_rain_delta(df, weather_col="Weather", position_col="Position").values

    # Circuit type dummies
    df = add_circuit_type_dummies(df, circuit_col="Circuit")

    # Quali position merge (same Year, Round, Abbreviation)
    if quali_df is not None and not quali_df.empty and "QualiPosition" in quali_df.columns:
        q = quali_df[["Year", "Round", "Abbreviation", "QualiPosition"]].drop_duplicates(
            subset=["Year", "Round", "Abbreviation"], keep="last"
        )
        df = df.merge(
            q,
            on=["Year", "Round", "Abbreviation"],
            how="left",
            suffixes=("", "_quali"),
        )
        if "QualiPosition" not in df.columns:
            df["QualiPosition"] = df["GridPosition"]  # fallback
        df["QualiPosition"] = df["QualiPosition"].fillna(df["GridPosition"])
    else:
        df["QualiPosition"] = df["GridPosition"].values

    # Practice session deltas (merge when available)
    if fp_df is not None and not fp_df.empty:
        fp_cols = [c for c in ["FP1_delta", "FP2_delta", "FP3_delta"] if c in fp_df.columns]
        if fp_cols:
            df = df.merge(
                fp_df[["Year", "Round", "Abbreviation"] + fp_cols].drop_duplicates(
                    subset=["Year", "Round", "Abbreviation"], keep="last"
                ),
                on=["Year", "Round", "Abbreviation"],
                how="left",
            )
            for c in fp_cols:
                if c not in df.columns:
                    df[c] = 0.0
                else:
                    df[c] = df[c].fillna(0.0)
    for c in ["FP1_delta", "FP2_delta", "FP3_delta"]:
        if c not in df.columns:
            df[c] = 0.0

    # Interaction and rain flag
    df["is_rain"] = (df["Weather"] == "Rain").astype(float)
    df["grid_pos_x_rain"] = df["GridPosition"] * df["is_rain"]

    return df
