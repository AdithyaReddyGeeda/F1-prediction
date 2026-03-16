"""
App config: 2026 grid and schedule (fallback when API/scraping fails), constants.
Update SCHEDULE_2026 when the official F1 calendar is published.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List

import pandas as pd


# --- 2026 confirmed grid (11 teams, 22 drivers) — used when FastF1/web return empty ---
# Format: DriverNumber, Abbreviation (for FastF1 matching), DriverName (full), TeamName
GRID_2026: List[dict] = [
    {"DriverNumber": 63, "Abbreviation": "RUS", "DriverName": "George Russell", "TeamName": "Mercedes"},
    {"DriverNumber": 47, "Abbreviation": "ANT", "DriverName": "Kimi Antonelli", "TeamName": "Mercedes"},
    {"DriverNumber": 16, "Abbreviation": "LEC", "DriverName": "Charles Leclerc", "TeamName": "Ferrari"},
    {"DriverNumber": 44, "Abbreviation": "HAM", "DriverName": "Lewis Hamilton", "TeamName": "Ferrari"},
    {"DriverNumber": 4, "Abbreviation": "NOR", "DriverName": "Lando Norris", "TeamName": "McLaren"},
    {"DriverNumber": 81, "Abbreviation": "PIA", "DriverName": "Oscar Piastri", "TeamName": "McLaren"},
    {"DriverNumber": 1, "Abbreviation": "VER", "DriverName": "Max Verstappen", "TeamName": "Red Bull Racing"},
    {"DriverNumber": 39, "Abbreviation": "HAD", "DriverName": "Isack Hadjar", "TeamName": "Red Bull Racing"},
    {"DriverNumber": 31, "Abbreviation": "OCO", "DriverName": "Esteban Ocon", "TeamName": "Haas"},
    {"DriverNumber": 87, "Abbreviation": "BEA", "DriverName": "Oliver Bearman", "TeamName": "Haas"},
    {"DriverNumber": 40, "Abbreviation": "LAW", "DriverName": "Liam Lawson", "TeamName": "Racing Bulls"},
    {"DriverNumber": 45, "Abbreviation": "LIN", "DriverName": "Arvid Lindblad", "TeamName": "Racing Bulls"},
    {"DriverNumber": 27, "Abbreviation": "HUL", "DriverName": "Nico Hulkenberg", "TeamName": "Audi"},
    {"DriverNumber": 28, "Abbreviation": "BOR", "DriverName": "Gabriel Bortoleto", "TeamName": "Audi"},
    {"DriverNumber": 10, "Abbreviation": "GAS", "DriverName": "Pierre Gasly", "TeamName": "Alpine"},
    {"DriverNumber": 99, "Abbreviation": "COL", "DriverName": "Franco Colapinto", "TeamName": "Alpine"},
    {"DriverNumber": 55, "Abbreviation": "SAI", "DriverName": "Carlos Sainz", "TeamName": "Williams"},
    {"DriverNumber": 23, "Abbreviation": "ALB", "DriverName": "Alexander Albon", "TeamName": "Williams"},
    {"DriverNumber": 11, "Abbreviation": "PER", "DriverName": "Sergio Perez", "TeamName": "Cadillac"},
    {"DriverNumber": 77, "Abbreviation": "BOT", "DriverName": "Valtteri Bottas", "TeamName": "Cadillac"},
    {"DriverNumber": 14, "Abbreviation": "ALO", "DriverName": "Fernando Alonso", "TeamName": "Aston Martin"},
    {"DriverNumber": 18, "Abbreviation": "STR", "DriverName": "Lance Stroll", "TeamName": "Aston Martin"},
]

# Rookies / new team drivers: use default avg position for "recent form" when no history
ROOKIE_DEFAULT_AVG_POSITION = 12.0

# Engine supplier by team (for feature). Used when inferring from team name.
# Extend for new seasons; unknown teams default to "Other".
ENGINE_BY_TEAM: Dict[str, str] = {
    "Mercedes": "Mercedes",
    "Ferrari": "Ferrari",
    "Red Bull Racing": "Honda RBPT",
    "Red Bull": "Honda RBPT",
    "McLaren": "Mercedes",
    "Aston Martin": "Honda RBPT",
    "Alpine": "Renault",
    "Williams": "Mercedes",
    "Haas": "Ferrari",
    "Racing Bulls": "Honda RBPT",
    "AlphaTauri": "Honda RBPT",
    "Alfa Romeo": "Ferrari",
    "Sauber": "Ferrari",
    "Audi": "Audi",
    "Cadillac": "Cadillac",
}

# Circuit metadata: laps and track length (km) for each 2026 event.
# Used to estimate total race distance and time for future races.
CIRCUIT_METADATA: dict = {
    "Australian Grand Prix":    {"laps": 58, "length_km": 5.278},
    "Chinese Grand Prix":       {"laps": 56, "length_km": 5.451},
    "Bahrain Grand Prix":       {"laps": 57, "length_km": 5.412},
    "Saudi Arabian Grand Prix": {"laps": 50, "length_km": 6.174},
    "Miami Grand Prix":         {"laps": 57, "length_km": 5.412},
    "Emilia Romagna Grand Prix": {"laps": 63, "length_km": 4.909},
    "Monaco Grand Prix":        {"laps": 78, "length_km": 3.337},
    "Spanish Grand Prix":       {"laps": 66, "length_km": 4.657},
    "Canadian Grand Prix":      {"laps": 70, "length_km": 4.361},
    "Austrian Grand Prix":      {"laps": 71, "length_km": 4.318},
    "British Grand Prix":       {"laps": 52, "length_km": 5.891},
    "Belgian Grand Prix":       {"laps": 44, "length_km": 7.004},
    "Hungarian Grand Prix":     {"laps": 70, "length_km": 4.381},
    "Dutch Grand Prix":         {"laps": 72, "length_km": 4.259},
    "Italian Grand Prix":       {"laps": 53, "length_km": 5.793},
    "Azerbaijan Grand Prix":    {"laps": 51, "length_km": 6.003},
    "Singapore Grand Prix":     {"laps": 62, "length_km": 4.940},
    "United States Grand Prix": {"laps": 56, "length_km": 5.513},
    "Mexico City Grand Prix":   {"laps": 71, "length_km": 4.304},
    "São Paulo Grand Prix":     {"laps": 71, "length_km": 4.309},
    "Las Vegas Grand Prix":     {"laps": 50, "length_km": 6.201},
    "Qatar Grand Prix":         {"laps": 57, "length_km": 5.380},
    "Abu Dhabi Grand Prix":     {"laps": 58, "length_km": 5.281},
}

# --- 2026 schedule fallback (race day = Sunday). Replace when official calendar is published. ---
# Format: RoundNumber, EventName, EventDate, Country
SCHEDULE_2026: List[dict] = [
    {"RoundNumber": 1,  "EventName": "Australian Grand Prix",      "EventDate": date(2026, 3, 16),  "Country": "Australia"},
    {"RoundNumber": 2,  "EventName": "Chinese Grand Prix",         "EventDate": date(2026, 3, 23),  "Country": "China"},
    {"RoundNumber": 3,  "EventName": "Bahrain Grand Prix",         "EventDate": date(2026, 4, 6),   "Country": "Bahrain"},
    {"RoundNumber": 4,  "EventName": "Saudi Arabian Grand Prix",   "EventDate": date(2026, 4, 20),  "Country": "Saudi Arabia"},
    {"RoundNumber": 5,  "EventName": "Miami Grand Prix",           "EventDate": date(2026, 5, 4),   "Country": "USA"},
    {"RoundNumber": 6,  "EventName": "Emilia Romagna Grand Prix",  "EventDate": date(2026, 5, 24),  "Country": "Italy"},
    {"RoundNumber": 7,  "EventName": "Monaco Grand Prix",          "EventDate": date(2026, 5, 31),  "Country": "Monaco"},
    {"RoundNumber": 8,  "EventName": "Spanish Grand Prix",         "EventDate": date(2026, 6, 7),   "Country": "Spain"},
    {"RoundNumber": 9,  "EventName": "Canadian Grand Prix",        "EventDate": date(2026, 6, 21),  "Country": "Canada"},
    {"RoundNumber": 10, "EventName": "Austrian Grand Prix",        "EventDate": date(2026, 7, 5),   "Country": "Austria"},
    {"RoundNumber": 11, "EventName": "British Grand Prix",         "EventDate": date(2026, 7, 19),  "Country": "UK"},
    {"RoundNumber": 12, "EventName": "Belgian Grand Prix",         "EventDate": date(2026, 7, 26),  "Country": "Belgium"},
    {"RoundNumber": 13, "EventName": "Hungarian Grand Prix",       "EventDate": date(2026, 8, 2),   "Country": "Hungary"},
    {"RoundNumber": 14, "EventName": "Dutch Grand Prix",           "EventDate": date(2026, 8, 30),  "Country": "Netherlands"},
    {"RoundNumber": 15, "EventName": "Italian Grand Prix",         "EventDate": date(2026, 9, 6),   "Country": "Italy"},
    {"RoundNumber": 16, "EventName": "Azerbaijan Grand Prix",      "EventDate": date(2026, 9, 20),  "Country": "Azerbaijan"},
    {"RoundNumber": 17, "EventName": "Singapore Grand Prix",       "EventDate": date(2026, 10, 4),  "Country": "Singapore"},
    {"RoundNumber": 18, "EventName": "United States Grand Prix",   "EventDate": date(2026, 10, 18), "Country": "USA"},
    {"RoundNumber": 19, "EventName": "Mexico City Grand Prix",     "EventDate": date(2026, 11, 1),  "Country": "Mexico"},
    {"RoundNumber": 20, "EventName": "São Paulo Grand Prix",       "EventDate": date(2026, 11, 8),  "Country": "Brazil"},
    {"RoundNumber": 21, "EventName": "Las Vegas Grand Prix",       "EventDate": date(2026, 11, 22), "Country": "USA"},
    {"RoundNumber": 22, "EventName": "Qatar Grand Prix",           "EventDate": date(2026, 11, 29), "Country": "Qatar"},
    {"RoundNumber": 23, "EventName": "Abu Dhabi Grand Prix",       "EventDate": date(2026, 12, 6),  "Country": "UAE"},
]


def get_schedule_fallback(year: int) -> pd.DataFrame:
    """Return fallback schedule for year when FastF1 returns empty (e.g. 2026 before API update)."""
    if year == 2026:
        return pd.DataFrame(SCHEDULE_2026)
    return pd.DataFrame(columns=["RoundNumber", "EventName", "EventDate", "Country"])


# Optional manual grid overrides for other years (e.g. future seasons before API updates)
MANUAL_GRIDS: Dict[int, List[dict]] = {
    2026: GRID_2026,
}


def get_manual_grid(year: int) -> pd.DataFrame:
    """Return hardcoded grid for a season if configured (e.g. 2026)."""
    rows = MANUAL_GRIDS.get(year)
    if not rows:
        return pd.DataFrame(columns=["DriverNumber", "Abbreviation", "DriverName", "TeamName"])
    return pd.DataFrame(rows)
