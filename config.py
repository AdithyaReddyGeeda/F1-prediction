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

# --- 2026 schedule fallback — replace/augment when official calendar is out ---
# Format: RoundNumber, EventName, EventDate, Country
SCHEDULE_2026: List[dict] = [
    {"RoundNumber": 1, "EventName": "Australian Grand Prix", "EventDate": date(2026, 3, 8), "Country": "Australia"},
    {"RoundNumber": 2, "EventName": "Chinese Grand Prix", "EventDate": date(2026, 4, 19), "Country": "China"},
    {"RoundNumber": 3, "EventName": "Miami Grand Prix", "EventDate": date(2026, 5, 3), "Country": "USA"},
    {"RoundNumber": 4, "EventName": "Emilia Romagna Grand Prix", "EventDate": date(2026, 5, 17), "Country": "Italy"},
    {"RoundNumber": 5, "EventName": "Monaco Grand Prix", "EventDate": date(2026, 5, 24), "Country": "Monaco"},
    {"RoundNumber": 6, "EventName": "Spanish Grand Prix", "EventDate": date(2026, 6, 7), "Country": "Spain"},
    {"RoundNumber": 7, "EventName": "Canadian Grand Prix", "EventDate": date(2026, 6, 21), "Country": "Canada"},
    {"RoundNumber": 8, "EventName": "Austrian Grand Prix", "EventDate": date(2026, 7, 5), "Country": "Austria"},
    {"RoundNumber": 9, "EventName": "British Grand Prix", "EventDate": date(2026, 7, 19), "Country": "UK"},
    {"RoundNumber": 10, "EventName": "Hungarian Grand Prix", "EventDate": date(2026, 8, 2), "Country": "Hungary"},
    {"RoundNumber": 11, "EventName": "Belgian Grand Prix", "EventDate": date(2026, 8, 30), "Country": "Belgium"},
    {"RoundNumber": 12, "EventName": "Dutch Grand Prix", "EventDate": date(2026, 9, 6), "Country": "Netherlands"},
    {"RoundNumber": 13, "EventName": "Italian Grand Prix", "EventDate": date(2026, 9, 20), "Country": "Italy"},
    {"RoundNumber": 14, "EventName": "Azerbaijan Grand Prix", "EventDate": date(2026, 10, 4), "Country": "Azerbaijan"},
    {"RoundNumber": 15, "EventName": "Singapore Grand Prix", "EventDate": date(2026, 10, 18), "Country": "Singapore"},
    {"RoundNumber": 16, "EventName": "United States Grand Prix", "EventDate": date(2026, 10, 25), "Country": "USA"},
    {"RoundNumber": 17, "EventName": "Mexico City Grand Prix", "EventDate": date(2026, 11, 1), "Country": "Mexico"},
    {"RoundNumber": 18, "EventName": "São Paulo Grand Prix", "EventDate": date(2026, 11, 8), "Country": "Brazil"},
    {"RoundNumber": 19, "EventName": "Las Vegas Grand Prix", "EventDate": date(2026, 11, 22), "Country": "USA"},
    {"RoundNumber": 20, "EventName": "Qatar Grand Prix", "EventDate": date(2026, 11, 29), "Country": "Qatar"},
    {"RoundNumber": 21, "EventName": "Saudi Arabian Grand Prix", "EventDate": date(2026, 12, 6), "Country": "Saudi Arabia"},
    {"RoundNumber": 22, "EventName": "Abu Dhabi Grand Prix", "EventDate": date(2026, 12, 13), "Country": "UAE"},
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
