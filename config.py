"""
App config: 2026 grid (fallback when API/scraping fails), constants.
"""
from __future__ import annotations

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
