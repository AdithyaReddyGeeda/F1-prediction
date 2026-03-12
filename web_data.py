from __future__ import annotations

import re
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup


def _safe_get(url: str, timeout: int = 10) -> Optional[str]:
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def fetch_season_grid_from_web(year: int) -> pd.DataFrame:
    """
    Best‑effort fetch of the F1 grid (drivers + teams) from the official site.

    Implementation notes:
    - Uses https://www.formula1.com/en/drivers.html which lists the current grid.
    - This reflects the *current* season; for future seasons it will effectively
      use the latest published grid once the site updates.
    - Driver numbers and abbreviations are inferred heuristically from the card text.
    """
    html = _safe_get("https://www.formula1.com/en/drivers.html")
    if html is None:
        return pd.DataFrame(columns=["DriverNumber", "Abbreviation", "TeamName"])

    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select("a.f1-driver-listing--item")
    if not cards:
        # Fallback selector in case the class changes slightly
        cards = soup.select("a.driver, a.f1-driver-card")

    rows = []
    for card in cards:
        try:
            name_el = card.select_one(".f1-driver-listing--name") or card.select_one(".driver-name")
            team_el = card.select_one(".f1-driver-listing--team") or card.select_one(".driver-team")
            number_el = card.select_one(".f1-driver-listing--number") or card.select_one(".driver-number")

            if not name_el or not team_el:
                continue

            name_text = name_el.get_text(strip=True)
            team_text = team_el.get_text(strip=True)
            num_text = number_el.get_text(strip=True) if number_el else ""

            # Abbreviation: take first 3 uppercase letters from last name as a heuristic
            parts = name_text.split()
            last_name = parts[-1] if parts else name_text
            abbrev = re.sub(r"[^A-Z]", "", last_name.upper())[:3] or last_name[:3].upper()

            try:
                number = int(re.sub(r"\D", "", num_text)) if num_text else 0
            except ValueError:
                number = 0

            rows.append(
                {
                    "DriverNumber": number,
                    "Abbreviation": abbrev,
                    "DriverName": name_text,
                    "TeamName": team_text,
                }
            )
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["DriverNumber", "Abbreviation", "TeamName"])

    df = pd.DataFrame(rows)
    # Drop duplicates just in case
    df = df.dropna(subset=["Abbreviation"]).drop_duplicates(subset=["Abbreviation"], keep="last")
    return df[["DriverNumber", "Abbreviation", "DriverName", "TeamName"]]

