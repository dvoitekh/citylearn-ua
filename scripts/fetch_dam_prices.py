"""Fetch Ukrainian day-ahead market (DAM) hourly prices from the Market Operator.

Downloads the official monthly XLS reports published at
https://www.oree.com.ua/index.php/pricectr ("Погодинні ціни купівлі-продажу
електроенергії", day-ahead market) and consolidates them into one tidy CSV:

    data/raw/oree_dam_<year>.csv   ->  date,h1..h24   (UAH/MWh)

The consolidated CSV is committed to the repository, so ``build_dataset.py``
works offline; re-run this script only to refresh or extend the archive.

Usage::

    python scripts/fetch_dam_prices.py [--year 2024] [--out data/raw]

Requires: requests, pandas, xlrd (the reports are legacy .xls).
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import pandas as pd
import requests
import xlrd

BASE = "https://www.oree.com.ua/index.php/pricectr"
GET_FILE = f"{BASE}/get_file"
UA = "Mozilla/5.0 (research; CityLearn-UA dataset build)"


def fetch_month(session: requests.Session, month: int, year: int) -> pd.DataFrame:
    """One month of hourly DAM prices as a date x h1..h24 frame (UAH/MWh)."""
    resp = session.post(
        GET_FILE,
        data={"price_date": f"{month:02d}.{year}", "market_type": "DAM", "zone": "1"},
        headers={"User-Agent": UA, "Referer": BASE},
        timeout=60,
    )
    resp.raise_for_status()
    if b"<!DOCTYP" in resp.content[:16]:
        raise RuntimeError(
            f"server returned HTML instead of a report for {month:02d}.{year} "
            "(usually transient rate limiting — retry in a few seconds)")
    # The reports are legacy .xls written by a generator that trips xlrd's
    # compound-document check; the payload itself is intact.
    book = xlrd.open_workbook(file_contents=resp.content,
                              ignore_workbook_corruption=True)
    sheet = book.sheet_by_index(0)

    # Layout: row 0 title, row 1 hour numbers, row 2 units, then one row per
    # day of the month with the day number in column 0.
    records = []
    for r in range(sheet.nrows):
        first = sheet.cell_value(r, 0)
        try:
            day = int(float(first))
        except (TypeError, ValueError):
            continue
        if not 1 <= day <= 31:
            continue
        values = []
        for c in range(1, 25):
            v = sheet.cell_value(r, c)
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                values.append(float("nan"))
        records.append([pd.Timestamp(year=year, month=month, day=day)] + values)

    if not records:
        raise ValueError(f"no day rows found in the report for {month:02d}.{year}")
    return pd.DataFrame(records,
                        columns=["date"] + [f"h{i}" for i in range(1, 25)])


def fetch_year(year: int, out_dir: str | Path = "data/raw") -> Path:
    session = requests.Session()
    session.get(BASE, headers={"User-Agent": UA}, timeout=60)  # session cookie
    frames = []
    for month in range(1, 13):
        for attempt in range(3):
            try:
                df = fetch_month(session, month, year)
                break
            except RuntimeError:
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))
        print(f"  {month:02d}.{year}: {len(df)} days")
        frames.append(df)
        time.sleep(2)  # be polite to the public service
    year_df = pd.concat(frames, ignore_index=True).sort_values("date")
    year_df["date"] = year_df["date"].dt.strftime("%Y-%m-%d")

    out = Path(out_dir) / f"oree_dam_{year}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    year_df.to_csv(out, index=False)
    hours = year_df.shape[0] * 24
    missing = int(year_df.iloc[:, 1:].isna().sum().sum())
    print(f"{out}: {len(year_df)} days / {hours} hours, {missing} missing values")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    fetch_year(args.year, args.out)
