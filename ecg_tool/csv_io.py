from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("time_s", "voltage_mV")


def read_ecg_csv(path: str | Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Unable to read CSV: {exc}") from exc
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError("CSV must contain time_s and voltage_mV columns.")
    frame = frame.loc[:, REQUIRED_COLUMNS].copy()
    original_voltage = frame["voltage_mV"].copy()
    frame["time_s"] = pd.to_numeric(frame["time_s"], errors="coerce")
    frame["voltage_mV"] = pd.to_numeric(frame["voltage_mV"], errors="coerce")
    if frame["time_s"].isna().any():
        raise ValueError("time_s cannot contain missing or non-numeric values.")
    invalid_voltage = frame["voltage_mV"].isna() & original_voltage.notna()
    if invalid_voltage.any():
        raise ValueError("voltage_mV may be blank, but cannot contain non-numeric text.")
    if len(frame) < 2:
        raise ValueError("CSV must contain at least two data rows.")
    time = frame["time_s"].to_numpy()
    if not np.all(np.diff(time) > 0):
        raise ValueError("time_s must be strictly increasing.")
    if frame["voltage_mV"].notna().sum() < 2:
        raise ValueError("CSV must contain at least two valid voltage values.")
    return frame
