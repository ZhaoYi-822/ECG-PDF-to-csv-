from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator, FuncFormatter


def calculate_full_limits(
    frame: pd.DataFrame,
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min = float(frame["time_s"].min())
    x_max = float(frame["time_s"].max())
    valid_voltage = frame["voltage_mV"].dropna().to_numpy(dtype=float)
    y_min = float(np.min(valid_voltage))
    y_max = float(np.max(valid_voltage))
    x_padding = max((x_max - x_min) * 0.01, 0.002)
    y_padding = max((y_max - y_min) * 0.10, 0.05)
    return (x_min - x_padding, x_max + x_padding), (
        y_min - y_padding,
        y_max + y_padding,
    )


def format_ecg_axes(axes, title: str, background: str) -> None:
    axes.set_title(title)
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("Voltage (mV)")
    axes.minorticks_on()
    if background == "ECG Grid":
        axes.xaxis.set_major_locator(MultipleLocator(0.2))
        axes.xaxis.set_minor_locator(MultipleLocator(0.04))
        axes.yaxis.set_major_locator(MultipleLocator(0.5))
        axes.yaxis.set_minor_locator(MultipleLocator(0.1))
        axes.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}" if abs(x-round(x)) < 1e-7 else ""))
        axes.set_facecolor("#fffafa")
        axes.grid(which="major", color="#e9a8a8", linewidth=0.75, alpha=0.75)
        axes.grid(which="minor", color="#f4cccc", linewidth=0.4, alpha=0.65)
    else:
        axes.set_facecolor("white")
        from matplotlib.ticker import AutoLocator, AutoMinorLocator, ScalarFormatter
        axes.xaxis.set_major_locator(AutoLocator())
        axes.xaxis.set_minor_locator(AutoMinorLocator())
        axes.yaxis.set_major_locator(AutoLocator())
        axes.yaxis.set_minor_locator(AutoMinorLocator())
        axes.xaxis.set_major_formatter(ScalarFormatter())
        axes.grid(False, which="both")


def save_ecg_figure(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    image_format: str,
    dpi: int,
    width: float,
    height: float,
    title: str,
    line_color: str,
    line_width: float,
    background: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    figure = Figure(figsize=(width, height), dpi=dpi)
    axes = figure.add_subplot(111)
    axes.plot(
        frame["time_s"],
        frame["voltage_mV"],
        color=line_color,
        linewidth=line_width,
    )
    format_ecg_axes(axes, title, background)
    axes.set_xlim(x_limits)
    axes.set_ylim(y_limits)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, format=image_format)
