from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import tempfile

import fitz
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import find_peaks

CALIBRATION_PULSE_MV = 1.0


@dataclass(frozen=True)
class ExtractionSettings:
    sample_rate_hz: int = 500
    segment_duration_s: float = 10.0
    gain_mm_per_mV: float = 10.0
    render_dpi: int = 300
    # Fractions of each detected strip height, measured from its top.
    baseline_mode: str = "second_grid"
    baseline_fractions: tuple[float, float, float] | None = None

    def validate(self) -> None:
        if self.baseline_mode not in ("second_grid", "auto"):
            raise ValueError("Unknown baseline mode.")
        if self.baseline_fractions is not None:
            if len(self.baseline_fractions) != 3 or any(
                not np.isfinite(v) or not 0 <= v <= 1 for v in self.baseline_fractions
            ):
                raise ValueError("Select one baseline within each of the three strips.")
        if not 50 <= self.sample_rate_hz <= 2000:
            raise ValueError("Sample rate must be between 50 and 2000 Hz.")
        if not 1 <= self.segment_duration_s <= 60:
            raise ValueError("Segment duration must be between 1 and 60 seconds.")
        if not 1 <= self.gain_mm_per_mV <= 50:
            raise ValueError("Gain must be between 1 and 50 mm/mV.")
        if not 100 <= self.render_dpi <= 600:
            raise ValueError("Render DPI must be between 100 and 600.")


@dataclass
class ExtractionResult:
    segments: list[pd.DataFrame]
    total: pd.DataFrame
    diagnostics: list[str]


def safe_base_name(value: str, fallback: str) -> str:
    value = value.strip() or fallback
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = value.rstrip(". ")
    if value.upper() in {
        "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        value = f"_{value}"
    if not value:
        raise ValueError("The output name cannot be empty.")
    return value


def extract_pdf(pdf_path: str | Path, settings: ExtractionSettings) -> ExtractionResult:
    settings.validate()
    page_rgb, effective_dpi, source_note = _load_first_page(
        Path(pdf_path), settings.render_dpi
    )
    page_rgb, waveform_mask, bands, rotation_note = _orient_page(page_rgb)

    px_per_mm_physical = effective_dpi / 25.4
    grid_values = []
    for y0, y1, x0, x1 in bands:
        grid = _estimate_small_grid_spacing(page_rgb[y0:y1, x0:x1])
        if grid is not None and 0.75 <= grid / px_per_mm_physical <= 1.25:
            grid_values.append(grid)
    grid_px = float(np.median(grid_values)) if grid_values else None
    grid_px_per_mV = grid_px * settings.gain_mm_per_mV if grid_px else None
    physical_px_per_mV = px_per_mm_physical * settings.gain_mm_per_mV
    pulse_height_px = _detect_calibration_pulse_px(
        page_rgb,
        bands[0],
        grid_px_per_mV or physical_px_per_mV,
    )
    if pulse_height_px is None:
        raise ValueError(
            "No reliable 1 mV calibration pulse was found on the first ECG strip. "
            "Voltage CSVs were not generated."
        )
    px_per_mV = pulse_height_px / CALIBRATION_PULSE_MV
    reference_px_per_mV = grid_px_per_mV or physical_px_per_mV
    if not 0.85 <= px_per_mV / reference_px_per_mV <= 1.15:
        raise ValueError(
            "The 1 mV calibration pulse disagrees with the ECG grid or the "
            "reported gain by more than 15%. Check the PDF and gain setting."
        )

    calibration_items = [f"PDF physical size {physical_px_per_mV:.2f} px/mV"]
    if grid_px_per_mV is not None:
        calibration_items.append(f"grid {grid_px_per_mV:.2f} px/mV")
    diagnostics = [
        f"{source_note}; {rotation_note}; detected 3 ECG strips; "
        f"processing resolution: {page_rgb.shape[1]} x {page_rgb.shape[0]} px.",
        f"Voltage scale: {pulse_height_px:.2f} px calibration pulse "
        f"= {CALIBRATION_PULSE_MV:.1f} mV; using {px_per_mV:.2f} px/mV. "
        + "Cross-check only: " + ", ".join(calibration_items) + ".",
    ]
    segments: list[pd.DataFrame] = []

    for index, (y0, y1, x0, x1) in enumerate(bands, start=1):
        mask_crop = waveform_mask[y0:y1, x0:x1]
        score_crop = _waveform_likelihood(page_rgb[y0:y1, x0:x1])
        x_px, y_px, valid_columns, coverage = _trace_waveform_subpixel(
            score_crop, mask_crop
        )

        sample_count = round(settings.sample_rate_hz * settings.segment_duration_s)
        target_x = np.arange(sample_count, dtype=float) * len(x_px) / sample_count
        sampled_y = np.interp(target_x, x_px, y_px)
        max_gap_px = max(
            1,
            round(
                len(x_px)
                / settings.segment_duration_s
                * 0.020
            ),
        )
        valid_columns = _fill_short_false_runs(valid_columns, max_gap_px)
        source_indices = np.clip(np.rint(target_x).astype(int), 0, len(x_px) - 1)
        valid_samples = valid_columns[source_indices]
        if np.count_nonzero(valid_samples) < sample_count * 0.35:
            raise ValueError(f"Segment {index} contains too little valid ECG data.")
        if settings.baseline_fractions is not None:
            baseline = settings.baseline_fractions[index - 1] * (y1 - y0 - 1)
            baseline_mode = "explicit reference from saved settings"
        elif settings.baseline_mode == "second_grid":
            baseline = _second_internal_major_grid_y(page_rgb, (y0, y1, x0, x1), reference_px_per_mV) - y0
            baseline_mode = "default second internal major grid line from bottom (border excluded)"
        else:
            baseline = _robust_baseline(sampled_y[valid_samples])
            baseline_mode = "automatic estimated baseline"
        diagnostics.append(f"Segment {index}: {baseline_mode}; zero line page y={y0 + baseline:.3f} px.")
        voltage = (baseline - sampled_y) / px_per_mV
        voltage[~valid_samples] = np.nan
        time_s = np.arange(sample_count, dtype=float) / settings.sample_rate_hz

        frame = pd.DataFrame({"time_s": time_s, "voltage_mV": voltage})
        segments.append(frame)
        missing_ranges = _missing_time_ranges(valid_samples, time_s)
        missing_note = (
            "no missing signal intervals"
            if not missing_ranges
            else "missing signal stored as NaN: " + ", ".join(missing_ranges)
        )
        diagnostics.append(
            f"Segment {index}: x={x0}-{x1}, y={y0}-{y1}; "
            f"trace coverage {coverage * 100:.1f}%; global subpixel path; "
            f"{missing_note}."
        )

    total_parts = []
    for i, frame in enumerate(segments):
        part = frame.copy()
        part["time_s"] += i * settings.segment_duration_s
        total_parts.append(part)
    total = pd.concat(total_parts, ignore_index=True)
    return ExtractionResult(segments=segments, total=total, diagnostics=diagnostics)


def write_four_csvs(
    result: ExtractionResult, output_dir: str | Path, base_name: str
) -> list[Path]:
    if len(result.segments) != 3:
        raise ValueError("Internal data must contain exactly 3 ECG segments.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [f"{base_name}_part{i}.csv" for i in range(1, 4)] + [
        f"{base_name}_total.csv"
    ]
    paths = [output_dir / name for name in names]
    temporary_paths: list[Path] = []
    try:
        for frame, path in zip([*result.segments, result.total], paths):
            handle, temporary_name = tempfile.mkstemp(
                prefix=f".{path.stem}_", suffix=".tmp", dir=output_dir
            )
            os.close(handle)
            temporary = Path(temporary_name)
            temporary_paths.append(temporary)
            frame.to_csv(
                temporary,
                index=False,
                float_format="%.6f",
                na_rep="",
            )
        for temporary, path in zip(temporary_paths, paths):
            os.replace(temporary, path)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    return paths


def _render_first_page(pdf_path: Path, dpi: int) -> np.ndarray:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    try:
        with fitz.open(pdf_path) as document:
            if document.page_count < 1:
                raise ValueError("The PDF has no pages.")
            page = document[0]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
            rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, pixmap.n
            )[:, :, :3]
    except fitz.FileDataError as exc:
        raise ValueError("Unable to read the PDF. It may be damaged or password-protected.") from exc
    return rgb.copy()


def _load_first_page(pdf_path: Path, render_dpi: int) -> tuple[np.ndarray, float, str]:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    try:
        with fitz.open(pdf_path) as document:
            if document.page_count < 1:
                raise ValueError("The PDF has no pages.")
            native = _assemble_native_image_tiles(document, document[0])
            if native is not None:
                image, dpi = native
                return image, dpi, f"assembled native embedded PDF images ({dpi:.1f} DPI)"
    except fitz.FileDataError as exc:
        raise ValueError("Unable to read the PDF. It may be damaged or password-protected.") from exc
    return (
        _render_first_page(pdf_path, render_dpi),
        float(render_dpi),
        f"native image assembly unavailable; rendered PDF at {render_dpi} DPI",
    )


def _assemble_native_image_tiles(
    document: fitz.Document, page: fitz.Page
) -> tuple[np.ndarray, float] | None:
    # If searchable text or vector drawings are overlaid on the images, direct
    # extraction would omit them; use the normal renderer in that case.
    if page.get_drawings() or page.get_text("words"):
        return None
    infos = [item for item in page.get_image_info(xrefs=True) if item.get("xref", 0)]
    if not infos:
        return None
    x_scales = []
    y_scales = []
    for item in infos:
        x0, y0, x1, y1 = item["bbox"]
        if x1 <= x0 or y1 <= y0:
            return None
        x_scales.append(item["width"] / (x1 - x0))
        y_scales.append(item["height"] / (y1 - y0))
    scale_x = float(np.median(x_scales))
    scale_y = float(np.median(y_scales))
    if (
        np.std(x_scales) / scale_x > 0.03
        or np.std(y_scales) / scale_y > 0.03
        or not 0.97 <= scale_x / scale_y <= 1.03
    ):
        return None
    height = round(page.rect.height * scale_y)
    width = round(page.rect.width * scale_x)
    if height <= 0 or width <= 0 or height * width > 120_000_000:
        return None
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    coverage = np.zeros((height, width), dtype=bool)
    for item in infos:
        pixmap = fitz.Pixmap(document, item["xref"])
        if pixmap.n - pixmap.alpha != 3:
            pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
        tile = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )[:, :, :3]
        x = round(item["bbox"][0] * scale_x)
        y = round(item["bbox"][1] * scale_y)
        copy_width = min(tile.shape[1], width - x)
        copy_height = min(tile.shape[0], height - y)
        if copy_width <= 0 or copy_height <= 0:
            continue
        canvas[y : y + copy_height, x : x + copy_width] = tile[
            :copy_height, :copy_width
        ]
        coverage[y : y + copy_height, x : x + copy_width] = True
    if coverage.mean() < 0.95:
        return None
    return canvas, (scale_x + scale_y) * 36.0


def _waveform_color_mask(image_rgb: np.ndarray) -> np.ndarray:
    warm = _waveform_likelihood(image_rgb) >= 0.16
    return ndimage.binary_closing(warm, structure=np.ones((3, 3), dtype=bool))


def _waveform_likelihood(image_rgb: np.ndarray) -> np.ndarray:
    rgb = image_rgb.astype(float) / 255.0
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = (maximum - minimum) / np.maximum(maximum, 1e-6)
    red_green = np.clip((red - green - 0.025) / 0.34, 0, 1)
    red_blue = np.clip((red - blue - 0.045) / 0.48, 0, 1)
    warm_balance = np.clip((green - blue * 0.45) / 0.35, 0, 1)
    saturation_score = np.clip((saturation - 0.08) / 0.45, 0, 1)
    score = np.minimum.reduce(
        [red_green, red_blue, warm_balance, saturation_score]
    )
    return score.astype(np.float32)


def _trace_waveform_subpixel(
    score: np.ndarray, support_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    height, width = score.shape
    if height < 3 or width < 10:
        raise ValueError("The ECG waveform region has invalid dimensions.")
    valid_columns = np.any(support_mask, axis=0)
    coverage = np.count_nonzero(valid_columns) / width
    if coverage < 0.35:
        raise ValueError("An ECG segment contains too little valid trace data.")

    # Global Viterbi-style path. The two distance-transform sweeps compute the
    # minimum previous cost with an absolute vertical-distance penalty in O(H).
    data_cost = -7.5 * ndimage.gaussian_filter(score, sigma=(0.65, 0.0))
    anchors = _visible_extrema_anchors(score)
    # Constrain the global path at visible raster extrema: continuity costs must
    # not allow shortcuts that truncate a narrow peak.
    for x, y in anchors.items():
        allowed = np.abs(np.arange(height)-y) <= 1.0
        data_cost[~allowed, x] = np.inf
    center = (height - 1) / 2
    previous = data_cost[:, 0] + 0.0015 * (np.arange(height) - center) ** 2
    parents = np.empty((width, height), dtype=np.int16 if height < 32767 else np.int32)
    parents[0] = np.arange(height)
    continuity_penalty = 0.075
    row_indices = np.arange(height, dtype=parents.dtype)

    for x in range(1, width):
        transformed = previous.copy()
        origin = row_indices.copy()
        for y in range(1, height):
            candidate = transformed[y - 1] + continuity_penalty
            if candidate < transformed[y]:
                transformed[y] = candidate
                origin[y] = origin[y - 1]
        for y in range(height - 2, -1, -1):
            candidate = transformed[y + 1] + continuity_penalty
            if candidate < transformed[y]:
                transformed[y] = candidate
                origin[y] = origin[y + 1]
        parents[x] = origin
        previous = transformed + data_cost[:, x]
        previous -= previous.min()

    path = np.empty(width, dtype=float)
    state = int(np.argmin(previous))
    for x in range(width - 1, -1, -1):
        path[x] = state
        if x:
            state = int(parents[x, state])

    # Refine the integer path with the color-probability centroid in a small
    # vertical neighborhood, yielding subpixel y coordinates.
    refined = path.copy()
    radius = 4
    for x, y_value in enumerate(path.astype(int)):
        y0 = max(0, y_value - radius)
        y1 = min(height, y_value + radius + 1)
        weights = score[y0:y1, x].astype(float)
        weight_sum = weights.sum()
        if weight_sum > 0.05:
            rows = np.arange(y0, y1, dtype=float)
            refined[x] = float(np.dot(rows, weights) / weight_sum)
    # No horizontal smoothing: it attenuates narrow extrema. Restore the
    # subpixel stroke-center anchors after neighborhood centroid refinement.
    for x, y in anchors.items():
        refined[x] = y
    return np.arange(width, dtype=float), refined, valid_columns, coverage



def _visible_extrema_anchors(score: np.ndarray) -> dict[int, float]:
    """Locate visible upper/lower turning points and account for stroke width.

    Colored envelope extrema are image evidence, not a prescribed ECG amplitude.
    The median width of ordinary thin columns estimates the raster stroke width.
    """
    colored = score >= 0.16
    labels, _ = ndimage.label(colored, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel())
    keep = sizes >= 8
    keep[0] = False
    colored &= keep[labels]
    rows = np.arange(score.shape[0])[:, None]
    valid = colored.any(axis=0)
    known = np.flatnonzero(valid)
    if len(known) < 3:
        return {}
    top = np.where(colored, rows, np.inf).min(axis=0)
    bottom = np.where(colored, rows, -np.inf).max(axis=0)
    thickness = bottom[valid]-top[valid]+1
    # Lower quartile uses near-horizontal strokes, not tall QRS columns.
    stroke_width = max(1., float(np.percentile(thickness, 25)))
    half_stroke = (stroke_width-1)/2
    prominence = max(4., stroke_width*1.5)
    xaxis = np.arange(score.shape[1])
    anchors = {}
    priorities = {}
    for envelope, sign in ((top, -1), (bottom, 1)):
        filled = np.interp(xaxis, known, envelope[known])
        peaks, props = find_peaks(sign*filled, prominence=prominence, distance=2, plateau_size=True)
        for x, strength, edge0, edge1 in zip(peaks, props['prominences'], props['left_edges'], props['right_edges']):
            if not valid[x]:
                continue
            # A confirmed local turning point needs supporting colored pixels
            # on both sides; gaps and isolated color noise do not anchor a peak.
            left, right = max(0, x-2), min(len(valid), x+3)
            if np.count_nonzero(valid[left:right]) < 3:
                continue
            y = envelope[x]-sign*half_stroke
            for xx in range(int(edge0), int(edge1)+1):
                if valid[xx] and (xx not in priorities or strength > priorities[xx]):
                    anchors[xx] = float(np.clip(y, top[xx], bottom[xx]))
                    priorities[xx] = strength
    return anchors

def _detect_calibration_pulse_px(
    image_rgb: np.ndarray,
    first_band: tuple[int, int, int, int],
    expected_px_per_mV: float,
) -> float | None:
    y0, y1, x0, x1 = first_band
    extra_top = max(20, int(image_rgb.shape[0] * 0.018))
    width = max(20, int((x1 - x0) * 0.15))
    crop = image_rgb[max(0, y0 - extra_top) : y1, x0 : x0 + width].astype(float)
    maximum = crop.max(axis=2)
    minimum = crop.min(axis=2)
    saturation = (maximum - minimum) / np.maximum(maximum, 1)
    neutral_dark = (crop.mean(axis=2) < 180) & (saturation < 0.18)
    labels, count = ndimage.label(neutral_dark, structure=np.ones((3, 3), dtype=bool))
    best: tuple[float, float] | None = None
    for label, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        y_slice, x_slice = slices
        component_height = y_slice.stop - y_slice.start
        component_width = x_slice.stop - x_slice.start
        area = int(np.count_nonzero(labels[slices] == label))
        if area < 40 or component_width < 8:
            continue
        ratio = component_height / max(expected_px_per_mV, 1e-6)
        if not 0.70 <= ratio <= 1.30:
            continue
        # Measure the centers of the two horizontal strokes, rather than the
        # outermost ink pixels. Bounding-box height includes stroke thickness.
        component = labels[slices] == label
        row_counts = np.count_nonzero(component, axis=1)
        horizontal_rows = np.flatnonzero(
            row_counts >= max(6, round(component_width * 0.25))
        )
        if len(horizontal_rows) < 2:
            continue
        strokes = _boolean_runs(np.isin(np.arange(component_height), horizontal_rows))
        if len(strokes) != 2:
            continue
        top_center = (strokes[0][0] + strokes[0][1] - 1) / 2
        bottom_center = (strokes[1][0] + strokes[1][1] - 1) / 2
        measured_height = bottom_center - top_center
        if not 0.70 <= measured_height / expected_px_per_mV <= 1.30:
            continue
        closeness = abs(np.log(ratio)) - min(area / 5000, 0.1)
        if best is None or closeness < best[0]:
            best = (closeness, float(measured_height))
    return None if best is None else best[1]


def _orient_page(
    page_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]], str]:
    candidates = []
    for rotations_ccw in range(4):
        image = np.rot90(page_rgb, rotations_ccw).copy()
        height, width = image.shape[:2]
        # The Samsung three-strip chart is landscape once displayed correctly.
        if width <= height:
            continue
        mask = _waveform_color_mask(image)
        try:
            rough_bands = _find_three_bands(mask)
        except ValueError:
            continue

        bands = []
        trace_score = 0.0
        try:
            for y0, y1, rough_x0, rough_x1 in rough_bands:
                chart_bounds = _find_chart_x_bounds(image[y0:y1])
                x0, x1 = chart_bounds or (rough_x0, rough_x1)
                _, _, coverage = _trace_waveform(mask[y0:y1, x0:x1])
                bands.append((y0, y1, x0, x1))
                trace_score += coverage + (x1 - x0) / width
        except ValueError:
            continue

        # Samsung prints a neutral gray calibration pulse at the left of strip 1.
        # It distinguishes the correct time direction from a 180-degree reversal.
        direction_score = _calibration_direction_score(image, bands[0])
        candidates.append((trace_score + direction_score, rotations_ccw, image, mask, bands))

    if not candidates:
        raise ValueError(
            "Unable to determine the ECG page orientation or detect three orange traces. "
            "Confirm that this is a three-strip Samsung Health Monitor ECG report."
        )

    _, rotations_ccw, image, mask, bands = max(candidates, key=lambda item: item[0])
    rotation_notes = {
        0: "page orientation unchanged",
        1: "page automatically rotated 90 degrees counterclockwise",
        2: "page automatically rotated 180 degrees",
        3: "page automatically rotated 90 degrees clockwise",
    }
    return image, mask, bands, rotation_notes[rotations_ccw]


def _find_chart_x_bounds(strip_rgb: np.ndarray) -> tuple[int, int] | None:
    gray = np.dot(strip_rgb[:, :, :3].astype(float), [0.299, 0.587, 0.114])
    # Horizontal grid lines make nearly every column inside the plot non-white.
    active = np.count_nonzero(gray < 245, axis=0) > max(5, int(strip_rgb.shape[0] * 0.06))
    runs = _boolean_runs(active)
    if not runs:
        return None
    x0, x1 = max(runs, key=lambda pair: pair[1] - pair[0])
    if x1 - x0 < strip_rgb.shape[1] * 0.35:
        return None
    return int(x0), int(x1)


def _calibration_direction_score(
    image_rgb: np.ndarray, band: tuple[int, int, int, int]
) -> float:
    y0, y1, x0, x1 = band
    crop = image_rgb[y0:y1, x0:x1].astype(float)
    maximum = crop.max(axis=2)
    minimum = crop.min(axis=2)
    saturation = (maximum - minimum) / np.maximum(maximum, 1)
    gray = crop.mean(axis=2)
    neutral_dark = (gray < 200) & (saturation < 0.18)
    edge_width = max(1, int(neutral_dark.shape[1] * 0.12))
    left = int(neutral_dark[:, :edge_width].sum())
    right = int(neutral_dark[:, -edge_width:].sum())
    return float(np.log1p(left) - np.log1p(right))


def _find_three_bands(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = mask.shape
    # Ignore page header/footer where colored logos and text can occur.
    top, bottom = int(height * 0.16), int(height * 0.94)
    usable = mask[top:bottom]
    row_coverage = np.count_nonzero(usable, axis=1)
    active = row_coverage > max(12, int(width * 0.006))
    groups = _boolean_runs(active)

    # Merge adjacent runs belonging to the same ECG strip.
    merge_gap = max(8, int(height * 0.025))
    merged: list[list[int]] = []
    for start, end in groups:
        start += top
        end += top
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    candidates = []
    for start, end in merged:
        # A long isoelectric baseline may occupy only a few rows while QRS peaks
        # extend much farther vertically, so reserve a page-relative strip height.
        pad_y = max(12, int(height * 0.045), int((end - start) * 0.35))
        y0, y1 = max(top, start - pad_y), min(bottom, end + pad_y + 1)
        column_counts = np.count_nonzero(mask[y0:y1], axis=0)
        cols = np.flatnonzero(column_counts >= 1)
        if not len(cols):
            continue
        x0, x1 = int(cols[0]), int(cols[-1] + 1)
        span = x1 - x0
        score = span * max(1, end - start)
        if span >= width * 0.35:
            candidates.append((score, y0, y1, x0, x1))

    if len(candidates) < 3:
        # Projection peaks are a more tolerant fallback for faint/thin traces.
        smooth = ndimage.gaussian_filter1d(row_coverage.astype(float), sigma=max(2, height / 250))
        min_distance = max(25, int(height * 0.12))
        peaks = _top_separated_indices(smooth, 3, min_distance)
        if len(peaks) == 3:
            half = int(height * 0.055)
            candidates = []
            for peak in peaks:
                center = peak + top
                y0, y1 = max(top, center - half), min(bottom, center + half)
                cols = np.flatnonzero(np.count_nonzero(mask[y0:y1], axis=0))
                if len(cols):
                    candidates.append((1, y0, y1, int(cols[0]), int(cols[-1] + 1)))

    if len(candidates) < 3:
        raise ValueError(
            "Unable to detect three orange ECG traces on PDF page 1. "
            "Confirm that this is a three-strip Samsung Health Monitor report and that the trace is not gray."
        )

    # Prefer the three broadest/highest-scoring bands, then restore time order.
    chosen = sorted(sorted(candidates, reverse=True)[:3], key=lambda item: item[1])
    bands = [(y0, y1, x0, x1) for _, y0, y1, x0, x1 in chosen]
    if any((x1 - x0) < width * 0.35 for y0, y1, x0, x1 in bands):
        raise ValueError("A detected ECG trace is too narrow for reliable digitization.")
    return bands


def _trace_waveform(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    height, width = mask.shape
    points = np.full(width, np.nan, dtype=float)
    previous = height / 2
    for x in range(width):
        ys = np.flatnonzero(mask[:, x])
        if not len(ys):
            continue
        # Choose the local cluster closest to the previous trace position. This
        # rejects grid labels and vertical calibration marks sharing the color.
        splits = np.split(ys, np.where(np.diff(ys) > 2)[0] + 1)
        centers = np.array([np.median(cluster) for cluster in splits])
        selected = float(centers[np.argmin(np.abs(centers - previous))])
        points[x] = selected
        previous = selected

    valid = np.flatnonzero(np.isfinite(points))
    if len(valid) < width * 0.35:
        raise ValueError("An ECG segment contains too little valid trace data.")
    coverage = len(valid) / width
    # This provisional path is used only for page-orientation scoring. Final
    # extraction preserves unsupported columns with a separate validity mask.
    x = np.arange(width, dtype=float)
    y = np.interp(x, valid, points[valid])
    y = ndimage.gaussian_filter1d(y, sigma=0.45)
    return x, y.astype(float), coverage


def _robust_baseline(samples: np.ndarray) -> float:
    # The center of the densest amplitude bin approximates the isoelectric line
    # and is less affected by tall R peaks than a mean.
    lo, hi = np.percentile(samples, [5, 95])
    if hi <= lo:
        return float(np.median(samples))
    counts, edges = np.histogram(samples, bins=80, range=(lo, hi))
    i = int(np.argmax(counts))
    return float((edges[i] + edges[i + 1]) / 2)


def _fill_short_false_runs(values: np.ndarray, maximum_gap: int) -> np.ndarray:
    result = values.astype(bool).copy()
    for start, end in _boolean_runs(~result):
        if start > 0 and end < len(result) and end - start <= maximum_gap:
            result[start:end] = True
    return result


def _missing_time_ranges(valid: np.ndarray, time_s: np.ndarray) -> list[str]:
    if len(time_s) < 2:
        return []
    step = float(np.median(np.diff(time_s)))
    ranges = []
    for start, end in _boolean_runs(~valid):
        ranges.append(f"{time_s[start]:.3f}-{time_s[end - 1] + step:.3f} s")
    return ranges


def _estimate_small_grid_spacing(crop_rgb: np.ndarray) -> float | None:
    gray = np.dot(crop_rgb[:, :, :3].astype(float), [0.299, 0.587, 0.114])
    # Grid lines produce periodic dips in average brightness. Remove slow lighting
    # changes and look for a consistent period in both directions.
    signals = [255 - gray.mean(axis=0), 255 - gray.mean(axis=1)]
    estimates = []
    for signal in signals:
        signal = signal.astype(float)
        slow_sigma = max(3.0, len(signal) / 80)
        slow = ndimage.gaussian_filter1d(signal, sigma=slow_sigma)
        centered = signal - slow
        centered -= centered.mean()
        if np.std(centered) < 0.3:
            continue
        corr = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
        # At 100-600 DPI, one physical millimeter is roughly 4-24 pixels.
        min_lag, max_lag = 4, min(28, len(corr) // 8)
        if max_lag <= min_lag:
            continue
        lag = min_lag + int(np.argmax(corr[min_lag : max_lag + 1]))
        strength = corr[lag] / max(corr[0], 1e-9)
        if strength > 0.12:
            estimates.append(float(lag))
    if not estimates:
        return None
    if len(estimates) == 2 and max(estimates) / min(estimates) > 1.35:
        return None
    return float(np.median(estimates))


def _boolean_runs(values: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, values, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _top_separated_indices(values: np.ndarray, count: int, distance: int) -> list[int]:
    order = np.argsort(values)[::-1]
    chosen: list[int] = []
    for item in order:
        item = int(item)
        if values[item] <= 0:
            break
        if all(abs(item - other) >= distance for other in chosen):
            chosen.append(item)
            if len(chosen) == count:
                break
    return sorted(chosen)


def _second_internal_major_grid_y(image, band, reference_px_per_mV):
    """Find the bottom border and major-grid period; zero is two periods above it.

    Minor grid lines are deliberately excluded. The target may be partially
    obscured by the orange ECG trace, so its lattice position is verified locally.
    """
    y0, y1, x0, x1 = band
    margin = max(5, round((x1-x0)*.05))
    crop = image[y0:y1, x0+margin:x1-margin].astype(float)
    gray = crop.mean(axis=2)
    neutral = crop.max(axis=2)-crop.min(axis=2) < 18
    strength = ((gray < 230) & neutral).mean(axis=1)
    runs = _boolean_runs(strength > .60)
    if len(runs) < 2:
        raise ValueError("Cannot identify the bottom major grid lines. No default zero reference was set.")
    centers = np.array([(a+b-1)/2 for a,b in runs])
    bottom, previous = centers[-1], centers[-2]
    period = bottom-previous
    if not .40 <= period/reference_px_per_mV <= .60:
        raise ValueError("Bottom grid spacing is inconsistent with a 0.5 mV major square.")
    target = bottom-2*period
    radius = max(2, round(period*.06))
    lo, hi = max(0, int(np.floor(target))-radius), min(len(strength), int(np.ceil(target))+radius+1)
    supported = _boolean_runs(strength[lo:hi] > .15)
    if not supported:
        raise ValueError("The second internal major horizontal line cannot be verified.")
    choices = [lo+(a+b-1)/2 for a,b in supported]
    zero = min(choices, key=lambda v: abs(v-target))
    if abs(zero-target) > radius:
        raise ValueError("The default zero reference is inconsistent with the grid lattice.")
    return float(y0+zero)
